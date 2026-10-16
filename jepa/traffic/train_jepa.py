from __future__ import annotations
import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
import torch
from torch import nn
from jepa.common import (
    atomic_save_new,
    atomic_write_json_new,
    configure_cpu_threads,
    cpu_state_dict,
    dataset_file_hashes,
    load_json_object,
    resolve_device,
    runtime_contract,
    set_seed,
    source_hashes,
    strict_dataclass_mapping,
)
from jepa.data import load_cifar10_python_batches
from jepa.four_image.feed import identity_pool_sha256
from jepa.four_image.models import FourImageJEPA, parameter_count
from jepa.losses import SIGReg, latent_collapse_metrics, prediction_mse
from .feed import (
    ReplacementFeedConfig,
    ReplacementSubsidyFeed,
    batch_statistics,
    select_identity_pool,
)


FORMAT_VERSION = 1
CHECKPOINT_KIND = "jepa_traffic_light_two_image_standard_checkpoint"
PREDICTION_WEIGHT = 0.9
SIGREG_WEIGHT = 0.1


@dataclass(frozen=True)
class Config:
    format_version: int
    run_id: str
    data_root: str
    output_checkpoint: str
    output_json: str
    seed: int
    train_sequence_seed: int
    evaluation_sequence_seed: int
    evaluation_sigreg_seed: int
    feed: ReplacementFeedConfig
    latent_dim: int
    encoder_base_channels: int
    encoder_representation_width: int
    encoder_frame_chunk_size: int
    predictor_hidden_dim: int
    epochs: int
    sequences_per_epoch: int
    batch_size: int
    learning_rate: float
    minimum_learning_rate: float
    warmup_steps: int
    weight_decay: float
    gradient_clip: float
    prediction_weight: float
    sigreg_weight: float
    sigreg_knots: int
    sigreg_projections: int
    sigreg_chunk_size: int
    bf16: bool
    log_every: int
    evaluation_every: int
    evaluation_batch_size: int
    evaluation_batches: int
    cpu_threads: int
    cpu_interop_threads: int
    device: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "Config":
        normalized = strict_dataclass_mapping(
            cls, value, "traffic-light JEPA config"
        )
        normalized["feed"] = ReplacementFeedConfig(**normalized["feed"])
        config = cls(**normalized)
        config.validate()
        return config

    @property
    def steps_per_epoch(self) -> int:
        return math.ceil(self.sequences_per_epoch / self.batch_size)

    @property
    def steps(self) -> int:
        return self.epochs * self.steps_per_epoch

    def validate(self) -> None:
        if self.format_version != FORMAT_VERSION:
            raise ValueError(f"format_version must equal {FORMAT_VERSION}")
        if Path(self.run_id).name != self.run_id or not self.run_id:
            raise ValueError("run_id must be one filename-safe component")
        for name in (
            "seed",
            "train_sequence_seed",
            "evaluation_sequence_seed",
            "evaluation_sigreg_seed",
        ):
            value = getattr(self, name)
            if type(value) is not int or not 0 <= value < 2**63:
                raise ValueError(f"{name} must be an integer in [0,2^63)")
        for name in (
            "latent_dim",
            "encoder_base_channels",
            "encoder_representation_width",
            "encoder_frame_chunk_size",
            "predictor_hidden_dim",
            "epochs",
            "sequences_per_epoch",
            "batch_size",
            "warmup_steps",
            "sigreg_knots",
            "sigreg_projections",
            "sigreg_chunk_size",
            "log_every",
            "evaluation_every",
            "evaluation_batch_size",
            "evaluation_batches",
            "cpu_threads",
            "cpu_interop_threads",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.warmup_steps >= self.steps:
            raise ValueError("warmup_steps must be smaller than total steps")
        for name in (
            "learning_rate",
            "minimum_learning_rate",
            "weight_decay",
            "gradient_clip",
            "prediction_weight",
            "sigreg_weight",
        ):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if (self.prediction_weight, self.sigreg_weight) not in (
            (0.95, 0.05), (0.90, 0.10), (0.80, 0.20)
        ):
            raise ValueError("Only prediction/SIGReg pairs are allowed")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if not 0.0 <= self.minimum_learning_rate <= self.learning_rate:
            raise ValueError("minimum_learning_rate is invalid")
        if self.weight_decay < 0.0 or self.gradient_clip <= 0.0:
            raise ValueError("optimizer scalars are invalid")
        if type(self.bf16) is not bool:
            raise TypeError("bf16 must be boolean")


def learning_rate_at_step(config: Config, step_index: int) -> float:
    if step_index < config.warmup_steps:
        return config.learning_rate * (step_index + 1) / config.warmup_steps
    progress = (step_index - config.warmup_steps) / max(
        1, config.steps - config.warmup_steps - 1
    )
    return config.minimum_learning_rate + 0.5 * (
        config.learning_rate - config.minimum_learning_rate
    ) * (1.0 + math.cos(math.pi * progress))


def collapse_summary(latents: torch.Tensor) -> dict[str, Any]:
    return {
        "flattened": asdict(latent_collapse_metrics(latents)),
        "per_time": [asdict(latent_collapse_metrics(values)) for values in latents],
    }


def current_source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    return source_hashes({str(p.relative_to(root)): p for p in sorted(root.rglob("*.py"))})


@torch.no_grad()
def evaluate(
    model: FourImageJEPA,
    sigreg: SIGReg,
    feed: ReplacementSubsidyFeed,
    config: Config,
    device: torch.device,
) -> dict[str, Any]:
    was_training = model.training
    model.eval()
    generator = torch.Generator(device="cpu").manual_seed(
        config.evaluation_sequence_seed
    )
    prediction_values = []
    sigreg_values = []
    collapse_values = []
    statistics = []
    devices = (
        [device.index if device.index is not None else torch.cuda.current_device()]
        if device.type == "cuda"
        else []
    )
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(config.evaluation_sigreg_seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(config.evaluation_sigreg_seed)
        for _ in range(config.evaluation_batches):
            batch = feed.sample(
                config.evaluation_batch_size,
                generator=generator,
                device=device,
            )
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=config.bf16,
            ):
                latents, predictions = model(batch.frames)
            prediction_values.append(
                float(
                    prediction_mse(
                        predictions.float(),
                        latents[1:].float(),
                        expected_time_steps=config.feed.sequence_length - 1,
                    )
                )
            )
            sigreg_values.append(
                float(
                    sigreg.forward_chunked(
                        latents.float(), chunk_size=config.sigreg_chunk_size
                    )
                )
            )
            collapse_values.append(collapse_summary(latents.float()))
            statistics.append(batch_statistics(batch))
    model.train(was_training)
    prediction_mean = sum(prediction_values) / len(prediction_values)
    sigreg_mean = sum(sigreg_values) / len(sigreg_values)
    return {
        "prediction_mse": prediction_mean,
        "sigreg": sigreg_mean,
        "total_objective": config.prediction_weight * prediction_mean
        + config.sigreg_weight * sigreg_mean,
        "latent_collapse_per_batch": collapse_values,
        "feed_statistics_per_batch": statistics,
    }


def run(config: Config) -> dict[str, Any]:
    config.validate()
    checkpoint_path = Path(config.output_checkpoint).expanduser()
    json_path = Path(config.output_json).expanduser()
    if checkpoint_path.exists() or json_path.exists():
        raise FileExistsError("refusing to overwrite traffic-light outputs")
    configure_cpu_threads(config.cpu_threads, config.cpu_interop_threads)
    device = resolve_device(config.device, bf16=config.bf16)
    set_seed(config.seed)
    root, dataset_hashes = dataset_file_hashes(config.data_root)
    full_pool = load_cifar10_python_batches(root, split="train")
    pool, source_indices = select_identity_pool(
        full_pool,
        identity_count=config.feed.identity_count,
        seed=config.feed.identity_subset_seed,
    )
    feed = ReplacementSubsidyFeed(pool, config.feed)
    model = FourImageJEPA(
        sequence_length=config.feed.sequence_length,
        latent_dim=config.latent_dim,
        base_channels=config.encoder_base_channels,
        representation_width=config.encoder_representation_width,
        predictor_hidden_dim=config.predictor_hidden_dim,
        encoder_frame_chunk_size=config.encoder_frame_chunk_size,
    ).to(device)
    sigreg = SIGReg(
        knots=config.sigreg_knots,
        num_projections=config.sigreg_projections,
        expected_time_steps=config.feed.sequence_length,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    counts = {
        "encoder": parameter_count(model.encoder),
        "predictor": parameter_count(model.predictor),
        "total": parameter_count(model),
    }
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    generator = torch.Generator(device="cpu").manual_seed(config.train_sequence_seed)
    print(
        json.dumps(
            {
                "event": "start",
                "run_id": config.run_id,
                "q": config.feed.visible_mode_probability,
                "objective": f"{config.prediction_weight} * prediction_mse + {config.sigreg_weight} * SIGReg",
                "target_detached": False,
                "additional_latent_regularizers": [],
                "parameter_counts": counts,
                "steps": config.steps,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    evaluations = [{"step": 0, "evaluation": evaluate(model, sigreg, feed, config, device)}]
    curve = []
    raw_examples = None
    step_index = 0
    model.train()
    for epoch_index in range(config.epochs):
        for batch_index in range(config.steps_per_epoch):
            start = batch_index * config.batch_size
            actual_batch_size = min(
                config.batch_size, config.sequences_per_epoch - start
            )
            if actual_batch_size <= 0:
                raise AssertionError("invalid final batch size")
            batch = feed.sample(
                actual_batch_size,
                generator=generator,
                device=device,
            )
            if raw_examples is None:
                raw_examples = {
                    "image_states": batch.image_states[:2].cpu().tolist(),
                    "lights": batch.lights[:2].cpu().tolist(),
                    "modes": batch.modes[:2].cpu().tolist(),
                    "use_visible_mode": batch.use_visible_mode[:2].cpu().tolist(),
                }
                print(
                    json.dumps(
                        {"event": "raw_data_examples", **raw_examples},
                        sort_keys=True,
                    ),
                    flush=True,
                )
            learning_rate = learning_rate_at_step(config, step_index)
            optimizer.param_groups[0]["lr"] = learning_rate
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=config.bf16,
            ):
                latents, predictions = model(batch.frames)
            prediction_loss = prediction_mse(
                predictions.float(),
                latents[1:].float(),
                expected_time_steps=config.feed.sequence_length - 1,
            )
            with torch.autocast(device_type=device.type, enabled=False):
                sigreg_loss = sigreg.forward_chunked(
                    latents.float(), chunk_size=config.sigreg_chunk_size
                )
            loss = (
                config.prediction_weight * prediction_loss
                + config.sigreg_weight * sigreg_loss
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite standard JEPA loss")
            loss.backward()
            gradient_norm = nn.utils.clip_grad_norm_(
                model.parameters(),
                config.gradient_clip,
                error_if_nonfinite=True,
            )
            optimizer.step()
            step_index += 1
            if (
                step_index == 1
                or step_index % config.log_every == 0
                or step_index == config.steps
            ):
                record = {
                    "step": step_index,
                    "epoch": epoch_index + 1,
                    "learning_rate": learning_rate,
                    "total_loss": float(loss.detach()),
                    "prediction_mse": float(prediction_loss.detach()),
                    "sigreg": float(sigreg_loss.detach()),
                    "gradient_norm": float(gradient_norm.detach()),
                    "latent_collapse": collapse_summary(latents.detach().float()),
                    "feed_statistics": batch_statistics(batch),
                }
                curve.append(record)
                print(
                    json.dumps({"event": "train", **record}, sort_keys=True),
                    flush=True,
                )
            if (
                step_index % config.evaluation_every == 0
                or step_index == config.steps
            ):
                evaluation = evaluate(model, sigreg, feed, config, device)
                evaluations.append({"step": step_index, "evaluation": evaluation})
                print(
                    json.dumps(
                        {
                            "event": "evaluation",
                            "step": step_index,
                            "prediction_mse": evaluation["prediction_mse"],
                            "sigreg": evaluation["sigreg"],
                            "total_objective": evaluation["total_objective"],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                model.train()
    if step_index != config.steps or raw_examples is None:
        raise AssertionError("training did not complete its exact contract")
    runtime = runtime_contract(device)
    if device.type == "cuda":
        runtime["peak_gpu_memory_allocated_bytes"] = int(
            torch.cuda.max_memory_allocated(device)
        )
    checkpoint = {
        "format_version": FORMAT_VERSION,
        "artifact_kind": CHECKPOINT_KIND,
        "config": asdict(config),
        "model_state_dict": cpu_state_dict(model),
        "parameter_counts": counts,
        "objective_contract": {
            "formula": f"{config.prediction_weight} * prediction_mse + {config.sigreg_weight} * SIGReg",
            "prediction_weight": config.prediction_weight,
            "sigreg_weight": config.sigreg_weight,
            "additional_latent_regularizers": [],
            "target_detached": False,
        },
        "dataset": {
            "root": str(root),
            "file_sha256": dataset_hashes,
            "split": "official_train_balanced_identity_subset",
            "identity_count": int(pool.images.shape[0]),
            "identity_source_indices_sha256": identity_pool_sha256(source_indices),
        },
        "transition": {
            "construction": "visible_or_hidden_two_way_selector_replaces_exactly_one_image",
            "visible_mode_probability": config.feed.visible_mode_probability,
            "next_light_independent": True,
            "shared_marginal_image_dynamics_across_q": True,
        },
        "data_volume": {
            "epochs": config.epochs,
            "sequences_per_epoch": config.sequences_per_epoch,
            "trained_sequences": config.epochs * config.sequences_per_epoch,
            "steps": config.steps,
        },
        "raw_data_examples": raw_examples,
        "source_hashes": current_source_hashes(),
        "runtime": runtime,
        "completed_steps": step_index,
        "completed_epochs": config.epochs,
        "learning_curve": curve,
        "evaluation_curve": evaluations,
    }
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_save_new(checkpoint, checkpoint_path)
    atomic_write_json_new(
        {key: value for key, value in checkpoint.items() if key != "model_state_dict"},
        json_path,
    )
    digest = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    print(
        json.dumps(
            {"event": "complete", "run_id": config.run_id, "checkpoint_sha256": digest},
            sort_keys=True,
        ),
        flush=True,
    )
    return checkpoint


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    run(Config.from_mapping(load_json_object(args.config.expanduser())))


if __name__ == "__main__":
    main()
