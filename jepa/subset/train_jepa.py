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
from jepa.losses import SIGReg, latent_collapse_metrics, prediction_mse
from .feed import (
    FRAME_SHAPE,
    RingBudgetFeed,
    RingBudgetFeedConfig,
    batch_statistics,
    bd_tile_coordinates_yx,
    pool_contract,
    position_names,
    ring_budget_feed_config_from_mapping,
    select_ring_pool,
    source_position_for_target,
)
from .models import RingBudgetJEPA, parameter_count


FORMAT_VERSION = 1
CHECKPOINT_KIND = "jepa_ring_budget_standard_checkpoint"


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
    feed: RingBudgetFeedConfig
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
        normalized = strict_dataclass_mapping(cls, value, "ring-budget JEPA config")
        normalized["feed"] = ring_budget_feed_config_from_mapping(normalized["feed"])
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
            scalar = getattr(self, name)
            if type(scalar) is not int or not 0 <= scalar < 2**63:
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
            raise ValueError("warmup_steps must be smaller than the total steps")
        if self.sigreg_knots < 2:
            raise ValueError("sigreg_knots must be at least two")
        if self.batch_size < 2 or self.evaluation_batch_size < 2:
            raise ValueError("SIGReg requires training and evaluation batches of at least two")
        if self.sequences_per_epoch % self.batch_size == 1:
            raise ValueError("the final training batch cannot contain one sequence")
        if self.train_sequence_seed == self.evaluation_sequence_seed:
            raise ValueError("training and evaluation sequence seeds must differ")
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
        if not any(math.isclose(self.prediction_weight,p,abs_tol=1e-12) and math.isclose(self.sigreg_weight,r,abs_tol=1e-12) for p,r in ((.95,.05),(.9,.1),(.8,.2))):
            raise ValueError("Only the prediction/SIGReg pairs are permitted")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if not 0.0 <= self.minimum_learning_rate <= self.learning_rate:
            raise ValueError("minimum_learning_rate is invalid")
        if self.weight_decay < 0.0 or self.gradient_clip <= 0.0:
            raise ValueError("optimizer scalars are invalid")
        if type(self.bf16) is not bool:
            raise TypeError("bf16 must be boolean")


def learning_rate_at_step(config: Config, step_index: int) -> float:
    if not 0 <= step_index < config.steps:
        raise ValueError("step_index lies outside the configured schedule")
    if step_index < config.warmup_steps:
        return config.learning_rate * (step_index + 1) / config.warmup_steps
    progress = (step_index - config.warmup_steps) / max(
        1, config.steps - config.warmup_steps - 1
    )
    return config.minimum_learning_rate + 0.5 * (
        config.learning_rate - config.minimum_learning_rate
    ) * (1.0 + math.cos(math.pi * progress))


def collapse_summary(latents: torch.Tensor) -> dict[str, Any]:
    if latents.ndim != 3:
        raise ValueError("latents must be time-major")
    per_time = [asdict(latent_collapse_metrics(values)) for values in latents]
    flattened = asdict(latent_collapse_metrics(latents))
    return {
        "per_time": per_time,
        "flattened": flattened,
        "minimum_per_time_effective_rank": min(row["effective_rank"] for row in per_time),
        "minimum_per_time_mean_variance": min(row["mean_variance"] for row in per_time),
    }


def _source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    return source_hashes({str(p.relative_to(root)): p for p in sorted(root.rglob("*.py"))})


@torch.no_grad()
def evaluate(
    model: RingBudgetJEPA,
    sigreg: SIGReg,
    feed: RingBudgetFeed,
    config: Config,
    device: torch.device,
) -> dict[str, Any]:
    was_training = model.training
    model.eval()
    data_generator = torch.Generator(device="cpu").manual_seed(config.evaluation_sequence_seed)
    devices = (
        [device.index if device.index is not None else torch.cuda.current_device()]
        if device.type == "cuda"
        else []
    )
    prediction_values = []
    sigreg_values = []
    collapse_values = []
    feed_values = []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(config.evaluation_sigreg_seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(config.evaluation_sigreg_seed)
        for _ in range(config.evaluation_batches):
            batch = feed.sample(
                config.evaluation_batch_size,
                generator=data_generator,
                device=device,
            )
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=config.bf16):
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
                float(sigreg.forward_chunked(latents.float(), chunk_size=config.sigreg_chunk_size))
            )
            collapse_values.append(collapse_summary(latents.float()))
            feed_values.append(batch_statistics(batch, config.feed))
    model.train(was_training)
    prediction_mean = sum(prediction_values) / len(prediction_values)
    sigreg_mean = sum(sigreg_values) / len(sigreg_values)
    return {
        "prediction_mse": prediction_mean,
        "sigreg": sigreg_mean,
        "total_objective": (
            config.prediction_weight * prediction_mean + config.sigreg_weight * sigreg_mean
        ),
        "per_batch_prediction_mse": prediction_values,
        "per_batch_sigreg": sigreg_values,
        "latent_collapse_per_batch": collapse_values,
        "feed_statistics_per_batch": feed_values,
    }


def _state_artifact(
    *,
    artifact_kind: str,
    config: Config,
    model: RingBudgetJEPA,
    counts: Mapping[str, int],
    dataset: Mapping[str, Any],
    pool: Mapping[str, Any],
    completed_epoch: int,
    completed_steps: int,
    curve: list[dict[str, Any]],
    evaluations: list[dict[str, Any]],
    raw_examples: Mapping[str, Any],
    source_manifest: Mapping[str, str],
) -> dict[str, Any]:
    names = position_names(config.feed.bd_ring_size)
    sources = source_position_for_target(config.feed.bd_ring_size)
    return {
        "format_version": FORMAT_VERSION,
        "artifact_kind": artifact_kind,
        "config": asdict(config),
        "model_state_dict": cpu_state_dict(model),
        "parameter_counts": dict(counts),
        "objective_contract": {
            "formula": f"{config.prediction_weight} * prediction_mse + {config.sigreg_weight} * SIGReg",
            "prediction_weight": config.prediction_weight,
            "sigreg_weight": config.sigreg_weight,
            "additional_latent_regularizers": [],
            "target_detached": False,
        },
        "dataset": dict(dataset),
        "identity_pool": dict(pool),
        "dynamics": {
            "state": "exact official CIFAR-10 training-image identity at each logical position",
            "position_order": list(names),
            "source_position_for_target": {
                target: names[source] for target, source in zip(names, sources, strict=True)
            },
            "cycles": [list(names[:2]), list(names[2:])],
            "copy_probability_ac": config.feed.q_ac,
            "copy_probability_bd": config.feed.q_bd,
            "copy_probability_bd_by_target_position": {
                name: probability
                for name, probability in zip(
                    names[2:], config.feed.bd_probabilities, strict=True
                )
            },
            "failed_copy_rule": "uniform random identity from the same shared pool",
            "arbitrary_identity_mapping_or_dictionary": False,
            "successful_branch": "direct exact-identity copy",
            "random_stream_contract": {
                "shared_six_position_random_stream": True,
                "additional_positions_beyond_six": (
                    "deterministic side stream derived from the supplied generator entry state"
                ),
                "supplied_generator_consumption_matched_across_ring_sizes": True,
            },
        },
        "renderer": {
            "frame_shape": list(FRAME_SHAPE),
            "final_pooling": "AdaptiveMaxPool2d(4,4); no trainable parameters",
            "ac_tile_size": 32,
            "ac_tile_coordinates_yx": [[16, 16], [80, 16]],
            "bd_ring_size": config.feed.bd_ring_size,
            "bd_tile_size": config.feed.render_tile_size,
            "bd_tile_size_override": config.feed.bd_render_tile_size,
            "bd_layout": config.feed.bd_layout,
            "bd_tile_coordinates_yx": [
                list(coordinates)
                for coordinates in bd_tile_coordinates_yx(
                    config.feed.bd_ring_size,
                    config.feed.bd_render_tile_size,
                    layout=config.feed.bd_layout,
                )
            ],
            "upsampling": {
                "mode": "bilinear",
                "align_corners": False,
                "bd_native_32_pixel_tiles_are_not_resampled": True,
                "ac_native_32_pixel_tiles_are_not_resampled": True,
            },
            "background": "black",
        },
        "data_volume": {
            "epochs": config.epochs,
            "sequences_per_epoch": config.sequences_per_epoch,
            "steps_per_epoch": config.steps_per_epoch,
            "configured_steps": config.steps,
            "completed_sequences": completed_epoch * config.sequences_per_epoch,
            "completed_collage_frames": (
                completed_epoch * config.sequences_per_epoch * config.feed.sequence_length
            ),
        },
        "raw_data_examples": dict(raw_examples),
        "source_hashes": dict(source_manifest),
        "completed_steps": completed_steps,
        "completed_epochs": completed_epoch,
        "learning_curve": list(curve),
        "evaluation_curve": list(evaluations),
    }


def run(config: Config) -> dict[str, Any]:
    config.validate()
    source_manifest = _source_hashes()
    checkpoint_path = Path(config.output_checkpoint).expanduser()
    json_path = Path(config.output_json).expanduser()
    existing = [path for path in (checkpoint_path, json_path) if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite outputs: {existing}")
    configure_cpu_threads(config.cpu_threads, config.cpu_interop_threads)
    device = resolve_device(config.device, bf16=config.bf16)
    set_seed(config.seed)
    root, dataset_hashes = dataset_file_hashes(config.data_root)
    full_pool = load_cifar10_python_batches(root, split="train")
    selected_pool = select_ring_pool(full_pool, config.feed)
    feed = RingBudgetFeed(selected_pool, config.feed)
    model = RingBudgetJEPA(
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
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    counts = {
        "encoder": parameter_count(model.encoder),
        "predictor": parameter_count(model.predictor),
        "total": parameter_count(model),
    }
    dataset = {
        "root": str(root),
        "file_sha256": dataset_hashes,
        "split": "official_train",
        "official_train_identity_count": int(full_pool.images.shape[0]),
    }
    pool = pool_contract(selected_pool)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    train_generator = torch.Generator(device="cpu").manual_seed(config.train_sequence_seed)
    print(
        json.dumps(
            {
                "event": "start",
                "run_id": config.run_id,
                "bd_ring_size": config.feed.bd_ring_size,
                "bd_tile_size": config.feed.render_tile_size,
                "objective": f"{config.prediction_weight} * prediction_mse + {config.sigreg_weight} * SIGReg",
                "additional_latent_regularizers": [],
                "target_detached": False,
                "parameter_counts": counts,
                "epochs": config.epochs,
                "sequences_per_epoch": config.sequences_per_epoch,
                "steps_per_epoch": config.steps_per_epoch,
                "steps": config.steps,
                "identity_pool": pool,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    evaluations = [
        {"step": 0, "epoch": 0, "evaluation": evaluate(model, sigreg, feed, config, device)}
    ]
    curve: list[dict[str, Any]] = []
    raw_examples = None
    model.train()
    step_index = 0
    for epoch_index in range(config.epochs):
        for batch_index in range(config.steps_per_epoch):
            start = batch_index * config.batch_size
            actual_batch_size = min(config.batch_size, config.sequences_per_epoch - start)
            if actual_batch_size <= 0:
                raise AssertionError("invalid final batch calculation")
            batch = feed.sample(actual_batch_size, generator=train_generator, device=device)
            if raw_examples is None:
                raw_examples = {
                    "pool_local_identity_states": batch.states[:2].cpu().tolist(),
                    "cifar_class_labels": batch.labels[:2].cpu().tolist(),
                    "official_train_image_indices": batch.source_image_indices[:2].cpu().tolist(),
                    "copy_branch": batch.copy_branch[:2].cpu().tolist(),
                }
                print(json.dumps({"event": "raw_data_examples", **raw_examples}, sort_keys=True), flush=True)
            learning_rate = learning_rate_at_step(config, step_index)
            optimizer.param_groups[0]["lr"] = learning_rate
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=config.bf16):
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
            loss = config.prediction_weight * prediction_loss + config.sigreg_weight * sigreg_loss
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite standard JEPA loss")
            loss.backward()
            gradient_norm = nn.utils.clip_grad_norm_(
                model.parameters(), config.gradient_clip, error_if_nonfinite=True
            )
            optimizer.step()
            step_index += 1
            if step_index == 1 or step_index % config.log_every == 0 or step_index == config.steps:
                record = {
                    "step": step_index,
                    "epoch": epoch_index + 1,
                    "batch_in_epoch": batch_index + 1,
                    "actual_batch_size": actual_batch_size,
                    "learning_rate": learning_rate,
                    "total_loss": float(loss.detach()),
                    "prediction_mse": float(prediction_loss.detach()),
                    "sigreg": float(sigreg_loss.detach()),
                    "gradient_norm": float(gradient_norm.detach()),
                    "latent_collapse": collapse_summary(latents.detach().float()),
                    "feed_statistics": batch_statistics(batch, config.feed),
                }
                curve.append(record)
                print(json.dumps({"event": "train", **record}, sort_keys=True), flush=True)
            if step_index % config.evaluation_every == 0 or step_index == config.steps:
                evaluation = evaluate(model, sigreg, feed, config, device)
                evaluations.append(
                    {"step": step_index, "epoch": epoch_index + 1, "evaluation": evaluation}
                )
                print(
                    json.dumps(
                        {
                            "event": "evaluation",
                            "step": step_index,
                            "epoch": epoch_index + 1,
                            "prediction_mse": evaluation["prediction_mse"],
                            "sigreg": evaluation["sigreg"],
                            "total_objective": evaluation["total_objective"],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                model.train()
    if step_index != config.steps:
        raise AssertionError("training did not consume the configured number of steps")
    if raw_examples is None:
        raise AssertionError("training produced no raw examples")
    checkpoint = _state_artifact(
        artifact_kind=CHECKPOINT_KIND,
        config=config,
        model=model,
        counts=counts,
        dataset=dataset,
        pool=pool,
        completed_epoch=config.epochs,
        completed_steps=step_index,
        curve=curve,
        evaluations=evaluations,
        raw_examples=raw_examples,
        source_manifest=source_manifest,
    )
    runtime = runtime_contract(device)
    if device.type == "cuda":
        runtime["peak_gpu_memory_allocated_bytes"] = int(torch.cuda.max_memory_allocated(device))
    checkpoint["runtime"] = runtime
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_save_new(checkpoint, checkpoint_path)
    atomic_write_json_new(
        {key: value for key, value in checkpoint.items() if key != "model_state_dict"},
        json_path,
    )
    digest = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    print(
        json.dumps({"event": "complete", "run_id": config.run_id, "checkpoint_sha256": digest}),
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
