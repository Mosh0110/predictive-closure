from __future__ import annotations
import argparse
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
from .corpus import MaterializedSequenceCorpus
from .feed import (
    DictionaryFeedConfig,
    FourImageDictionaryFeed,
    RENDER_LAYOUTS,
    dictionary_sha256,
    identity_pool_sha256,
    select_identity_pool,
    transition_statistics,
)
from .models import FourImageJEPA, parameter_count


FORMAT_VERSION = 1
CHECKPOINT_KIND = "jepa_four_image_dictionary_standard_checkpoint"
PREDICTION_WEIGHT = 0.9
SIGREG_WEIGHT = 0.1


def dictionary_construction_name(mode: str) -> str:
    if mode == "identity":
        return "four_identity_maps"
    if mode == "random_permutation":
        return "four_independent_torch_randperm"
    raise ValueError(f"unknown dictionary mode {mode!r}")


@dataclass(frozen=True)
class Config:
    format_version: int
    run_id: str
    data_root: str
    training_corpus: str
    output_checkpoint: str
    output_json: str
    seed: int
    train_sequence_seed: int
    evaluation_sequence_seed: int
    evaluation_sigreg_seed: int
    feed: DictionaryFeedConfig
    render_layout: str
    latent_dim: int
    encoder_base_channels: int
    encoder_representation_width: int
    encoder_frame_chunk_size: int
    predictor_hidden_dim: int
    steps: int
    schedule_steps: int
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
        migrated = dict(value)
        migrated.setdefault("render_layout", "columns")
        normalized = strict_dataclass_mapping(cls, migrated, "four-image JEPA config")
        normalized["feed"] = DictionaryFeedConfig(**normalized["feed"])
        config = cls(**normalized)
        config.validate()
        return config

    def validate(self) -> None:
        if self.format_version != FORMAT_VERSION:
            raise ValueError(f"format_version must equal {FORMAT_VERSION}")
        if Path(self.run_id).name != self.run_id or not self.run_id:
            raise ValueError("run_id must be one filename-safe component")
        if not self.training_corpus:
            raise ValueError("training_corpus must be nonempty")
        if self.render_layout not in RENDER_LAYOUTS:
            raise ValueError(f"render_layout must be one of {RENDER_LAYOUTS}")
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
            "steps",
            "schedule_steps",
            "batch_size",
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
        if self.schedule_steps > self.steps:
            raise ValueError("schedule_steps cannot exceed steps")
        if not 0 <= self.warmup_steps < self.schedule_steps:
            raise ValueError("warmup_steps must lie in [0,schedule_steps)")
        if self.sigreg_knots < 2:
            raise ValueError("sigreg_knots must be at least two")
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
        if (self.prediction_weight, self.sigreg_weight) not in ((.95,.05),(.9,.1),(.8,.2)):
            raise ValueError("unsupported prediction/SIGReg coefficient pair")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if not 0.0 <= self.minimum_learning_rate <= self.learning_rate:
            raise ValueError("minimum_learning_rate is invalid")
        if self.weight_decay < 0.0 or self.gradient_clip <= 0.0:
            raise ValueError("optimizer scalars are invalid")
        if type(self.bf16) is not bool:
            raise TypeError("bf16 must be boolean")


def learning_rate_at_step(config: Config, step_index: int) -> float:
    schedule_index = min(step_index, config.schedule_steps - 1)
    if schedule_index < config.warmup_steps:
        return config.learning_rate * (schedule_index + 1) / config.warmup_steps
    progress = (schedule_index - config.warmup_steps) / max(
        1, config.schedule_steps - config.warmup_steps - 1
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
        "minimum_per_time_effective_rank": min(
            record["effective_rank"] for record in per_time
        ),
        "minimum_per_time_mean_variance": min(
            record["mean_variance"] for record in per_time
        ),
    }


def _source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    return source_hashes({str(p.relative_to(root)): p for p in sorted(root.rglob("*.py"))})


@torch.no_grad()
def evaluate(
    model: FourImageJEPA,
    sigreg: SIGReg,
    feed: FourImageDictionaryFeed,
    config: Config,
    device: torch.device,
) -> dict[str, Any]:
    was_training = model.training
    model.eval()
    data_generator = torch.Generator(device="cpu").manual_seed(
        config.evaluation_sequence_seed
    )
    devices = (
        [device.index if device.index is not None else torch.cuda.current_device()]
        if device.type == "cuda"
        else []
    )
    prediction_values = []
    sigreg_values = []
    collapse = []
    transition_records = []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(config.evaluation_sigreg_seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(config.evaluation_sigreg_seed)
        for _ in range(config.evaluation_batches):
            batch = feed.sample(
                config.evaluation_batch_size,
                generator=data_generator,
                device=device,
                render_layout=config.render_layout,
            )
            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=config.bf16
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
            collapse.append(collapse_summary(latents.float()))
            transition_records.append(transition_statistics(batch))
    model.train(was_training)
    transition_mean = {}
    for position in ("A", "C", "B", "D"):
        transition_mean[position] = {}
        for key in transition_records[0][position]:
            values = [record[position][key] for record in transition_records]
            finite_values = [value for value in values if value is not None]
            transition_mean[position][key] = (
                sum(finite_values) / len(finite_values) if finite_values else None
            )
    prediction_mean = sum(prediction_values) / len(prediction_values)
    sigreg_mean = sum(sigreg_values) / len(sigreg_values)
    return {
        "prediction_mse": prediction_mean,
        "sigreg": sigreg_mean,
        "total_objective": (
            config.prediction_weight * prediction_mean
            + config.sigreg_weight * sigreg_mean
        ),
        "per_batch_prediction_mse": prediction_values,
        "per_batch_sigreg": sigreg_values,
        "latent_collapse_per_batch": collapse,
        "empirical_transitions": transition_mean,
    }


def run(config: Config) -> dict[str, Any]:
    config.validate()
    checkpoint_path = Path(config.output_checkpoint).expanduser()
    json_path = Path(config.output_json).expanduser()
    if checkpoint_path.exists() or json_path.exists():
        raise FileExistsError("refusing to overwrite four-image JEPA outputs")
    configure_cpu_threads(config.cpu_threads, config.cpu_interop_threads)
    device = resolve_device(config.device, bf16=config.bf16)
    set_seed(config.seed)
    root, dataset_hashes = dataset_file_hashes(config.data_root)
    full_pool = load_cifar10_python_batches(root, split="train")
    pool, identity_source_indices = select_identity_pool(
        full_pool,
        identity_count=config.feed.identity_count,
        seed=config.feed.identity_subset_seed,
    )
    feed = FourImageDictionaryFeed(pool, config.feed)
    corpus = MaterializedSequenceCorpus.load(config.training_corpus)
    corpus.validate_against(
        pool=pool,
        identity_source_indices=identity_source_indices,
        feed_config=config.feed,
        dictionaries=feed.dictionaries,
        dataset_file_sha256=dataset_hashes,
        expected_seed=config.train_sequence_seed,
    )
    steps_per_epoch = math.ceil(corpus.sequences_per_epoch / config.batch_size)
    expected_steps = corpus.epoch_count * steps_per_epoch
    if config.steps != expected_steps:
        raise ValueError(
            f"steps must consume the corpus exactly once: expected {expected_steps}, "
            f"got {config.steps}"
        )
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
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    counts = {
        "encoder": parameter_count(model.encoder),
        "predictor": parameter_count(model.predictor),
        "total": parameter_count(model),
    }
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    dictionary_key_exposures = (
        corpus.epoch_count
        * corpus.sequences_per_epoch
        * (config.feed.sequence_length - 1)
        / pool.images.shape[0]
    )
    print(
        json.dumps(
            {
                "event": "start",
                "run_id": config.run_id,
                "objective": f"{config.prediction_weight} * prediction_mse + {config.sigreg_weight} * SIGReg",
                "additional_latent_regularizers": [],
                "target_detached": False,
                "parameter_counts": counts,
                "materialized_epochs": corpus.epoch_count,
                "sequences_per_epoch": corpus.sequences_per_epoch,
                "steps_per_epoch": steps_per_epoch,
                "expected_dictionary_key_exposures_per_map": dictionary_key_exposures,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    evaluations = [
        {"step": 0, "evaluation": evaluate(model, sigreg, feed, config, device)}
    ]
    curve = []
    model.train()
    for step_index in range(config.steps):
        epoch_index = step_index // steps_per_epoch
        batch_index = step_index % steps_per_epoch
        start = batch_index * config.batch_size
        stop = min(start + config.batch_size, corpus.sequences_per_epoch)
        learning_rate = learning_rate_at_step(config, step_index)
        optimizer.param_groups[0]["lr"] = learning_rate
        batch = corpus.batch(
            epoch_index,
            start,
            stop,
            pool=pool,
            dictionaries=feed.dictionaries,
            device=device,
            render_layout=config.render_layout,
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=config.bf16
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
            model.parameters(), config.gradient_clip, error_if_nonfinite=True
        )
        optimizer.step()
        step = step_index + 1
        if step == 1 or step % config.log_every == 0 or step == config.steps:
            record = {
                "step": step,
                "epoch": epoch_index + 1,
                "batch_in_epoch": batch_index + 1,
                "actual_batch_size": stop - start,
                "learning_rate": learning_rate,
                "total_loss": float(loss.detach()),
                "prediction_mse": float(prediction_loss.detach()),
                "sigreg": float(sigreg_loss.detach()),
                "gradient_norm": float(gradient_norm.detach()),
                "latent_collapse": collapse_summary(latents.detach().float()),
                "empirical_transitions": transition_statistics(batch),
            }
            curve.append(record)
            print(json.dumps({"event": "train", **record}, sort_keys=True), flush=True)
        if step % config.evaluation_every == 0 or step == config.steps:
            evaluation = evaluate(model, sigreg, feed, config, device)
            evaluations.append({"step": step, "evaluation": evaluation})
            print(
                json.dumps(
                    {
                        "event": "evaluation",
                        "step": step,
                        "prediction_mse": evaluation["prediction_mse"],
                        "sigreg": evaluation["sigreg"],
                        "total_objective": evaluation["total_objective"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            model.train()
    runtime = runtime_contract(device)
    if device.type == "cuda":
        runtime["peak_gpu_memory_allocated_bytes"] = int(
            torch.cuda.max_memory_allocated(device)
        )
    trained_sequences = corpus.epoch_count * corpus.sequences_per_epoch
    frame_count = trained_sequences * config.feed.sequence_length
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
            "split": "official_train_deterministic_class_balanced_identity_subset",
            "official_train_identity_count": int(full_pool.images.shape[0]),
            "identity_count": int(pool.images.shape[0]),
            "identity_subset_seed": config.feed.identity_subset_seed,
            "identity_source_indices_sha256": identity_pool_sha256(
                identity_source_indices
            ),
        },
        "dictionaries": {
            "mode": config.feed.dictionary_mode,
            "construction": dictionary_construction_name(config.feed.dictionary_mode),
            "seed": config.feed.dictionary_seed,
            "sha256": dictionary_sha256(feed.dictionaries),
            "source_position_for_target": {"A": "B", "C": "D", "B": "A", "D": "C"},
            "probabilities": dict(zip(("A", "C", "B", "D"), config.feed.probabilities, strict=True)),
        },
        "training_corpus": corpus.metadata(),
        "data_volume": {
            "materialized_sequences": trained_sequences,
            "materialized_epochs": corpus.epoch_count,
            "sequences_per_epoch": corpus.sequences_per_epoch,
            "steps_per_epoch": steps_per_epoch,
            "collage_frames": frame_count,
            "position_image_appearances": 4 * frame_count,
            "dictionary_transitions_per_map": trained_sequences
            * (config.feed.sequence_length - 1),
            "expected_dictionary_key_exposures_per_map": dictionary_key_exposures,
        },
        "source_hashes": _source_hashes(),
        "runtime": runtime,
        "completed_steps": config.steps,
        "completed_epochs": corpus.epoch_count,
        "learning_curve": curve,
        "evaluation_curve": evaluations,
    }
    atomic_save_new(checkpoint, checkpoint_path)
    atomic_write_json_new(
        {key: value for key, value in checkpoint.items() if key != "model_state_dict"},
        json_path,
    )
    print(json.dumps({"event": "complete", "run_id": config.run_id}), flush=True)
    return checkpoint


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    run(Config.from_mapping(load_json_object(args.config.expanduser())))


if __name__ == "__main__":
    main()
