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
from jepa.losses import SIGReg, prediction_mse
from .corpus import MaterializedSequenceCorpus
from .feed import (
    DictionaryFeedConfig,
    FourImageDictionaryFeed,
    dictionary_sha256,
    identity_pool_sha256,
    select_identity_pool,
    transition_statistics,
)
from .models import FourImageJEPA, parameter_count
from .side_model import build_model
from .reconstruction import (
    COARSE_IMAGE_SIZE,
    RECONSTRUCTION_POSITIONS,
    RECONSTRUCTION_POSITION_NAMES,
    CoarseTargetStandardizer,
    LinearCoarseReconstructionDecoder,
    coarse_pool_images,
    reconstruction_mse,
    standardized_reconstruction_targets,
    tensor_sha256,
)
from .train_jepa import (
    Config as StandardConfig,
    collapse_summary,
    dictionary_construction_name,
    learning_rate_at_step,
)


FORMAT_VERSION = 1
CHECKPOINT_KIND = "jepa_four_image_dictionary_single_feature_steering_checkpoint"
PREDICTION_WEIGHT = 0.9
SIGREG_WEIGHT = 0.1


@dataclass(frozen=True)
class Config(StandardConfig):
    reconstruction_positions: list[int]
    reconstruction_target_size: int
    reconstruction_weight: float
    reconstruction_decoder_seed: int
    condition: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "Config":
        normalized = strict_dataclass_mapping(cls, value, "single-feature reconstruction JEPA config")
        normalized["feed"] = DictionaryFeedConfig(**normalized["feed"])
        config = cls(**normalized)
        config.validate()
        return config

    def validate(self) -> None:
        super().validate()
        if self.reconstruction_positions != list(RECONSTRUCTION_POSITIONS):
            raise ValueError("reconstruction_positions must contain only position 1 (C)")
        if self.reconstruction_target_size != COARSE_IMAGE_SIZE:
            raise ValueError(f"reconstruction_target_size must equal {COARSE_IMAGE_SIZE}")
        if not math.isfinite(self.reconstruction_weight) or self.reconstruction_weight < 0:
            raise ValueError("reconstruction_weight must be finite and nonnegative")
        if self.condition not in ("baseline", "reconstruction", "side"):
            raise ValueError("unsupported steering condition")
        if (self.condition == "reconstruction") != (self.reconstruction_weight > 0):
            raise ValueError("only reconstruction condition can have a positive weight")
        if self.reconstruction_decoder_seed != 900000 + self.seed:
            raise ValueError("decoder seed differs")
        if self.render_layout != "columns" or self.latent_dim != 3:
            raise ValueError("steering requires columns layout and latent_dim=3")


def _source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    return source_hashes({str(p.relative_to(root)): p for p in sorted(root.rglob("*.py"))})


def _decoder(
    config: Config, device: torch.device
) -> LinearCoarseReconstructionDecoder:
    devices = (
        [device.index if device.index is not None else torch.cuda.current_device()]
        if device.type == "cuda"
        else []
    )
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(config.reconstruction_decoder_seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(config.reconstruction_decoder_seed)
        decoder = LinearCoarseReconstructionDecoder(
            config.latent_dim, config.reconstruction_target_size
        ).to(device)
    return decoder


def _target_components(
    pool_images: torch.Tensor, device: torch.device, target_size: int
) -> tuple[torch.Tensor, CoarseTargetStandardizer]:
    images = pool_images.to(device=device, dtype=torch.float32).div_(255.0)
    coarse = coarse_pool_images(images, target_size)
    standardizer = CoarseTargetStandardizer.fit(coarse, target_size)
    return coarse, standardizer


def objective_contract(config: Config) -> dict[str, Any]:
    return {
        "formula": f"{config.prediction_weight} * prediction_mse + {config.sigreg_weight} * SIGReg + beta * current_C_coarse_reconstruction_mse",
        "prediction_weight": config.prediction_weight,
        "sigreg_weight": config.sigreg_weight,
        "reconstruction_weight": config.reconstruction_weight,
        "additional_latent_regularizers": [],
        "prediction_target_detached": False,
        "reconstruction_target_detached": True,
        "reconstruction_decoder_input": "current main-encoder latent z_t only",
        "reconstruction_decoder_architecture": "single linear map",
        "reconstruction_logical_positions": list(RECONSTRUCTION_POSITION_NAMES) if config.condition == "reconstruction" else [],
        "reconstruction_position_indices": list(RECONSTRUCTION_POSITIONS) if config.condition == "reconstruction" else [],
        "reconstruction_time_relation": "current-time image from current-time latent",
        "reconstruction_target": "per-element-standardized 4x4 RGB average-pooled appearance",
        "reconstruction_target_size": config.reconstruction_target_size,
        "condition": config.condition,
        "side_information": "current C image, independent same-architecture encoder" if config.condition == "side" else None,
        "side_time": "t only; no t+1 information",
        "main_input_masked": False,
        "reconstruction_loss_applied": config.condition == "reconstruction",
        "decoder_used_by_predictor": False,
    }


@torch.no_grad()
def evaluate(
    model: FourImageJEPA,
    decoder: LinearCoarseReconstructionDecoder,
    sigreg: SIGReg,
    feed: FourImageDictionaryFeed,
    coarse_images: torch.Tensor,
    standardizer: CoarseTargetStandardizer,
    config: Config,
    device: torch.device,
) -> dict[str, Any]:
    model_was_training = model.training
    decoder_was_training = decoder.training
    model.eval()
    decoder.eval()
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
    reconstruction_values = []
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
                reconstruction_predictions = decoder(latents)
            targets = standardized_reconstruction_targets(
                batch.states, coarse_images, standardizer
            )
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
            reconstruction_values.append(
                float(reconstruction_mse(reconstruction_predictions.float(), targets))
            )
            collapse.append(collapse_summary(latents.float()))
            transition_records.append(transition_statistics(batch))
    model.train(model_was_training)
    decoder.train(decoder_was_training)
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
    reconstruction_mean = sum(reconstruction_values) / len(reconstruction_values)
    return {
        "prediction_mse": prediction_mean,
        "sigreg": sigreg_mean,
        "reconstruction_mse": reconstruction_mean,
        "constant_mean_reconstruction_mse": 1.0,
        "total_objective": (
            config.prediction_weight * prediction_mean
            + config.sigreg_weight * sigreg_mean
            + config.reconstruction_weight * reconstruction_mean
        ),
        "per_batch_prediction_mse": prediction_values,
        "per_batch_sigreg": sigreg_values,
        "per_batch_reconstruction_mse": reconstruction_values,
        "latent_collapse_per_batch": collapse,
        "empirical_transitions": transition_mean,
    }


def run(config: Config) -> dict[str, Any]:
    config.validate()
    checkpoint_path = Path(config.output_checkpoint).expanduser()
    json_path = Path(config.output_json).expanduser()
    if checkpoint_path.exists() or json_path.exists():
        raise FileExistsError("refusing to overwrite single-feature reconstruction JEPA outputs")
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
    model = build_model(config).to(device)
    sigreg = SIGReg(
        knots=config.sigreg_knots,
        num_projections=config.sigreg_projections,
        expected_time_steps=config.feed.sequence_length,
    ).to(device)
    decoder = _decoder(config, device)
    coarse_images, standardizer = _target_components(
        pool.images, device, config.reconstruction_target_size
    )
    trainable_parameters = [*model.parameters(), *decoder.parameters()]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    counts = {
        "encoder": parameter_count(model.encoder),
        "predictor": parameter_count(model.predictor),
        "reconstruction_decoder": parameter_count(decoder),
        "jepa_total": parameter_count(model),
        "side_encoder": parameter_count(model.side_encoder) if config.condition == "side" else 0,
        "training_total": parameter_count(model) + parameter_count(decoder),
    }
    if counts["reconstruction_decoder"] != 192:
        raise AssertionError("the 3-to-48 decoder must have 192 parameters")
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
                "objective": objective_contract(config)["formula"],
                "objective_contract": objective_contract(config),
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
        {
            "step": 0,
            "evaluation": evaluate(
                model,
                decoder,
                sigreg,
                feed,
                coarse_images,
                standardizer,
                config,
                device,
            ),
        }
    ]
    curve = []
    model.train()
    decoder.train()
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
            reconstruction_predictions = decoder(latents)
        reconstruction_targets = standardized_reconstruction_targets(
            batch.states, coarse_images, standardizer
        )
        prediction_loss = prediction_mse(
            predictions.float(),
            latents[1:].float(),
            expected_time_steps=config.feed.sequence_length - 1,
        )
        with torch.autocast(device_type=device.type, enabled=False):
            sigreg_loss = sigreg.forward_chunked(
                latents.float(), chunk_size=config.sigreg_chunk_size
            )
            reconstruction_loss = reconstruction_mse(
                reconstruction_predictions.float(), reconstruction_targets
            )
        loss = (
            config.prediction_weight * prediction_loss
            + config.sigreg_weight * sigreg_loss
            + config.reconstruction_weight * reconstruction_loss
        )
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite single-feature reconstruction JEPA loss")
        loss.backward()
        gradient_norm = nn.utils.clip_grad_norm_(
            trainable_parameters, config.gradient_clip, error_if_nonfinite=True
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
                "reconstruction_mse": float(reconstruction_loss.detach()),
                "weighted_reconstruction_mse": float(
                    config.reconstruction_weight * reconstruction_loss.detach()
                ),
                "gradient_norm": float(gradient_norm.detach()),
                "latent_collapse": collapse_summary(latents.detach().float()),
                "empirical_transitions": transition_statistics(batch),
            }
            curve.append(record)
            print(json.dumps({"event": "train", **record}, sort_keys=True), flush=True)
        if step % config.evaluation_every == 0 or step == config.steps:
            evaluation = evaluate(
                model,
                decoder,
                sigreg,
                feed,
                coarse_images,
                standardizer,
                config,
                device,
            )
            evaluations.append({"step": step, "evaluation": evaluation})
            print(
                json.dumps(
                    {
                        "event": "evaluation",
                        "step": step,
                        "prediction_mse": evaluation["prediction_mse"],
                        "sigreg": evaluation["sigreg"],
                        "reconstruction_mse": evaluation["reconstruction_mse"],
                        "total_objective": evaluation["total_objective"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            model.train()
            decoder.train()
    runtime = runtime_contract(device)
    if device.type == "cuda":
        runtime["peak_gpu_memory_allocated_bytes"] = int(
            torch.cuda.max_memory_allocated(device)
        )
        runtime["peak_gpu_memory_reserved_bytes"] = int(
            torch.cuda.max_memory_reserved(device)
        )
    trained_sequences = corpus.epoch_count * corpus.sequences_per_epoch
    frame_count = trained_sequences * config.feed.sequence_length
    checkpoint = {
        "format_version": FORMAT_VERSION,
        "artifact_kind": CHECKPOINT_KIND,
        "config": asdict(config),
        "model_state_dict": cpu_state_dict(model),
        "reconstruction_decoder_state_dict": cpu_state_dict(decoder),
        "parameter_counts": counts,
        "objective_contract": objective_contract(config),
        "reconstruction_target_contract": {
            "logical_positions": list(RECONSTRUCTION_POSITION_NAMES),
            "logical_position_indices": list(RECONSTRUCTION_POSITIONS),
            "derived_from": "current batch states and the frozen JEPA training identity pool",
            "physical_collage_layout_used": False,
            "future_states_used": False,
            "labels_used": False,
            "target_size": config.reconstruction_target_size,
            "target_feature_count": decoder.output_dim,
            "coarse_pool_sha256": tensor_sha256(coarse_images),
            "standardization": standardizer.contract(),
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
            "probabilities": dict(
                zip(("A", "C", "B", "D"), config.feed.probabilities, strict=True)
            ),
        },
        "training_corpus": corpus.metadata(),
        "data_volume": {
            "materialized_sequences": trained_sequences,
            "materialized_epochs": corpus.epoch_count,
            "sequences_per_epoch": corpus.sequences_per_epoch,
            "steps_per_epoch": steps_per_epoch,
            "collage_frames": frame_count,
            "position_image_appearances": 4 * frame_count,
            "reconstruction_position_image_targets": len(RECONSTRUCTION_POSITIONS)
            * frame_count,
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
        {
            key: value
            for key, value in checkpoint.items()
            if key not in {"model_state_dict", "reconstruction_decoder_state_dict"}
        },
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
