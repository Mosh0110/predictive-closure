from __future__ import annotations
import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
import torch
import torch.nn.functional as F
from torch import nn
from jepa.common import (
    atomic_save_new,
    atomic_write_json_new,
    configure_cpu_threads,
    cpu_state_dict,
    dataset_file_hashes,
    load_json_object,
    module_state_sha256,
    resolve_device,
    set_seed,
    strict_dataclass_mapping,
)
from jepa.data import load_cifar10_python_batches
from jepa.four_image.models import FourImageJEPA, parameter_count
from .feed import (
    COLOR_COUNT,
    IMAGE_POSITIONS,
    ReplacementFeedConfig,
    ReplacementSubsidyFeed,
    color_palette,
    render_collages,
    select_identity_pool,
)
from .train_jepa import (
    CHECKPOINT_KIND as JEPA_CHECKPOINT_KIND,
    Config as JEPAConfig,
    current_source_hashes,
)


FORMAT_VERSION = 1
ARTIFACT_KIND = "jepa_traffic_light_two_image_probe_result"


@dataclass(frozen=True)
class Config:
    format_version: int
    run_id: str
    jepa_checkpoint: str
    data_root: str
    output_checkpoint: str
    output_json: str
    seed: int
    probe_identity_count: int
    probe_identity_subset_seed: int
    current_train_seed: int
    current_validation_seed: int
    current_test_seed: int
    future_train_seed: int
    future_validation_seed: int
    future_test_seed: int
    train_samples: int
    validation_samples: int
    test_samples: int
    encoding_batch_size: int
    probe_hidden_dim: int
    probe_steps: int
    probe_batch_size: int
    learning_rate: float
    minimum_learning_rate: float
    warmup_steps: int
    weight_decay: float
    gradient_clip: float
    log_every: int
    bf16: bool
    cpu_threads: int
    cpu_interop_threads: int
    device: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "Config":
        config = cls(
            **strict_dataclass_mapping(
                cls, value, "traffic-light probe config"
            )
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.format_version != FORMAT_VERSION:
            raise ValueError(f"format_version must equal {FORMAT_VERSION}")
        if Path(self.run_id).name != self.run_id or not self.run_id:
            raise ValueError("run_id must be one filename-safe component")
        seed_names = (
            "seed",
            "probe_identity_subset_seed",
            "current_train_seed",
            "current_validation_seed",
            "current_test_seed",
            "future_train_seed",
            "future_validation_seed",
            "future_test_seed",
        )
        for name in seed_names:
            value = getattr(self, name)
            if type(value) is not int or not 0 <= value < 2**63:
                raise ValueError(f"{name} must be an integer in [0,2^63)")
        split_seeds = [getattr(self, name) for name in seed_names[2:]]
        if len(set(split_seeds)) != len(split_seeds):
            raise ValueError("probe split seeds must be pairwise distinct")
        for name in (
            "probe_identity_count",
            "train_samples",
            "validation_samples",
            "test_samples",
            "encoding_batch_size",
            "probe_hidden_dim",
            "probe_steps",
            "probe_batch_size",
            "warmup_steps",
            "log_every",
            "cpu_threads",
            "cpu_interop_threads",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.probe_identity_count % 10:
            raise ValueError("probe_identity_count must be divisible by ten")
        if self.probe_batch_size > self.train_samples:
            raise ValueError("probe batch exceeds training samples")
        if self.warmup_steps >= self.probe_steps:
            raise ValueError("warmup_steps must be smaller than probe_steps")
        for name in (
            "learning_rate",
            "minimum_learning_rate",
            "weight_decay",
            "gradient_clip",
        ):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if not 0.0 <= self.minimum_learning_rate <= self.learning_rate:
            raise ValueError("minimum_learning_rate is invalid")
        if self.weight_decay < 0.0 or self.gradient_clip <= 0.0:
            raise ValueError("optimizer scalars are invalid")
        if type(self.bf16) is not bool:
            raise TypeError("bf16 must be boolean")


class Probe(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


def _load_encoder(
    checkpoint_path: Path,
    data_root: Path,
    device: torch.device,
) -> tuple[nn.Module, JEPAConfig, dict[str, Any], str]:
    raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(raw, Mapping) or raw.get("artifact_kind") != JEPA_CHECKPOINT_KIND:
        raise ValueError("checkpoint is not a traffic-light artifact")
    config = JEPAConfig.from_mapping(raw["config"])
    if raw.get("completed_steps") != config.steps:
        raise ValueError("JEPA checkpoint is not terminal")
    if raw.get("source_hashes") != current_source_hashes():
        raise ValueError("JEPA source hashes do not reproduce")
    _, hashes = dataset_file_hashes(data_root)
    if raw["dataset"]["file_sha256"] != hashes:
        raise ValueError("CIFAR files differ from the JEPA checkpoint")
    model = FourImageJEPA(
        sequence_length=config.feed.sequence_length,
        latent_dim=config.latent_dim,
        base_channels=config.encoder_base_channels,
        representation_width=config.encoder_representation_width,
        predictor_hidden_dim=config.predictor_hidden_dim,
        encoder_frame_chunk_size=config.encoder_frame_chunk_size,
    ).to(device)
    model.load_state_dict(raw["model_state_dict"], strict=True)
    model.encoder.eval().requires_grad_(False)
    digest = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    return model.encoder, config, dict(raw["parameter_counts"]), digest


@torch.no_grad()
def _encode_frames(
    encoder: nn.Module,
    frames: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
    bf16: bool,
) -> torch.Tensor:
    results = []
    for chunk in frames.split(batch_size):
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=bf16,
        ):
            results.append(encoder(chunk.to(device)).float().cpu())
    return torch.cat(results)


def _current_split(
    encoder: nn.Module,
    pool: Any,
    *,
    sample_count: int,
    seed: int,
    config: Config,
    jepa_config: JEPAConfig,
    device: torch.device,
) -> dict[str, Any]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    states = torch.randint(
        pool.images.shape[0],
        (sample_count, len(IMAGE_POSITIONS)),
        generator=generator,
        dtype=torch.int64,
    )
    lights = torch.randint(
        COLOR_COUNT,
        (sample_count,),
        generator=generator,
        dtype=torch.int64,
    )
    images = pool.images[states].to(dtype=torch.float32).div_(255.0)
    frames = render_collages(
        images[:, None],
        lights[:, None],
        color_palette().mul(jepa_config.feed.light_strength),
        light_patch_size=jepa_config.feed.light_patch_size,
    )[:, 0]
    latents = _encode_frames(
        encoder,
        frames,
        batch_size=config.encoding_batch_size,
        device=device,
        bf16=config.bf16,
    )
    return {
        "latents": latents,
        "light": lights,
        "classes": pool.labels.index_select(0, states.reshape(-1)).reshape_as(states),
        "raw_examples": [
            {
                "image_states": states[index].tolist(),
                "image_classes": pool.labels[states[index]].tolist(),
                "light": int(lights[index]),
            }
            for index in range(2)
        ],
    }


def _future_split(
    encoder: nn.Module,
    feed: ReplacementSubsidyFeed,
    *,
    sample_count: int,
    seed: int,
    config: Config,
    device: torch.device,
) -> dict[str, Any]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    transition_count = feed.config.sequence_length - 1
    sequence_count = math.ceil(sample_count / transition_count)
    frames = []
    future_lights = []
    current_lights = []
    modes = []
    remaining = sequence_count
    while remaining:
        batch_size = min(remaining, 512)
        batch = feed.sample(batch_size, generator=generator, device="cpu")
        frames.append(batch.frames[:, :-1].reshape(-1, *batch.frames.shape[2:]))
        future_lights.append(batch.lights[:, 1:].reshape(-1))
        current_lights.append(batch.lights[:, :-1].reshape(-1))
        modes.append(batch.modes.reshape(-1))
        remaining -= batch_size
    flat_frames = torch.cat(frames)[:sample_count]
    labels = torch.cat(future_lights)[:sample_count]
    current = torch.cat(current_lights)[:sample_count]
    actual_modes = torch.cat(modes)[:sample_count]
    latents = _encode_frames(
        encoder,
        flat_frames,
        batch_size=config.encoding_batch_size,
        device=device,
        bf16=config.bf16,
    )
    return {
        "latents": latents,
        "labels": labels,
        "raw_examples": [
            {
                "current_light": int(current[index]),
                "future_light": int(labels[index]),
                "replacement_mode": int(actual_modes[index]),
            }
            for index in range(2)
        ],
    }


def _lr(config: Config, step_index: int) -> float:
    if step_index < config.warmup_steps:
        return config.learning_rate * (step_index + 1) / config.warmup_steps
    progress = (step_index - config.warmup_steps) / max(
        1, config.probe_steps - config.warmup_steps - 1
    )
    return config.minimum_learning_rate + 0.5 * (
        config.learning_rate - config.minimum_learning_rate
    ) * (1.0 + math.cos(math.pi * progress))


@torch.no_grad()
def _evaluate(
    model: nn.Module,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    *,
    chance: float,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    logits = model(inputs.to(device)).float().cpu()
    predictions = logits.argmax(dim=1)
    accuracy = float((predictions == labels).float().mean())
    return {
        "cross_entropy": float(F.cross_entropy(logits, labels)),
        "accuracy": accuracy,
        "chance_accuracy": chance,
        "label_histogram": torch.bincount(
            labels, minlength=logits.shape[1]
        ).tolist(),
        "prediction_histogram": torch.bincount(
            predictions, minlength=logits.shape[1]
        ).tolist(),
    }


def _train_probe(
    name: str,
    splits: Mapping[str, tuple[torch.Tensor, torch.Tensor]],
    *,
    latent_dim: int,
    output_dim: int,
    chance: float,
    config: Config,
    device: torch.device,
    seed_offset: int,
) -> tuple[Probe, dict[str, Any]]:
    torch.manual_seed(config.seed + seed_offset)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed + seed_offset)
    model = Probe(latent_dim, config.probe_hidden_dim, output_dim).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    generator = torch.Generator(device="cpu").manual_seed(
        config.seed + 1_000 + seed_offset
    )
    train_inputs, train_labels = splits["train"]
    train_inputs = train_inputs.to(device)
    train_labels = train_labels.to(device)
    best_validation_accuracy = -1.0
    best_state = None
    curve = []
    model.train()
    for step_index in range(config.probe_steps):
        indices = torch.randint(
            train_inputs.shape[0],
            (config.probe_batch_size,),
            generator=generator,
        ).to(device)
        learning_rate = _lr(config, step_index)
        optimizer.param_groups[0]["lr"] = learning_rate
        logits = model(train_inputs.index_select(0, indices))
        loss = F.cross_entropy(logits, train_labels.index_select(0, indices))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = nn.utils.clip_grad_norm_(
            model.parameters(),
            config.gradient_clip,
            error_if_nonfinite=True,
        )
        optimizer.step()
        step = step_index + 1
        if step == 1 or step % config.log_every == 0 or step == config.probe_steps:
            validation = _evaluate(
                model,
                *splits["validation"],
                chance=chance,
                device=device,
            )
            record = {
                "step": step,
                "loss": float(loss.detach()),
                "gradient_norm": float(gradient_norm.detach()),
                "learning_rate": learning_rate,
                "validation_accuracy": validation["accuracy"],
            }
            curve.append(record)
            print(json.dumps({"event": name, **record}, sort_keys=True), flush=True)
            if validation["accuracy"] > best_validation_accuracy:
                best_validation_accuracy = validation["accuracy"]
                best_state = cpu_state_dict(model)
            model.train()
    if best_state is None:
        raise AssertionError("probe produced no validation checkpoint")
    model.load_state_dict(best_state, strict=True)
    results = {
        "parameters": parameter_count(model),
        "best_validation_accuracy": best_validation_accuracy,
        "curve": curve,
        "train": _evaluate(
            model, *splits["train"], chance=chance, device=device
        ),
        "validation": _evaluate(
            model, *splits["validation"], chance=chance, device=device
        ),
        "test": _evaluate(
            model, *splits["test"], chance=chance, device=device
        ),
    }
    return model, results


def run(config: Config) -> dict[str, Any]:
    config.validate()
    checkpoint_output = Path(config.output_checkpoint).expanduser()
    json_output = Path(config.output_json).expanduser()
    if checkpoint_output.exists() or json_output.exists():
        raise FileExistsError("refusing to overwrite probe outputs")
    configure_cpu_threads(config.cpu_threads, config.cpu_interop_threads)
    device = resolve_device(config.device, bf16=config.bf16)
    set_seed(config.seed)
    encoder, jepa_config, parameter_counts, checkpoint_digest = _load_encoder(
        Path(config.jepa_checkpoint).expanduser(),
        Path(config.data_root).expanduser(),
        device,
    )
    encoder_hash_before = module_state_sha256(encoder)
    root, _ = dataset_file_hashes(config.data_root)
    full_test_pool = load_cifar10_python_batches(root, split="test")
    pool, source_indices = select_identity_pool(
        full_test_pool,
        identity_count=config.probe_identity_count,
        seed=config.probe_identity_subset_seed,
    )
    feed = ReplacementSubsidyFeed(
        pool,
        ReplacementFeedConfig(
            sequence_length=jepa_config.feed.sequence_length,
            visible_mode_probability=jepa_config.feed.visible_mode_probability,
            identity_count=config.probe_identity_count,
            identity_subset_seed=config.probe_identity_subset_seed,
            light_patch_size=jepa_config.feed.light_patch_size,
            light_strength=jepa_config.feed.light_strength,
        ),
    )
    current = {
        "train": _current_split(
            encoder,
            pool,
            sample_count=config.train_samples,
            seed=config.current_train_seed,
            config=config,
            jepa_config=jepa_config,
            device=device,
        ),
        "validation": _current_split(
            encoder,
            pool,
            sample_count=config.validation_samples,
            seed=config.current_validation_seed,
            config=config,
            jepa_config=jepa_config,
            device=device,
        ),
        "test": _current_split(
            encoder,
            pool,
            sample_count=config.test_samples,
            seed=config.current_test_seed,
            config=config,
            jepa_config=jepa_config,
            device=device,
        ),
    }
    future = {
        "train": _future_split(
            encoder,
            feed,
            sample_count=config.train_samples,
            seed=config.future_train_seed,
            config=config,
            device=device,
        ),
        "validation": _future_split(
            encoder,
            feed,
            sample_count=config.validation_samples,
            seed=config.future_validation_seed,
            config=config,
            device=device,
        ),
        "test": _future_split(
            encoder,
            feed,
            sample_count=config.test_samples,
            seed=config.future_test_seed,
            config=config,
            device=device,
        ),
    }

    probe_models = {}
    probe_results = {}
    light_splits = {
        split: (current[split]["latents"], current[split]["light"])
        for split in current
    }
    model, result = _train_probe(
        "current_light_probe",
        light_splits,
        latent_dim=jepa_config.latent_dim,
        output_dim=COLOR_COUNT,
        chance=1.0 / COLOR_COUNT,
        config=config,
        device=device,
        seed_offset=10,
    )
    probe_models["current_light"] = cpu_state_dict(model)
    probe_results["current_light"] = result
    future_splits = {
        split: (future[split]["latents"], future[split]["labels"])
        for split in future
    }
    model, result = _train_probe(
        "future_light_probe",
        future_splits,
        latent_dim=jepa_config.latent_dim,
        output_dim=COLOR_COUNT,
        chance=1.0 / COLOR_COUNT,
        config=config,
        device=device,
        seed_offset=20,
    )
    probe_models["future_light"] = cpu_state_dict(model)
    probe_results["future_light"] = result
    for position_index, position in enumerate(IMAGE_POSITIONS):
        position_splits = {
            split: (
                current[split]["latents"],
                current[split]["classes"][:, position_index],
            )
            for split in current
        }
        model, result = _train_probe(
            f"class_probe_{position}",
            position_splits,
            latent_dim=jepa_config.latent_dim,
            output_dim=10,
            chance=0.1,
            config=config,
            device=device,
            seed_offset=30 + position_index,
        )
        probe_models[f"class_{position}"] = cpu_state_dict(model)
        probe_results[f"class_{position}"] = result

    encoder_hash_after = module_state_sha256(encoder)
    if encoder_hash_after != encoder_hash_before:
        raise AssertionError("frozen encoder changed during probing")
    if any(parameter.grad is not None for parameter in encoder.parameters()):
        raise AssertionError("frozen encoder accumulated gradients")
    artifact = {
        "format_version": FORMAT_VERSION,
        "artifact_kind": ARTIFACT_KIND,
        "config": asdict(config),
        "jepa_checkpoint_sha256": checkpoint_digest,
        "q": jepa_config.feed.visible_mode_probability,
        "latent_dim": jepa_config.latent_dim,
        "jepa_parameter_counts": parameter_counts,
        "evaluation_pool": {
            "split": "official_test_balanced_identity_subset",
            "identity_count": int(pool.images.shape[0]),
            "identity_source_indices_sha256": hashlib.sha256(
                source_indices.contiguous().numpy().astype("<i8", copy=False).tobytes()
            ).hexdigest(),
            "encoder_saw_these_identities_during_jepa_training": False,
        },
        "probe_results": probe_results,
        "raw_data_examples": {
            "current_train": current["train"]["raw_examples"],
            "future_train": future["train"]["raw_examples"],
        },
        "encoder_state_sha256_before": encoder_hash_before,
        "encoder_state_sha256_after": encoder_hash_after,
        "encoder_unchanged": True,
        "probe_state_dicts": probe_models,
    }
    checkpoint_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    atomic_save_new(artifact, checkpoint_output)
    atomic_write_json_new(
        {key: value for key, value in artifact.items() if key != "probe_state_dicts"},
        json_output,
    )
    summary = {
        name: result["test"]["accuracy"] for name, result in probe_results.items()
    }
    print(
        json.dumps(
            {"event": "complete", "run_id": config.run_id, **summary},
            sort_keys=True,
        ),
        flush=True,
    )
    return artifact


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    run(Config.from_mapping(load_json_object(args.config.expanduser())))


if __name__ == "__main__":
    main()
