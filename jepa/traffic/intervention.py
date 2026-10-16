from __future__ import annotations
import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
import torch
from jepa.common import (
    atomic_write_json_new,
    configure_cpu_threads,
    dataset_file_hashes,
    load_json_object,
    resolve_device,
    set_seed,
    strict_dataclass_mapping,
)
from jepa.data import load_cifar10_python_batches
from jepa.four_image.models import FourImageJEPA
from jepa.losses import prediction_mse
from .feed import (
    COLOR_COUNT,
    ReplacementFeedConfig,
    ReplacementSubsidyFeed,
    color_palette,
    select_identity_pool,
)
from .train_jepa import (
    CHECKPOINT_KIND as JEPA_CHECKPOINT_KIND,
    Config as JEPAConfig,
    current_source_hashes,
)


FORMAT_VERSION = 1
ARTIFACT_KIND = "jepa_traffic_light_two_image_light_intervention"


@dataclass(frozen=True)
class Config:
    format_version: int
    run_id: str
    jepa_checkpoint: str
    data_root: str
    output_json: str
    seed: int
    sequence_seed: int
    evaluation_identity_count: int
    evaluation_identity_subset_seed: int
    batches: int
    batch_size: int
    bf16: bool
    cpu_threads: int
    cpu_interop_threads: int
    device: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "Config":
        config = cls(
            **strict_dataclass_mapping(
                cls, value, "traffic-light intervention config"
            )
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.format_version != FORMAT_VERSION:
            raise ValueError(f"format_version must equal {FORMAT_VERSION}")
        if Path(self.run_id).name != self.run_id or not self.run_id:
            raise ValueError("run_id must be one filename-safe component")
        for name in (
            "seed",
            "sequence_seed",
            "evaluation_identity_subset_seed",
        ):
            value = getattr(self, name)
            if type(value) is not int or not 0 <= value < 2**63:
                raise ValueError(f"{name} must be an integer in [0,2^63)")
        for name in (
            "evaluation_identity_count",
            "batches",
            "batch_size",
            "cpu_threads",
            "cpu_interop_threads",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.evaluation_identity_count % 10:
            raise ValueError("evaluation_identity_count must be divisible by ten")
        if type(self.bf16) is not bool:
            raise TypeError("bf16 must be boolean")


def _load(
    checkpoint_path: Path,
    data_root: Path,
    device: torch.device,
) -> tuple[FourImageJEPA, JEPAConfig]:
    raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(raw, Mapping) or raw.get("artifact_kind") != JEPA_CHECKPOINT_KIND:
        raise ValueError("checkpoint is not a traffic-light artifact")
    config = JEPAConfig.from_mapping(raw["config"])
    if raw.get("completed_steps") != config.steps:
        raise ValueError("checkpoint is not terminal")
    if raw.get("source_hashes") != current_source_hashes():
        raise ValueError("JEPA source hashes do not reproduce")
    _, hashes = dataset_file_hashes(data_root)
    if raw["dataset"]["file_sha256"] != hashes:
        raise ValueError("CIFAR files differ from the checkpoint")
    model = FourImageJEPA(
        sequence_length=config.feed.sequence_length,
        latent_dim=config.latent_dim,
        base_channels=config.encoder_base_channels,
        representation_width=config.encoder_representation_width,
        predictor_hidden_dim=config.predictor_hidden_dim,
        encoder_frame_chunk_size=config.encoder_frame_chunk_size,
    ).to(device)
    model.load_state_dict(raw["model_state_dict"], strict=True)
    model.eval().requires_grad_(False)
    return model, config


def _replace_source_lights(
    frames: torch.Tensor,
    lights: torch.Tensor,
    *,
    light_patch_size: int = 32,
    light_strength: float = 1.0,
) -> torch.Tensor:
    if frames.ndim != 5 or lights.shape != frames.shape[:2]:
        raise ValueError("frames and lights have incompatible shapes")
    if type(light_patch_size) is not int or not 1 <= light_patch_size <= 32:
        raise ValueError("light_patch_size must be an integer in [1,32]")
    if not math.isfinite(light_strength) or not 0.0 < light_strength <= 1.0:
        raise ValueError("light_strength must lie in (0,1]")
    changed = frames.clone()
    counterfactual = (lights[:, :-1] + 1) % COLOR_COUNT
    pixels = color_palette().mul(light_strength).to(frames.device).index_select(
        0, counterfactual.reshape(-1)
    )
    pixels = pixels.reshape(frames.shape[0], frames.shape[1] - 1, 3, 1, 1)
    offset = 32 + (32 - light_patch_size) // 2
    end = offset + light_patch_size
    changed[:, :-1, :, offset:end, offset:end] = pixels
    if not torch.equal(changed[:, -1], frames[:, -1]):
        raise AssertionError("the final target-only frame changed")
    if torch.equal(
        changed[:, :-1, :, 32:, 32:],
        frames[:, :-1, :, 32:, 32:],
    ):
        raise AssertionError("counterfactual intervention changed no source lights")
    if not torch.equal(changed[:, :, :, :32, :], frames[:, :, :, :32, :]):
        raise AssertionError("counterfactual intervention changed an image pixel")
    if not torch.equal(changed[:, :, :, 32:, :32], frames[:, :, :, 32:, :32]):
        raise AssertionError("counterfactual intervention changed an image pixel")
    return changed


@torch.no_grad()
def run(config: Config) -> dict[str, Any]:
    config.validate()
    output_path = Path(config.output_json).expanduser()
    if output_path.exists():
        raise FileExistsError("refusing to overwrite intervention output")
    configure_cpu_threads(config.cpu_threads, config.cpu_interop_threads)
    device = resolve_device(config.device, bf16=config.bf16)
    set_seed(config.seed)
    model, jepa_config = _load(
        Path(config.jepa_checkpoint).expanduser(),
        Path(config.data_root).expanduser(),
        device,
    )
    root, _ = dataset_file_hashes(config.data_root)
    full_test_pool = load_cifar10_python_batches(root, split="test")
    pool, _ = select_identity_pool(
        full_test_pool,
        identity_count=config.evaluation_identity_count,
        seed=config.evaluation_identity_subset_seed,
    )
    feed = ReplacementSubsidyFeed(
        pool,
        ReplacementFeedConfig(
            sequence_length=jepa_config.feed.sequence_length,
            visible_mode_probability=jepa_config.feed.visible_mode_probability,
            identity_count=config.evaluation_identity_count,
            identity_subset_seed=config.evaluation_identity_subset_seed,
            light_patch_size=jepa_config.feed.light_patch_size,
            light_strength=jepa_config.feed.light_strength,
        ),
    )
    generator = torch.Generator(device="cpu").manual_seed(config.sequence_seed)
    rows = []
    for batch_index in range(config.batches):
        batch = feed.sample(config.batch_size, generator=generator, device=device)
        counterfactual_frames = _replace_source_lights(
            batch.frames,
            batch.lights,
            light_patch_size=jepa_config.feed.light_patch_size,
            light_strength=jepa_config.feed.light_strength,
        )
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=config.bf16,
        ):
            original_latents = model.encode_frames(batch.frames)
            original_predictions = model.predictor(original_latents[:-1])
            counterfactual_latents = model.encode_frames(counterfactual_frames)
            counterfactual_predictions = model.predictor(
                counterfactual_latents[:-1]
            )
        targets = original_latents[1:].float()
        original_error = float(
            prediction_mse(
                original_predictions.float(),
                targets,
                expected_time_steps=jepa_config.feed.sequence_length - 1,
            )
        )
        counterfactual_error = float(
            prediction_mse(
                counterfactual_predictions.float(),
                targets,
                expected_time_steps=jepa_config.feed.sequence_length - 1,
            )
        )
        row = {
            "batch": batch_index,
            "realized_visible_branch_rate": float(
                batch.use_visible_mode.float().mean()
            ),
            "original_prediction_mse": original_error,
            "counterfactual_prediction_mse": counterfactual_error,
            "prediction_mse_increase": counterfactual_error - original_error,
            "relative_prediction_mse_increase": (
                counterfactual_error - original_error
            )
            / original_error,
            "source_latent_change_mse": float(
                (
                    counterfactual_latents[:-1].float()
                    - original_latents[:-1].float()
                )
                .square()
                .mean()
            ),
            "prediction_change_mse": float(
                (
                    counterfactual_predictions.float()
                    - original_predictions.float()
                )
                .square()
                .mean()
            ),
        }
        rows.append(row)
        print(json.dumps({"event": "batch", **row}, sort_keys=True), flush=True)
    numeric_keys = [key for key in rows[0] if key != "batch"]
    means = {
        key: sum(row[key] for row in rows) / len(rows) for key in numeric_keys
    }
    sems = {}
    for key in numeric_keys:
        values = [row[key] for row in rows]
        mean = means[key]
        sems[key] = math.sqrt(
            sum((value - mean) ** 2 for value in values) / (len(values) - 1)
        ) / math.sqrt(len(values))
    artifact = {
        "format_version": FORMAT_VERSION,
        "artifact_kind": ARTIFACT_KIND,
        "config": asdict(config),
        "q": jepa_config.feed.visible_mode_probability,
        "intervention": "flip each two-color source light while preserving all images and original target latents",
        "per_batch": rows,
        "mean": means,
        "standard_error": sems,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json_new(artifact, output_path)
    print(
        json.dumps(
            {
                "event": "complete",
                "q": jepa_config.feed.visible_mode_probability,
                **means,
            },
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
