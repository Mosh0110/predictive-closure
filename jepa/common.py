from __future__ import annotations
import hashlib
import json
import math
import os
import random
import tempfile
from dataclasses import fields
from pathlib import Path
from typing import Any, Mapping, TypeVar
import numpy as np
import torch
from torch import nn


DATASET_FILENAMES = (
    "data_batch_1",
    "data_batch_2",
    "data_batch_3",
    "data_batch_4",
    "data_batch_5",
    "test_batch",
    "batches.meta",
)

T = TypeVar("T")


def require_integer(name: str, value: Any, *, minimum: int) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")
    return value


def require_seed(name: str, value: Any) -> int:
    require_integer(name, value, minimum=0)
    if value >= 2**63:
        raise ValueError(f"{name} must be smaller than 2^63, got {value}")
    return value


def require_finite(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric, got {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return result


def strict_dataclass_mapping(cls: type[T], value: Mapping[str, Any], name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be an object, got {type(value).__name__}")
    expected = {field.name for field in fields(cls)}
    observed = set(value)
    missing = sorted(expected - observed)
    extra = sorted(observed - expected)
    if missing or extra:
        raise ValueError(
            f"{name} keys differ from schema; missing={missing}, extra={extra}"
        )
    return dict(value)


def load_json_object(path: str | Path) -> dict[str, Any]:
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"JSON file does not exist: {path}")

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result

    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle, object_pairs_hook=reject_duplicate_keys)
    if type(value) is not dict:
        raise TypeError(f"{path} must contain a JSON object")
    return value


def resolve_device(device_spec: str, *, bf16: bool) -> torch.device:
    if not isinstance(device_spec, str) or not device_spec:
        raise ValueError("device must be a nonempty string")
    device = torch.device(device_spec)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError(f"only CPU and CUDA devices are supported, got {device}")
    if device.type == "cuda":
        if device.index is None:
            raise ValueError("CUDA device must include an index, for example cuda:0")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        torch.cuda.set_device(device)
        if bf16 and not torch.cuda.is_bf16_supported():
            raise RuntimeError(f"{device} does not support bfloat16")
    elif bf16:
        raise ValueError("bf16 requires a CUDA device")
    return device


def set_seed(seed: int) -> None:
    require_seed("seed", seed)
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def source_hashes(paths: Mapping[str, Path]) -> dict[str, str]:
    result: dict[str, str] = {}
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"source file is missing: {path}")
        result[name] = sha256_bytes(path.read_bytes())
    return result


def dataset_file_hashes(data_root: str | Path) -> tuple[Path, dict[str, str]]:
    root = Path(data_root).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(f"dataset root is not a directory: {root}")
    result: dict[str, str] = {}
    for filename in DATASET_FILENAMES:
        path = root / filename
        if not path.is_file():
            raise FileNotFoundError(f"dataset file is missing: {path}")
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        result[filename] = digest.hexdigest()
    return root, result


def cpu_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in module.state_dict().items()
    }


def atomic_save_new(payload: Mapping[str, Any], path: str | Path) -> str:
    output_path = Path(path).expanduser()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite output: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            torch.save(dict(payload), handle)
            handle.flush()
            os.fsync(handle.fileno())
        digest = hashlib.sha256(temporary_path.read_bytes()).hexdigest()
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return digest


def atomic_write_json_new(payload: Mapping[str, Any], path: str | Path) -> None:
    output_path = Path(path).expanduser()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite output: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _base_runtime_contract(device: torch.device) -> dict[str, Any]:
    return {
        "torch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "device": str(device),
        "device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
        "device_capability": (
            list(torch.cuda.get_device_capability(device))
            if device.type == "cuda"
            else None
        ),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
    }


def configure_cpu_threads(num_threads: int, num_interop_threads: int) -> None:
    for name, value in (
        ("num_threads", num_threads),
        ("num_interop_threads", num_interop_threads),
    ):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer, got {value!r}")
    torch.set_num_threads(num_threads)
    if torch.get_num_interop_threads() != num_interop_threads:
        torch.set_num_interop_threads(num_interop_threads)
    if torch.get_num_threads() != num_threads:
        raise RuntimeError("PyTorch intra-op thread setting did not take effect")
    if torch.get_num_interop_threads() != num_interop_threads:
        raise RuntimeError("PyTorch inter-op thread setting did not take effect")


def runtime_contract(device: torch.device) -> dict[str, object]:
    result = _base_runtime_contract(device)
    result["torch_num_threads"] = torch.get_num_threads()
    result["torch_num_interop_threads"] = torch.get_num_interop_threads()
    return result


def module_state_sha256(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in module.state_dict().items():
        value = tensor.detach().cpu().contiguous()
        for field in (name, str(value.dtype), repr(tuple(value.shape))):
            encoded = field.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "little"))
            digest.update(encoded)
        raw = value.reshape(-1).view(torch.uint8).numpy().tobytes()
        digest.update(len(raw).to_bytes(8, "little"))
        digest.update(raw)
    return digest.hexdigest()

