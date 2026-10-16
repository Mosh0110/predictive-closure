from __future__ import annotations
import hashlib
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
import numpy as np
import torch
from torch import Tensor

IMAGE_CHANNELS = 3


IMAGE_HEIGHT = 32


IMAGE_WIDTH = 32


FLAT_IMAGE_SIZE = IMAGE_CHANNELS * IMAGE_HEIGHT * IMAGE_WIDTH


NUM_CLASSES = 10


_EXAMPLES_PER_BATCH = 10_000


_TRAIN_BATCHES = tuple(f"data_batch_{index}" for index in range(1, 6))


_TEST_BATCHES = ("test_batch",)


_REQUIRED_FILES = frozenset((*_TRAIN_BATCHES, *_TEST_BATCHES, "batches.meta"))


@dataclass(frozen=True)
class CIFAR10Pool:
    images: Tensor
    labels: Tensor
    split: str

    def __post_init__(self) -> None:
        if self.images.device.type != "cpu" or self.labels.device.type != "cpu":
            raise ValueError("CIFAR10Pool images and labels must remain on CPU")
        if self.images.dtype != torch.uint8:
            raise TypeError(f"images must have dtype torch.uint8, got {self.images.dtype}")
        expected_shape = (self.labels.numel(), IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH)
        if tuple(self.images.shape) != expected_shape:
            raise ValueError(
                f"images must have shape {expected_shape}, got {tuple(self.images.shape)}"
            )
        if self.labels.dtype != torch.int64:
            raise TypeError(f"labels must have dtype torch.int64, got {self.labels.dtype}")
        if self.labels.ndim != 1:
            raise ValueError(f"labels must be rank one, got shape {tuple(self.labels.shape)}")
        if self.labels.numel() == 0:
            raise ValueError("the image pool must not be empty")
        if torch.any((self.labels < 0) | (self.labels >= NUM_CLASSES)):
            raise ValueError("labels must be integers in [0, 9]")

        counts = torch.bincount(self.labels, minlength=NUM_CLASSES)
        if torch.any(counts != counts[0]):
            raise ValueError(f"the pool must be class balanced, got counts {counts.tolist()}")
        if int(counts[0]) < 2:
            raise ValueError("each class needs at least two images so A and B can differ")

    @property
    def examples_per_class(self) -> int:
        return self.labels.numel() // NUM_CLASSES


def _validate_source_indices(name: str, indices: Tensor) -> None:
    if indices.device.type != "cpu":
        raise ValueError(f"{name} must remain on CPU")
    if indices.dtype != torch.int64 or indices.ndim != 1:
        raise TypeError(f"{name} must be a rank-one int64 tensor")
    if indices.numel() == 0:
        raise ValueError(f"{name} must be nonempty")


def source_indices_sha256(indices: Tensor) -> str:
    _validate_source_indices("indices", indices)
    little_endian = indices.numpy().astype("<i8", copy=False)
    return hashlib.sha256(little_endian.tobytes(order="C")).hexdigest()


def _load_pickle(path: Path) -> dict[bytes, object]:
    with path.open("rb") as handle:
        value = pickle.load(handle, encoding="bytes")
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a dictionary, got {type(value).__name__}")
    return value


def _require_keys(payload: dict[bytes, object], required: set[bytes], path: Path) -> None:
    missing = required.difference(payload)
    if missing:
        names = sorted(key.decode("ascii", errors="replace") for key in missing)
        raise ValueError(f"{path} is missing required keys: {names}")


def _validate_layout(root: Path) -> None:
    if not root.is_dir():
        raise NotADirectoryError(f"CIFAR-10 batch root is not a directory: {root}")
    missing = sorted(name for name in _REQUIRED_FILES if not (root / name).is_file())
    if missing:
        raise FileNotFoundError(f"CIFAR-10 batch root {root} is missing files: {missing}")


def _validate_metadata(root: Path) -> None:
    path = root / "batches.meta"
    metadata = _load_pickle(path)
    _require_keys(metadata, {b"label_names", b"num_cases_per_batch", b"num_vis"}, path)

    label_names = metadata[b"label_names"]
    if not isinstance(label_names, list) or len(label_names) != NUM_CLASSES:
        raise ValueError(f"{path} must contain exactly {NUM_CLASSES} label names")
    if int(metadata[b"num_cases_per_batch"]) != _EXAMPLES_PER_BATCH:
        raise ValueError(
            f"{path} num_cases_per_batch must be {_EXAMPLES_PER_BATCH}, "
            f"got {metadata[b'num_cases_per_batch']}"
        )
    if int(metadata[b"num_vis"]) != FLAT_IMAGE_SIZE:
        raise ValueError(
            f"{path} num_vis must be {FLAT_IMAGE_SIZE}, got {metadata[b'num_vis']}"
        )


def _load_batch(path: Path) -> tuple[Tensor, Tensor]:
    payload = _load_pickle(path)
    _require_keys(payload, {b"batch_label", b"data", b"filenames", b"labels"}, path)

    data = payload[b"data"]
    if not isinstance(data, np.ndarray):
        raise TypeError(f"{path} data must be a numpy array, got {type(data).__name__}")
    expected_shape = (_EXAMPLES_PER_BATCH, FLAT_IMAGE_SIZE)
    if data.shape != expected_shape:
        raise ValueError(f"{path} data must have shape {expected_shape}, got {data.shape}")
    if data.dtype != np.uint8:
        raise TypeError(f"{path} data must have dtype uint8, got {data.dtype}")

    labels_object = payload[b"labels"]
    if not isinstance(labels_object, list) or len(labels_object) != _EXAMPLES_PER_BATCH:
        raise ValueError(f"{path} labels must be a list of length {_EXAMPLES_PER_BATCH}")
    labels = torch.tensor(labels_object, dtype=torch.int64)
    if torch.any((labels < 0) | (labels >= NUM_CLASSES)):
        raise ValueError(f"{path} labels must all be integers in [0, 9]")

    filenames = payload[b"filenames"]
    if not isinstance(filenames, list) or len(filenames) != _EXAMPLES_PER_BATCH:
        raise ValueError(f"{path} filenames must be a list of length {_EXAMPLES_PER_BATCH}")

    images = torch.from_numpy(data.copy()).reshape(
        _EXAMPLES_PER_BATCH,
        IMAGE_CHANNELS,
        IMAGE_HEIGHT,
        IMAGE_WIDTH,
    )
    return images, labels


def load_cifar10_python_batches(
    root: str | Path,
    split: Literal["train", "test"] = "train",
) -> CIFAR10Pool:
    root = Path(root)
    if split not in {"train", "test"}:
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")
    _validate_layout(root)
    _validate_metadata(root)

    batch_names = _TRAIN_BATCHES if split == "train" else _TEST_BATCHES
    loaded = [_load_batch(root / name) for name in batch_names]
    images = torch.cat([batch[0] for batch in loaded], dim=0)
    labels = torch.cat([batch[1] for batch in loaded], dim=0)

    expected_total = len(batch_names) * _EXAMPLES_PER_BATCH
    expected_shape = (expected_total, IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH)
    if tuple(images.shape) != expected_shape:
        raise AssertionError(f"internal loader error: expected {expected_shape}, got {images.shape}")

    counts = torch.bincount(labels, minlength=NUM_CLASSES)
    expected_per_class = expected_total // NUM_CLASSES
    if expected_total % NUM_CLASSES != 0 or torch.any(counts != expected_per_class):
        raise ValueError(
            f"official {split} split must contain {expected_per_class} images per class, "
            f"got {counts.tolist()}"
        )
    return CIFAR10Pool(images=images, labels=labels, split=split)

