from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import torch
from jepa.common import atomic_save_new, atomic_write_json_new, dataset_file_hashes
from jepa.data import load_cifar10_python_batches
from .corpus import ARTIFACT_KIND, FORMAT_VERSION, tensor_sha256
from .feed import DictionaryFeedConfig, build_dictionaries, dictionary_sha256, identity_pool_sha256, sample_state_sequences, select_identity_pool

def materialize(
    *,
    data_root: str,
    output_corpus: Path,
    output_json: Path,
    feed_config: DictionaryFeedConfig,
    corpus_seed: int,
    epoch_count: int,
    sequences_per_epoch: int,
) -> dict[str, object]:
    if output_corpus.exists() or output_json.exists():
        raise FileExistsError("refusing to overwrite a corpus artifact")
    if type(epoch_count) is not int or not 1 <= epoch_count <= 10_000:
        raise ValueError("epoch_count must be in [1,10000]")
    if type(sequences_per_epoch) is not int or sequences_per_epoch <= 0:
        raise ValueError("sequences_per_epoch must be positive")
    if type(corpus_seed) is not int or not 0 <= corpus_seed < 2**63:
        raise ValueError("corpus_seed must be an integer in [0,2^63)")
    root, dataset_hashes = dataset_file_hashes(data_root)
    full_pool = load_cifar10_python_batches(root, split="train")
    pool, identity_source_indices = select_identity_pool(
        full_pool,
        identity_count=feed_config.identity_count,
        seed=feed_config.identity_subset_seed,
    )
    state_count = int(pool.images.shape[0])
    if state_count > 2**16:
        raise ValueError("uint16 corpus storage cannot hold this identity pool")
    dictionaries = build_dictionaries(
        state_count,
        feed_config.dictionary_seed,
        mode=feed_config.dictionary_mode,
    )
    states = torch.empty(
        epoch_count,
        sequences_per_epoch,
        feed_config.sequence_length,
        4,
        dtype=torch.uint16,
    )
    branches = torch.empty(
        epoch_count,
        sequences_per_epoch,
        feed_config.sequence_length - 1,
        4,
        dtype=torch.bool,
    )
    generator = torch.Generator(device="cpu").manual_seed(corpus_seed)
    for epoch in range(epoch_count):
        epoch_states, epoch_branches, _, _ = sample_state_sequences(
            sequences_per_epoch,
            state_count,
            dictionaries,
            feed_config,
            generator=generator,
        )
        states[epoch].copy_(epoch_states.to(torch.uint16))
        branches[epoch].copy_(epoch_branches)
        print(
            {
                "event": "materialized_epoch",
                "epoch": epoch + 1,
                "epoch_count": epoch_count,
            },
            flush=True,
        )
    artifact = {
        "format_version": FORMAT_VERSION,
        "artifact_kind": ARTIFACT_KIND,
        "feed": asdict(feed_config),
        "state_count": state_count,
        "identity_source_indices": identity_source_indices,
        "identity_source_indices_sha256": identity_pool_sha256(
            identity_source_indices
        ),
        "corpus_seed": corpus_seed,
        "epoch_count": epoch_count,
        "sequences_per_epoch": sequences_per_epoch,
        "states": states,
        "dictionary_branch": branches,
        "states_sha256": tensor_sha256(states),
        "dictionary_branch_sha256": tensor_sha256(branches),
        "dictionary_sha256": dictionary_sha256(dictionaries),
        "dataset_file_sha256": dataset_hashes,
    }
    metadata = {
        key: value
        for key, value in artifact.items()
        if key not in {"states", "dictionary_branch", "identity_source_indices"}
    }
    metadata["storage"] = {
        "states_shape": list(states.shape),
        "states_dtype": str(states.dtype),
        "states_bytes": states.numel() * states.element_size(),
        "dictionary_branch_shape": list(branches.shape),
        "dictionary_branch_dtype": str(branches.dtype),
        "dictionary_branch_bytes": branches.numel() * branches.element_size(),
        "total_tensor_bytes": (
            states.numel() * states.element_size()
            + branches.numel() * branches.element_size()
        ),
    }
    atomic_save_new(artifact, output_corpus)
    atomic_write_json_new(metadata, output_json)
    return artifact

