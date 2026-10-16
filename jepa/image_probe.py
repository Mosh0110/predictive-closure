import argparse
from dataclasses import asdict
import hashlib
from pathlib import Path
from types import SimpleNamespace
import torch
from .common import (atomic_save_new, atomic_write_json_new, configure_cpu_threads,
                     cpu_state_dict, dataset_file_hashes, load_json_object,
                     module_state_sha256, resolve_device, set_seed)
from .data import load_cifar10_python_batches
from .four_image import train_jepa, train_steering
from .four_image.corpus import tensor_sha256
from .four_image.feed import select_identity_pool
from .four_image.models import FourImageJEPA
from .four_image.side_model import build_model
from .four_image.probes import (IndependentMLPBank, _select_targets, _membership_states,
                                _encode_states, _train_membership)
from .subset import train_jepa as subset_train
from .subset.encoding import encode_states
from .subset.models import RingBudgetJEPA
from .subset.probe_data import states_for_positions


def load_encoder(path, data_root, device):
    raw = torch.load(path, map_location='cpu', weights_only=False)
    module = {train_jepa.CHECKPOINT_KIND: train_jepa,
              train_steering.CHECKPOINT_KIND: train_steering,
              subset_train.CHECKPOINT_KIND: subset_train}[raw['artifact_kind']]
    config = module.Config.from_mapping(raw['config'])
    assert raw['completed_steps'] == config.steps
    assert raw['source_hashes'] == module._source_hashes()
    _, hashes = dataset_file_hashes(data_root)
    assert raw['dataset']['file_sha256'] == hashes
    kwargs = dict(sequence_length=config.feed.sequence_length, latent_dim=config.latent_dim,
                  base_channels=config.encoder_base_channels,
                  representation_width=config.encoder_representation_width,
                  predictor_hidden_dim=config.predictor_hidden_dim,
                  encoder_frame_chunk_size=config.encoder_frame_chunk_size)
    if module is train_steering:
        model = build_model(config).to(device)
    elif module is subset_train:
        model = RingBudgetJEPA(**kwargs).to(device)
    else:
        model = FourImageJEPA(**kwargs).to(device)
    model.load_state_dict(raw['model_state_dict'], strict=True)
    return model.encoder.eval().requires_grad_(False), config, module is subset_train


def encode(encoder, pool, states, jc, pc, device, subset):
    if subset:
        return encode_states(encoder, pool, states, ring_size=jc.feed.bd_ring_size,
                             bd_render_tile_size=jc.feed.bd_render_tile_size,
                             bd_layout=jc.feed.bd_layout, batch_size=pc.encoding_batch_size,
                             device=device, bf16=pc.bf16)
    return _encode_states(encoder, pool, states, batch_size=pc.encoding_batch_size,
                          device=device, bf16=pc.bf16, render_layout=jc.render_layout)


def run(settings):
    pc = SimpleNamespace(**settings)
    assert pc.probe_hidden_dims == [16]
    assert pc.evaluation_split == 'test'
    assert len({pc.membership_train_seed, pc.membership_validation_seed, pc.membership_test_seed}) == 3
    assert not Path(pc.output_json).exists() and not Path(pc.output_checkpoint).exists()
    configure_cpu_threads(pc.cpu_threads, pc.cpu_interop_threads)
    device = resolve_device(pc.device, bf16=pc.bf16)
    set_seed(pc.seed)
    checkpoint = Path(pc.jepa_checkpoint)
    checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    encoder, jc, subset = load_encoder(checkpoint, pc.data_root, device)
    before = module_state_sha256(encoder)
    pool, indices = select_identity_pool(load_cifar10_python_batches(pc.data_root, split='test'),
        identity_count=pc.evaluation_identity_count, seed=pc.evaluation_identity_subset_seed)
    targets = _select_targets(pool, pc.targets_per_class, pc.target_identity_seed)
    width = jc.feed.bd_ring_size + 2 if subset else 4
    splits, hashes, examples = {}, {}, {}
    for name in ('train', 'validation', 'test'):
        count = getattr(pc, f'membership_{name}_samples_per_task')
        seed = getattr(pc, f'membership_{name}_seed')
        if subset:
            states, labels, tt, positions = states_for_positions(pool, targets, count, seed, width)
        else:
            states, labels, tt, positions = _membership_states(pool=pool, targets=targets,
                                                             samples_per_task=count, seed=seed)
        hashes[name] = dict(states=tensor_sha256(states), labels=tensor_sha256(labels))
        positive = int(torch.nonzero(labels[0], as_tuple=False)[0])
        negative = int(torch.nonzero(~labels[0].bool(), as_tuple=False)[0])
        examples[name] = dict(target=int(tt[0]), position=int(positions[0]),
                              positive=states[0,positive].tolist(), negative=states[0,negative].tolist())
        splits[name] = dict(inputs=encode(encoder, pool, states, jc, pc, device, subset), labels=labels)
    generator = torch.Generator().manual_seed(pc.standardization_seed)
    shape = (1, pc.standardization_samples, width) if subset else (pc.standardization_samples, width)
    states = torch.randint(len(pool.images), shape, generator=generator)
    values = encode(encoder, pool, states, jc, pc, device, subset)
    if subset:
        values = values[0]
    center, scale = values.mean(0, keepdim=True), values.std(0, unbiased=False, keepdim=True)
    assert bool((scale > 1e-8).all())
    for split in splits.values():
        split['inputs'] = (split['inputs'] - center) / scale
    model = IndependentMLPBank(len(targets)*width, jc.latent_dim, 16, 1)
    pc.membership_probe_steps = pc.probe_steps
    curve = _train_membership(model, splits['train']['inputs'], splits['train']['labels'],
                              config=pc, device=device)
    model.eval()
    names = tuple(chr(65+i) for i in range(width)) if subset else ('A', 'C', 'B', 'D')
    evaluations = {}
    with torch.no_grad():
        for name, split in splits.items():
            logits = model(split['inputs'].to(device)).squeeze(-1).float().cpu()
            accuracy = ((logits >= 0) == split['labels'].bool()).float().mean(1)
            evaluations[name] = [dict(position=names[int(p)], target=int(t), accuracy=float(a))
                                  for p, t, a in zip(positions, tt, accuracy, strict=True)]
    assert module_state_sha256(encoder) == before
    assert all(p.grad is None for p in encoder.parameters())
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == checkpoint_hash
    result = dict(config=settings, training_config=asdict(jc), encoder_sha256=checkpoint_hash,
                  encoder_unchanged=True, data_hashes=hashes, raw_examples=examples,
                  target_ids=targets.tolist(), target_source_ids=indices[targets].tolist(),
                  training_curve=curve, evaluations=evaluations,
                  standardization=dict(mean=center.tolist(), std=scale.tolist()))
    atomic_save_new(dict(probe_state_dict=cpu_state_dict(model), mean=center, std=scale,
                         result=result), Path(pc.output_checkpoint))
    atomic_write_json_new(result, Path(pc.output_json))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    run(load_json_object(parser.parse_args().config))
