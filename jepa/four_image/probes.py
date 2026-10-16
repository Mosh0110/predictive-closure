from __future__ import annotations
import json
import math
from typing import Any
import torch
import torch.nn.functional as F
from torch import nn
from jepa.data import CIFAR10Pool
from .feed import render_collages
from .models import parameter_count

class IndependentMLPBank(nn.Module):
    def __init__(self, task_count: int, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        for name, value in (
            ("task_count", task_count),
            ("input_dim", input_dim),
            ("hidden_dim", hidden_dim),
            ("output_dim", output_dim),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be positive")
        self.task_count = task_count
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight_one = nn.Parameter(torch.empty(task_count, input_dim, hidden_dim))
        self.bias_one = nn.Parameter(torch.empty(task_count, hidden_dim))
        self.weight_two = nn.Parameter(torch.empty(task_count, hidden_dim, hidden_dim))
        self.bias_two = nn.Parameter(torch.empty(task_count, hidden_dim))
        self.weight_three = nn.Parameter(torch.empty(task_count, hidden_dim, output_dim))
        self.bias_three = nn.Parameter(torch.empty(task_count, output_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for task in range(self.task_count):
            nn.init.kaiming_uniform_(self.weight_one[task], a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.weight_two[task], a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.weight_three[task], a=math.sqrt(5))
            for weight, bias in (
                (self.weight_one[task], self.bias_one[task]),
                (self.weight_two[task], self.bias_two[task]),
                (self.weight_three[task], self.bias_three[task]),
            ):
                bound = 1 / math.sqrt(weight.shape[0])
                nn.init.uniform_(bias, -bound, bound)

    @property
    def parameters_per_task(self) -> int:
        return parameter_count(self) // self.task_count

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 3 or inputs.shape[0] != self.task_count or inputs.shape[2] != self.input_dim:
            raise ValueError("bank inputs have the wrong shape")
        hidden = torch.einsum("tbi,tih->tbh", inputs, self.weight_one) + self.bias_one[:, None]
        hidden = F.gelu(hidden)
        hidden = torch.einsum("tbh,thk->tbk", hidden, self.weight_two) + self.bias_two[:, None]
        hidden = F.gelu(hidden)
        return torch.einsum("tbh,tho->tbo", hidden, self.weight_three) + self.bias_three[:, None]


def _class_indices(pool: CIFAR10Pool) -> list[torch.Tensor]:
    return [
        torch.nonzero(pool.labels == label, as_tuple=False).flatten()
        for label in range(10)
    ]


def _select_targets(pool: CIFAR10Pool, per_class: int, seed: int) -> torch.Tensor:
    by_class = _class_indices(pool)
    if any(indices.numel() < per_class + 1 for indices in by_class):
        raise ValueError("not enough identities per class for membership targets")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    selected = []
    for indices in by_class:
        permutation = torch.randperm(indices.numel(), generator=generator)
        selected.append(indices[permutation[:per_class]])
    return torch.cat(selected)


def _sample_excluding(
    state_count: int,
    excluded: int,
    shape: tuple[int, ...],
    *,
    generator: torch.Generator,
) -> torch.Tensor:
    raw = torch.randint(state_count - 1, shape, generator=generator)
    return raw + (raw >= excluded).to(torch.int64)


def _membership_states(
    *,
    pool: CIFAR10Pool,
    targets: torch.Tensor,
    samples_per_task: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    task_targets = targets.repeat(4)
    task_positions = torch.arange(4, dtype=torch.int64).repeat_interleave(targets.numel())
    task_count = task_targets.numel()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    labels = torch.cat(
        (
            torch.ones(samples_per_task // 2),
            torch.zeros(samples_per_task // 2),
        )
    ).repeat(task_count, 1)
    states = torch.empty(task_count, samples_per_task, 4, dtype=torch.int64)
    by_class = _class_indices(pool)
    for task in range(task_count):
        target = int(task_targets[task])
        position = int(task_positions[task])
        values = _sample_excluding(
            pool.images.shape[0],
            target,
            (samples_per_task, 4),
            generator=generator,
        )
        values[: samples_per_task // 2, position] = target
        same_class = by_class[int(pool.labels[target])]
        negative_candidates = same_class[same_class != target]
        negative_indices = torch.randint(
            negative_candidates.numel(),
            (samples_per_task // 2,),
            generator=generator,
        )
        values[samples_per_task // 2 :, position] = negative_candidates[negative_indices]
        permutation = torch.randperm(samples_per_task, generator=generator)
        states[task] = values[permutation]
        labels[task] = labels[task, permutation]
    return states, labels, task_targets, task_positions


@torch.no_grad()
def _encode_states(
    encoder: nn.Module,
    pool: CIFAR10Pool,
    states: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
    bf16: bool,
    render_layout: str = "columns",
) -> torch.Tensor:
    leading = states.shape[:-1]
    flat = states.reshape(-1, 4)
    results = []
    for start in range(0, flat.shape[0], batch_size):
        indices = flat[start : start + batch_size]
        images = pool.images[indices].to(device, dtype=torch.float32).div_(255.0)
        frames = render_collages(images[:, None], layout=render_layout)[:, 0]
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=bf16):
            results.append(encoder(frames).float().cpu())
    encoded = torch.cat(results).reshape(*leading, -1)
    return encoded


def _lr(config: Any, step_index: int, total_steps: int) -> float:
    if step_index < config.warmup_steps:
        return config.learning_rate * (step_index + 1) / config.warmup_steps
    progress = (step_index - config.warmup_steps) / max(
        1, total_steps - config.warmup_steps - 1
    )
    return config.minimum_learning_rate + 0.5 * (
        config.learning_rate - config.minimum_learning_rate
    ) * (1.0 + math.cos(math.pi * progress))


def _gather_task_batches(inputs: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    return inputs.gather(1, indices[:, :, None].expand(-1, -1, inputs.shape[2]))


def _train_membership(
    model: IndependentMLPBank,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    *,
    config: Any,
    device: torch.device,
) -> list[dict[str, float]]:
    model.to(device).train()
    inputs = inputs.to(device)
    labels = labels.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    generator = torch.Generator(device="cpu").manual_seed(config.seed + 100)
    curve = []
    for step_index in range(config.membership_probe_steps):
        indices = torch.randint(
            inputs.shape[1],
            (inputs.shape[0], config.membership_batch_size_per_task),
            generator=generator,
        ).to(device)
        learning_rate = _lr(config, step_index, config.membership_probe_steps)
        optimizer.param_groups[0]["lr"] = learning_rate
        logits = model(_gather_task_batches(inputs, indices)).squeeze(-1)
        target = labels.gather(1, indices)
        loss = F.binary_cross_entropy_with_logits(logits, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip, error_if_nonfinite=True)
        optimizer.step()
        step = step_index + 1
        if step == 1 or step % config.log_every == 0 or step == config.membership_probe_steps:
            record = {
                "step": float(step),
                "loss": float(loss.detach()),
                "gradient_norm": float(gradient_norm.detach()),
                "learning_rate": learning_rate,
            }
            curve.append(record)
            print(json.dumps({"event": "membership_probe", **record}, sort_keys=True), flush=True)
    return curve


