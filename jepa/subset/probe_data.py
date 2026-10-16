def states_for_positions(pool, targets, count, seed, width):
    import torch
    from jepa.four_image.probes import _sample_excluding, _class_indices
    task_targets = targets.repeat(width)
    positions = torch.arange(width, dtype=torch.int64).repeat_interleave(targets.numel())
    generator = torch.Generator().manual_seed(seed)
    labels = torch.cat((torch.ones(count // 2), torch.zeros(count // 2))).repeat(len(task_targets), 1)
    states = torch.empty(len(task_targets), count, width, dtype=torch.int64)
    by_class = _class_indices(pool)
    for task in range(len(task_targets)):
        target, position = int(task_targets[task]), int(positions[task])
        values = _sample_excluding(len(pool.images), target, (count, width), generator=generator)
        values[:count // 2, position] = target
        same_class = by_class[int(pool.labels[target])]
        candidates = same_class[same_class != target]
        values[count // 2:, position] = candidates[torch.randint(len(candidates), (count // 2,), generator=generator)]
        permutation = torch.randperm(count, generator=generator)
        states[task], labels[task] = values[permutation], labels[task, permutation]
        occurrences = (states[task] == target).sum(-1)
        assert torch.equal(occurrences, labels[task].long())
        assert bool((states[task, labels[task].bool(), position] == target).all())
        assert bool((pool.labels[states[task, ~labels[task].bool(), position]] == pool.labels[target]).all())
    return states, labels, task_targets, positions

