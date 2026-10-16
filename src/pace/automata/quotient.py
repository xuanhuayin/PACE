from types import SimpleNamespace
import numpy as np


def validate(event):
    d, a, tg = map(np.asarray, (event.destination, event.accept, event.token_group))
    groups = event.groups
    if (
        a.ndim != 1
        or a.dtype != bool
        or (not len(a))
        or (d.shape != (len(a), len(groups), 2))
        or (d.dtype.kind not in "iu")
        or (d < -1).any()
        or (d >= len(a)).any()
        or (type(event.start) is not int)
        or (not 0 <= event.start < len(a))
    ):
        raise ValueError("Valid nonempty S,G,2 automaton required")
    if (
        tg.ndim != 1
        or tg.dtype.kind not in "iu"
        or (not len(tg))
        or (sorted((t for group in groups for t in group)) != list(range(len(tg))))
        or any((not group or any((tg[t] != g for t in group)) for g, group in enumerate(groups)))
    ):
        raise ValueError("Full vocabulary partition required")


def mapped_branches(destination, state_class):
    return np.sort(np.where(destination >= 0, state_class[destination.clip(min=0)], -1), axis=-1)


def certify(original, quotient, state_class, group_class):
    """Finite commuting certificate; by induction preserves every word's path count."""
    validate(original)
    validate(quotient)
    sc, gc = (np.asarray(state_class), np.asarray(group_class))
    if (
        sc.shape != original.accept.shape
        or gc.shape != (len(original.groups),)
        or sc.dtype.kind not in "iu"
        or (gc.dtype.kind not in "iu")
        or (set(sc.tolist()) != set(range(len(quotient.accept))))
        or (set(gc.tolist()) != set(range(len(quotient.groups))))
    ):
        raise ValueError("Surjective integer quotient maps required")
    if quotient.start != int(sc[original.start]):
        raise ValueError("Initial state changed")
    if not np.array_equal(original.accept, quotient.accept[sc]):
        raise ValueError("Acceptance changed")
    if not np.array_equal(gc[original.token_group], quotient.token_group):
        raise ValueError("Token labels changed")
    for old_group, new_group in enumerate(gc):
        left = mapped_branches(original.destination[:, old_group], sc)
        right = np.sort(quotient.destination[sc, new_group], axis=-1)
        if not np.array_equal(left, right):
            raise ValueError("Transition multiplicity changed")
    return True


def minimize(event):
    validate(event)
    d = np.asarray(event.destination)
    partition = event.accept.astype(np.int32)
    count = len(np.unique(partition))
    rounds = 0
    while True:
        mapped = mapped_branches(d, partition).reshape(len(partition), -1)
        signature = np.column_stack((partition, mapped))
        _, updated = np.unique(signature, axis=0, return_inverse=True)
        updated = updated.astype(np.int32)
        new_count = int(updated.max()) + 1
        partition = updated
        rounds += 1
        if new_count == count:
            break
        count = new_count
    representatives = np.full(new_count, len(partition), dtype=np.int64)
    np.minimum.at(representatives, partition, np.arange(len(partition)))
    compressed = mapped_branches(d[representatives], partition)
    signatures, groups, columns, group_class = ({}, [], [], [])
    for old_group, tokens in enumerate(event.groups):
        key = compressed[:, old_group, :].tobytes()
        if key not in signatures:
            signatures[key] = len(groups)
            groups.append([])
            columns.append(old_group)
        group = signatures[key]
        groups[group].extend(tokens)
        group_class.append(group)
    group_class = np.asarray(group_class, dtype=np.int32)
    quotient = SimpleNamespace(
        destination=compressed[:, columns, :].copy(),
        accept=event.accept[representatives].copy(),
        start=int(partition[event.start]),
        token_group=group_class[event.token_group],
        groups=tuple((tuple(sorted(g)) for g in groups)),
    )
    certify(event, quotient, partition, group_class)
    return (quotient, partition, group_class, rounds)
