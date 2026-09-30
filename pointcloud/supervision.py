"""Explicit annotation validity, separate from vegetation/background semantics."""
import numpy as np


def annotation_targets(ids, classification, *, partial_zero=False, bush_ids=(), height=None):
    """Return instance (-1 ignore/0 background/>0 tree) and semantic targets.

    Classes follow the FOR-instance schema, NOT standard ASPRS class meanings.
    WildForest's unlabelled zero is unknown; its documented z=0 is terrain.
    """
    ids = np.asarray(ids, dtype=np.int64)
    classes = np.asarray(classification)
    instance = np.full(len(ids), -1, dtype=np.int32)
    semantic = np.full(len(ids), -1, dtype=np.int8)
    trusted_background = np.isin(classes, [1, 2])
    if partial_zero and height is not None:
        trusted_background |= (ids == 0) & (np.abs(height) <= .05)
    trees = (ids > 0) & ~np.isin(classes, [1, 2, 3])
    instance[trees] = ids[trees]
    semantic[trees | np.isin(classes, [4, 5, 6])] = 1
    bushes = np.isin(ids, list(bush_ids)) if len(bush_ids) else np.zeros(len(ids), bool)
    trusted_background |= bushes
    instance[trusted_background] = 0
    semantic[trusted_background] = 0
    instance[classes == 3] = -1
    semantic[classes == 3] = -1
    return instance, semantic
