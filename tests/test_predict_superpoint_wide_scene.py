"""Small checks for resumable whole-scene EZ-SP export helpers."""
import numpy as np
import pytest

from scripts.predict_superpoint_wide_scene import collect_stripes, save_stripe, support_polygon


def test_support_polygon_keeps_separate_crowns_and_fills_small_hole():
    first = np.array([(x, y) for x in np.arange(0, 2, .5)
                      for y in np.arange(0, 2, .5) if (x, y) != (.5, .5)])
    second = first + np.array([8., 0.])
    geometry = support_polygon(np.vstack((first, second)))
    assert geometry.is_valid
    assert len(geometry.geoms) == 2
    assert geometry.contains(__import__('shapely').geometry.Point(.75, .75))


def test_stripe_collection_checks_coverage_and_offsets(tmp_path):
    folder = tmp_path / 'work' / 'stripes'
    folder.mkdir(parents=True)
    save_stripe(folder / 'stripe_000.npz', 'same',
                [np.array([0, 1], np.int32)], [np.array([.8, .9], np.float32)],
                [0, 2], [np.array([0, 1], np.int32)],
                [np.array([.7, .8], np.float16)], [.6], 1)
    save_stripe(folder / 'stripe_001.npz', 'same',
                [np.array([2, 3], np.int32)], [np.array([.3, .4], np.float32)],
                [0, 1], [np.array([2], np.int32)],
                [np.array([.9], np.float16)], [.5], 1)
    raw, windows = collect_stripes(tmp_path, 'same', 2, 4)
    assert windows == 2
    assert raw['candidate_offset'].tolist() == [0, 2, 3]
    assert raw['point_index'].tolist() == [0, 1, 2]
    assert np.allclose(raw['point_probability'], [.8, .9, .3, .4])
    with pytest.raises(ValueError, match='signature mismatch'):
        collect_stripes(tmp_path, 'other', 2, 4)
