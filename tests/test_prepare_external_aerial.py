import sys
import tempfile
import unittest
import json
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from prepare_external_aerial import (  # noqa: E402
    exclude_boundary_instances,
    prepared_record,
    prepare_arrays,
    voxelize,
)


class PrepareExternalAerialTest(unittest.TestCase):
    def test_prepared_record_requires_every_artifact(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            chm = folder / "chm_test.tif"
            (folder / "metadata.json").write_text(json.dumps({"chm": str(chm)}))
            for name in ("points_0p25m.npz", "crowns_full.gpkg"):
                (folder / name).touch()

            self.assertIsNone(prepared_record(folder))

            (folder / "points.laz").touch()
            chm.touch()
            self.assertEqual(prepared_record(folder)["chm"], str(chm))

    def test_voxelize_prefers_supervised_tree_point(self):
        xyz = np.asarray([[0.01, 0.01, 0.01], [0.02, 0.02, 0.02], [0.30, 0.0, 0.0]])
        intensity = np.asarray([1, 2, 3], dtype=np.float32)
        tree_id = np.asarray([0, 17, 0])
        classification = np.asarray([0, 5, 0], dtype=np.uint8)

        result = voxelize(xyz, intensity, tree_id, classification)

        self.assertEqual(result[0].shape, (2, 3))
        self.assertEqual(result[2].tolist(), [17, 0])

    def test_boundary_instances_are_ignored_not_background(self):
        xy = np.asarray([[0.1, 5.0], [1.0, 5.0], [4.0, 4.0], [5.0, 5.0]])
        ids = np.asarray([1, 1, 2, 2])

        output, excluded = exclude_boundary_instances(xy, ids, (0.0, 0.0, 10.0, 10.0), 0.75)

        self.assertEqual(excluded, [1])
        self.assertEqual(output.tolist(), [-1, -1, 2, 2])

    def test_prepare_arrays_removes_ignored_points(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.bin"
            source.write_bytes(b"source")
            xyz = np.asarray(
                [
                    [0.0, 0.0, 2.0],
                    [1.0, 0.0, 2.0],
                    [1.0, 1.0, 2.0],
                    [0.0, 1.0, 2.0],
                    [2.0, 2.0, 0.0],
                    [3.0, 3.0, 1.0],
                ]
            )
            row = prepare_arrays(
                dataset_id="test",
                collection="TEST",
                source_dataset="test",
                source_path=source,
                folder=root / "prepared",
                xyz=xyz,
                intensity=np.arange(6, dtype=np.float32),
                ids=np.asarray([1, 1, 1, 1, 0, -1]),
                classification=np.asarray([5, 5, 5, 5, 2, 5], dtype=np.uint8),
                ground_class=2,
                boundary_margin=-1.0,
                annotation_method="test",
                licence="test",
                doi="test",
            )
            with np.load(row["output"]) as arrays:
                self.assertFalse(np.any(arrays["tree_id"] < 0))
            self.assertEqual(row["ignored_points_removed"], 1)
            self.assertTrue(Path(row["source_las"]).is_file())


if __name__ == "__main__":
    unittest.main()
