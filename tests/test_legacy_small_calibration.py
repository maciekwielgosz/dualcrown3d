import sys
import unittest
from pathlib import Path

import numpy as np
from shapely.geometry import box

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pointcloud.instance_output import _legacy_merge
from scripts.calibrate_legacy_small_trees import small_hits,variants


class LegacySmallTreeCalibrationTests(unittest.TestCase):
    def test_small_crown_matching_uses_iou_and_reference_area(self):
        result=small_hits([box(0,0,1,1),box(4,0,8,4)],
                          [box(0,0,1,1),box(4,0,5,1)])
        self.assertEqual(result['small_4_gt'],1)
        self.assertEqual(result['small_4_tp'],1)
        self.assertEqual(result['small_10_gt'],1)
        self.assertEqual(result['small_10_tp'],1)

    def test_relaxed_filters_do_not_change_the_baseline_defaults(self):
        config=variants()['baseline']
        self.assertEqual(config['minimum_voxels'],12)
        self.assertEqual(config['vote_cluster_config']['min_voxels'],12)
        self.assertNotIn('minimum_height_m',config)
        self.assertNotIn('minimum_area_m2',config)
        xyz=np.array([[x*.25,y*.25,1.6] for x in range(4) for y in range(2)],np.float32)
        arrays=dict(coord=xyz,source_origin=np.zeros(3),voxel_size=np.float32(.25))
        candidate=[(1.,np.arange(8),np.ones(8,np.float32),1.)]
        strict={**config,'minimum_voxels':8}
        labels,_,items=_legacy_merge(arrays,candidate,strict)
        self.assertFalse(labels.any())
        self.assertEqual(items,[])
        relaxed={**strict,'minimum_height_m':1.5,'minimum_area_m2':.4}
        labels,_,items=_legacy_merge(arrays,candidate,relaxed)
        self.assertTrue((labels==1).all())
        self.assertEqual(len(items),1)


if __name__=='__main__':
    unittest.main()
