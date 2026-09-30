import unittest
import numpy as np
import torch

from pointcloud.supervision import annotation_targets
from pointcloud.decoder_v4 import HybridTreeMaskDecoder, set_mask_losses
from pointcloud.adaptive_fusion import revise_anchors
from pointcloud.instance_output import point_instance_metrics
from pointcloud.data import prepare_crop


class SupervisionTests(unittest.TestCase):
    def test_unknown_outside_and_known_background_are_distinct(self):
        ids,semantic=annotation_targets([0,0,1,2,0,0],[3,2,5,0,0,4],bush_ids=[2])
        np.testing.assert_array_equal(ids,[-1,0,1,0,-1,-1])
        np.testing.assert_array_equal(semantic,[-1,0,1,0,-1,1])

    def test_unknown_predictions_do_not_count_as_false_positives(self):
        result=point_instance_metrics(np.array([-1,-1,0,1,1]),np.array([7,7,0,9,9]))
        self.assertEqual((result['tp'],result['fp'],result['fn']),(1,0,0))
        result=point_instance_metrics(np.array([-1,-1,0,1,1]),np.array([7,7,8,9,9]))
        self.assertEqual(result['fp'],1)

    def test_crop_keeps_semantic_targets_aligned_after_density_and_geometry(self):
        rng=np.random.default_rng(4);xyz=rng.uniform(0,10,(600,3)).astype(np.float32)
        ids=np.arange(1,601,dtype=np.int64)
        a=dict(coord=xyz,intensity=np.ones(600,np.float32),tree_id=ids,
               instance_offset=np.zeros_like(xyz),voxel_size=np.float32(.25),semantic_target=ids%2)
        b=prepare_crop(a,rng,20,500,True,(.5,),True)
        torch.testing.assert_close(b['semantic_target'],b['tree_id']%2)

    def test_hybrid_can_segment_semantic_false_negatives_and_ignores_gt_at_eval(self):
        torch.manual_seed(8)
        decoder=HybridTreeMaskDecoder(queries=8,layers=1,memory_tokens=32).eval()
        xyz=torch.rand(50,3)*5
        data=dict(coord=xyz,feat=torch.cat((xyz,torch.ones(50,1)),1),tree_id=torch.ones(50,dtype=torch.long))
        features=torch.rand(50,72);logits=torch.tensor([8.,-8.]).repeat(50,1);offset=torch.zeros(50,3)
        a=decoder(features,data,logits,offset)
        b=decoder(features,{**data,'tree_id':torch.zeros(50,dtype=torch.long)},logits,offset)
        torch.testing.assert_close(a['mask_logits'],b['mask_logits'])
        self.assertTrue(torch.isfinite(a['mask_logits']).all())
        self.assertTrue((a['mask_logits']>-100).all())

    def test_hungarian_masks_cover_nonseeded_tree_and_ignore_unknown_points(self):
        logits=torch.tensor([[8.,8.,-8.,-8.,1.],[-8.,-8.,8.,8.,1.]],requires_grad=True)
        ids=torch.tensor([1,1,2,2,-1]).repeat_interleave(2)
        logits=logits.repeat_interleave(2,dim=1).detach().requires_grad_()
        output=dict(mask_logits=logits,object_logits=torch.zeros(2,requires_grad=True),
                    embedding=torch.randn(10,5,requires_grad=True),seed_index=torch.tensor([0,1]))
        loss=set_mask_losses(output,ids)['mask_loss'];loss.backward()
        self.assertLess(float(loss.detach()),.3)
        self.assertEqual(float(logits.grad[:,8:].abs().sum()),0.)

    def test_split_and_merge_can_revise_assigned_instances(self):
        xyz=np.array([[x,y,5.] for x in np.arange(0,4,.25) for y in (0.,.25)],np.float32)
        mask=np.where(xyz[:,0]<2,1,2).astype(np.uint32)
        config=dict(minimum_voxels=4,revision_confidence=.4,revision_mode='split')
        labels,source=revise_anchors(xyz,np.ones(len(xyz),np.uint32),mask,np.full(len(xyz),.9),config)
        self.assertEqual(len(np.unique(labels)),2);self.assertTrue((source==6).all())
        labels,source=revise_anchors(xyz,mask,np.ones(len(xyz),np.uint32),np.full(len(xyz),.9),
                                    {**config,'revision_mode':'merge'})
        self.assertEqual(len(np.unique(labels)),1);self.assertTrue((source==7).all())
        labels,_=revise_anchors(xyz,mask,np.ones(len(xyz),np.uint32),np.full(len(xyz),.1),config)
        np.testing.assert_array_equal(labels,mask)


if __name__=='__main__':unittest.main()
