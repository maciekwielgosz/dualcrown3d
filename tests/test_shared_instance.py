import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from shapely.geometry import box

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pointcloud.data import prepare_crop
from pointcloud.dual_head import DualHeadLitePT
from pointcloud.shared_instance import polygon_targets, shared_instance_losses
from pointcloud.shared_merge import _geometry_from_cells, merge_shared_queries
from scripts.predict_dual_head import predict


class FakeBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer=nn.Linear(4,72)
    def forward(self,data):
        return SimpleNamespace(feat=self.layer(data['feat']))


class FakeLegacy(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone=FakeBackbone()
        self.semantic_head=nn.Linear(72,2)
        self.offset_head=nn.Linear(72,3)
        self.offset_scale_m=10.


class SharedInstanceTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.coord=torch.rand(120,3)*8
        self.coord[:,2]+=2
        self.ids=torch.where(self.coord[:,0]<4,1,2)

    def test_shared_queries_and_both_heads_receive_gradients(self):
        model=DualHeadLitePT(legacy=FakeLegacy(),queries=8,decoder_layers=1,
                             memory_tokens=32,decoder_policy='shared_v5')
        model.configure_training('mask')
        batch=dict(coord=self.coord,feat=torch.cat((self.coord,torch.ones(120,1)),1),
                   offset=torch.tensor([120]),tree_id=self.ids,
                   semantic_target=torch.ones(120,dtype=torch.long),
                   instance_offset=torch.zeros(120,3),local_to_world=torch.eye(3,dtype=torch.double))
        ids,masks,valid=polygon_targets(batch['coord'],batch['semantic_target'],
                                         batch['tree_id'],batch['local_to_world'],
                                         {1:box(0,0,4,8),2:box(4,0,8,8)})
        batch.update(crown_ids=ids,crown_target=masks,crown_valid=valid)
        output=model(batch)
        self.assertEqual(output['instance_masks']['mask_logits'].shape[0],
                         output['instance_masks']['crown_logits'].shape[0])
        losses=shared_instance_losses(output,batch)
        losses['loss'].backward()
        self.assertTrue(torch.isfinite(losses['loss']))
        self.assertGreater(float(model.point_decoder.crown.query.weight.grad.abs().sum()),0)
        self.assertGreater(float(model.point_decoder.mask_query.weight.grad.abs().sum()),0)

    def test_full_polygon_is_not_only_the_observed_point_footprint(self):
        coord=np.array([[1,1,5],[1.25,1,5],[2,2,0],[4,4,0]],np.float32)
        data=dict(coord=coord,tree_id=np.array([1,1,0,0]),intensity=np.ones(4,np.float32),
                  instance_offset=np.zeros((4,3),np.float32),source_origin=np.array([100.,200.,0.]),
                  voxel_size=np.float32(.25))
        crop=prepare_crop(data,np.random.default_rng(0),20,100,False,return_transform=True)
        world=crop['coord'][:,:2].double() @ crop['local_to_world'][:2,:2].T + crop['local_to_world'][:2,2]
        expected={(round(x,2),round(y,2)) for x,y in coord[:,:2]+data['source_origin'][:2]}
        actual={(round(x,2),round(y,2)) for x,y in world.numpy()}
        self.assertEqual(actual,expected)
        ids,masks,valid=polygon_targets(crop['coord'],crop['semantic_target'],
                                         crop['tree_id'],crop['local_to_world'],
                                         {1:box(100,200,104.5,204.5)})
        self.assertEqual(ids.tolist(),[1])
        self.assertGreater(int(masks.sum()),int((crop['tree_id']==1).sum()))
        self.assertTrue(valid[masks[0]].all())

    def fixture(self,proposals):
        xyz=np.asarray([[x*.25,y*.25,5.] for x in range(8) for y in range(6)],np.float32)
        lengths=list(map(len,proposals))
        raw=dict(object_score=np.full(len(proposals),.9,np.float32),
                 candidate_quality=np.full(len(proposals),.8,np.float32),
                 candidate_offset=np.cumsum([0,*lengths]),
                 point_index=np.concatenate(proposals).astype(np.int32),
                 point_score=np.ones(sum(lengths),np.float16),
                 crown_offset=np.cumsum([0,*[12]*len(proposals)]),
                 crown_cell=np.tile(np.arange(12,dtype=np.int64),len(proposals)),
                 crown_score=np.ones(12*len(proposals),np.float16),
                 crown_grid_origin=np.zeros(2),crown_grid_width=np.int64(20),
                 crown_grid_size=np.float32(.5))
        arrays=dict(coord=xyz,source_origin=np.array([100.,200.,0.]),voxel_size=np.float32(.25))
        config=dict(object_threshold=.1,quality_threshold=.1,mask_threshold=.5,
                    crown_threshold=.5,minimum_voxels=12,duplicate_iou=.5,
                    minimum_unique_fraction=.25,crown_output='head_union_support')
        return arrays,raw,config

    def test_crown_cells_produce_a_polygon(self):
        raw=dict(crown_grid_width=np.int64(10),crown_grid_size=np.float32(1.),
                 crown_grid_origin=np.zeros(2))
        cells=np.array([y*10+x for y in range(2,5) for x in range(2,5)])
        crown=_geometry_from_cells(cells,raw)
        self.assertIsNotNone(crown)
        self.assertAlmostEqual(crown.area,9.)
        self.assertEqual(tuple(crown.bounds),(2.,2.,5.,5.))

    def test_no_candidates_returns_unassigned_points(self):
        arrays,raw,config=self.fixture([np.arange(24)])
        raw['object_score'][:]=0.
        ids,confidence,records,source=merge_shared_queries(arrays,raw,config)
        self.assertTrue((ids==0).all())
        self.assertTrue((confidence==0).all())
        self.assertEqual(records,[])
        self.assertTrue((source==0).all())

    def test_duplicate_queries_share_an_id(self):
        arrays,raw,config=self.fixture([np.arange(24),np.arange(24)])
        ids,_,records,_=merge_shared_queries(arrays,raw,config)
        self.assertEqual(len(records),1)
        self.assertEqual(set(np.unique(ids)),{0,1})

    def test_partly_overlapping_distinct_tree_is_kept(self):
        arrays,raw,config=self.fixture([np.arange(24),np.arange(18,48)])
        ids,_,records,_=merge_shared_queries(arrays,raw,config)
        self.assertEqual(len(records),2)
        self.assertEqual(set(np.unique(ids)),{1,2})
        self.assertTrue(all(record['geometry'].is_valid for record in records))

    def test_window_capture_keeps_crown_with_its_point_query(self):
        class Stub(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight=nn.Parameter(torch.zeros(1))
            def forward(self,data):
                n=len(data['coord'])
                return dict(semantic_logits=torch.zeros(n,2),offset_m=torch.zeros(n,3),
                            point_semantic_logits=torch.zeros(n,2),
                            instance_masks=dict(mask_logits=torch.ones(1,n)*4,
                                object_logits=torch.tensor([4.]),quality_logits=torch.tensor([4.]),
                                crown_logits=torch.ones(1,2,2)*4,
                                crown_xy=torch.tensor([[0.,0.],[.5,0.],[0.,.5],[.5,.5]])))
        xyz=np.array([[0,0,3],[.5,0,3],[0,.5,3],[.5,.5,3]],np.float32)
        arrays=dict(coord=xyz,intensity=np.ones(4,np.float32),
                    grid_coord=np.floor(xyz/.25).astype(np.int32))
        raw=predict(Stub(),arrays,max_points=4,collect_crowns=True)
        self.assertEqual(len(raw['object_score']),1)
        self.assertEqual(raw['candidate_offset'][-1],4)
        self.assertEqual(raw['crown_offset'][-1],4)
        self.assertEqual(raw['candidate_quality'].shape,raw['object_score'].shape)


if __name__=='__main__':
    unittest.main()
