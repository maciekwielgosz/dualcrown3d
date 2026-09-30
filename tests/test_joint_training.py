import unittest
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn

from pointcloud.dual_head import DualHeadLitePT
from pointcloud.joint_training import sampling_weights, joint_losses, fixed_serialization


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Module()
        self.enc.enc0 = nn.Linear(4, 72)
        self.enc.enc4 = nn.Sequential(nn.Linear(72, 72), nn.BatchNorm1d(72))
        self.dec = nn.Linear(72, 72)
        self.shuffle_orders = True

    def forward(self, data):
        return SimpleNamespace(feat=self.dec(self.enc.enc4(self.enc.enc0(data['feat']))))


class Legacy(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = Backbone()
        self.semantic_head = nn.Linear(72, 2)
        self.offset_head = nn.Linear(72, 3)
        self.offset_scale_m = 10.


class JointTrainingTests(unittest.TestCase):
    def test_partial_training_updates_intended_weights_but_not_bn_buffers(self):
        torch.manual_seed(3)
        model = DualHeadLitePT(legacy=Legacy(), queries=8, decoder_layers=2, memory_tokens=32)
        model.configure_training('partial'); model.train(); fixed_serialization(model)
        before = {k:v.clone() for k,v in model.state_dict().items()}
        optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=.001)
        xyz = torch.rand(80,3)
        batch = dict(coord=xyz,feat=torch.cat((xyz,torch.ones(80,1)),1),offset=torch.tensor([80]),
                     tree_id=torch.arange(80)%4+1,instance_offset=torch.ones(80,3))
        losses = joint_losses(model(batch),batch)
        self.assertTrue(torch.isfinite(losses['loss']))
        losses['loss'].backward(); optimizer.step()
        for prefix in ('legacy.backbone.enc.enc4.', 'legacy.backbone.dec.', 'legacy.semantic_head.',
                       'legacy.offset_head.', 'point_decoder.'):
            self.assertTrue(any(not torch.equal(v,before[k]) for k,v in model.state_dict().items() if k.startswith(prefix)))
        for key,value in model.state_dict().items():
            if key.startswith('legacy.backbone.enc.enc0.') or 'running_' in key or 'num_batches_tracked' in key:
                self.assertTrue(torch.equal(value,before[key]),key)

    def test_domain_collection_and_parent_balance(self):
        rows = [dict(source_dataset='real',collection='A',dataset_id='a'),
                dict(source_dataset='real',collection='B',dataset_id='b1'),
                dict(source_dataset='real',collection='B',dataset_id='b2')]
        for parent,variants in [('p',3),('q',1)]:
            for i in range(variants):
                rows.append(dict(source_dataset='TreeScanPL10k_HELIOS',collection='H',
                                 dataset_id=f'{parent}{i}',parent_plot=parent))
        w=sampling_weights(rows,.75)
        self.assertAlmostEqual(w.sum(),1)
        self.assertAlmostEqual(w[:3].sum(),.75)
        self.assertAlmostEqual(w[0],w[1]+w[2])
        self.assertAlmostEqual(w[3:6].sum(),w[6])

    def test_ignored_labels_do_not_become_semantic_background_targets(self):
        torch.manual_seed(9)
        model=DualHeadLitePT(legacy=Legacy(),queries=8,memory_tokens=32)
        model.configure_training('heads'); model.train()
        xyz=torch.rand(64,3)
        data=dict(coord=xyz,feat=torch.cat((xyz,torch.ones(64,1)),1),offset=torch.tensor([64]),
                  tree_id=torch.cat((torch.full((32,),-1),torch.ones(32,dtype=torch.long))),
                  instance_offset=torch.zeros(64,3))
        result=model(data)
        result['semantic_logits'].retain_grad()
        loss=joint_losses(result,data)['legacy_semantic']; loss.backward()
        self.assertEqual(float(result['semantic_logits'].grad[:32].abs().sum()),0.)


if __name__ == '__main__': unittest.main()
