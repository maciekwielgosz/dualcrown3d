"""Whole-plot proposal verification and complete-mask instance assignment."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn


class ProposalVerifier(nn.Module):
    def __init__(self, features=14):
        super().__init__()
        self.network=nn.Sequential(nn.Linear(features,64),nn.GELU(),nn.Dropout(.15),
                                   nn.Linear(64,32),nn.GELU(),nn.Dropout(.15),
                                   nn.Linear(32,1))

    def forward(self,x):
        return self.network(x).squeeze(-1)


def assign_complete_masks(arrays, graph, probability, *, object_threshold=.5,
                          claimed_limit=.3, min_points=8, min_height_m=1.5):
    """Select graph-fused complete masks before assigning any point-level IDs.

    A heavily claimed candidate is dropped in its entirety.  It is never
    converted into a new crown made solely of its unclaimed residual.
    """
    p=np.asarray(probability,np.float32)
    if len(p)!=len(graph['feature']):
        raise ValueError('Verifier probability count does not match graph')
    n=len(arrays['coord'])
    labels=np.zeros(n,np.uint32)
    confidence=np.zeros(n,np.float32)
    rank=p * graph['feature'][:,4].clip(.01,1.)
    accepted=[]
    for j in np.argsort(-rank,kind='stable'):
        if p[j]<object_threshold:
            continue
        a,b=graph['offset'][j:j+2]
        ids=graph['point_index'][a:b]
        scores=graph['point_score'][a:b]
        if len(ids)<min_points or arrays['coord'][ids,2].max()<min_height_m:
            continue
        free=labels[ids]==0
        if 1.-free.mean()>claimed_limit or free.sum()<min_points:
            continue
        identifier=len(accepted)+1
        labels[ids[free]]=identifier
        confidence[ids[free]]=scores[free]*p[j]
        accepted.append(int(j))
    return labels,confidence,accepted
