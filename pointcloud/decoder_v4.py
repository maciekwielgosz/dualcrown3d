"""Hybrid spatial/feature queries and annotation-aware set prediction.

Weights retain the v3 names and shapes. Decoder policy is checkpoint metadata;
changing it is an explicit architecture experiment, never an implicit upgrade.
"""
import torch
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment

from pointcloud.dual_head import DensePointDecoder, TreeMaskDecoder, discriminative_loss


class HybridTreeMaskDecoder(TreeMaskDecoder):
    teacher_probability = 0.

    def forward(self, features, data, logits, offset):
        dense = DensePointDecoder.forward(self, features, data, logits, offset)
        hidden, xyz = dense.pop('features'), data['coord']
        embedding = self.embedding(hidden)
        probability = dense['semantic_logits'].softmax(1)[:, 1]
        foreground = (probability.detach() >= .1) & (xyz[:, 2] >= .5)
        if (self.training and 'tree_id' in data and
                torch.rand((), device=xyz.device) < self.teacher_probability):
            foreground = data['tree_id'] > 0
        indices = torch.nonzero(foreground, as_tuple=False).flatten()
        if not len(indices):
            indices = torch.nonzero(xyz[:, 2] >= .5, as_tuple=False).flatten()
        if not len(indices):
            indices = torch.arange(len(xyz), device=xyz.device)
        result = {**dense, 'embedding': embedding, 'foreground_index': indices,
                  'mask_supervision': 'hungarian_v4'}
        if not len(indices):
            return {**result, 'mask_logits': hidden.new_empty((0, len(xyz))),
                    'object_logits': hidden.new_empty(0), 'seed_index': indices, 'aux_outputs': []}
        # Deduplicate spatial cells before applying the attention memory limit.
        grid = torch.floor((xyz[indices]-xyz[indices].amin(0))/.75).long()
        extent = grid.amax(0)+1
        key = grid[:, 0] + extent[0]*(grid[:, 1]+extent[1]*grid[:, 2])
        order = torch.argsort(key, stable=True)
        first = torch.ones(len(order), dtype=torch.bool, device=xyz.device)
        first[1:] = key[order[1:]] != key[order[:-1]]
        representatives = indices[order[first]]
        if len(representatives) > self.memory_tokens:
            representatives = representatives[torch.linspace(0, len(representatives)-1,
                                              self.memory_tokens, device=xyz.device).long()]
        memory_index = representatives
        spatial = self.fps(xyz[memory_index]*xyz.new_tensor([1., 1., .5]), (self.queries+1)//2)
        diverse = self.fps(embedding[memory_index].detach(), self.queries)
        seed_local = torch.cat((spatial, diverse))
        unique, inverse = torch.unique(seed_local, return_inverse=True)
        positions = torch.full((len(unique),), len(seed_local), dtype=torch.long, device=xyz.device)
        positions.scatter_reduce_(0, inverse, torch.arange(len(seed_local), device=xyz.device), reduce='amin')
        seed_index = memory_index[unique[torch.argsort(positions)[:self.queries]]]
        position = self.position(xyz/xyz.new_tensor([10., 10., 30.]))
        memory = self.attention_memory(hidden[memory_index])+position[memory_index]
        # All points can receive masks, even if the semantic branch misses them.
        mask_features = F.normalize(self.mask_memory(hidden)+position, dim=1)
        query = hidden[seed_index]+position[seed_index]
        centers = xyz[seed_index, :2]+dense['offset_m'][seed_index, :2]
        distance2 = (xyz[:, :2][None]-centers[:, None]).square().sum(2)

        def decode(q):
            normalized = self.query_norm(q)
            masks = 8.*(F.normalize(self.mask_query(normalized), dim=1) @ mask_features.T)
            radius = 1.+5.*self.radius(normalized).sigmoid()
            masks = masks+(2.5-.5*distance2/radius.square()).clamp(min=-30.)
            masks = masks+.25*torch.logit(probability.clamp(.02, .98))[None]
            return dict(mask_logits=masks, object_logits=self.score(normalized).squeeze(1))
        output=decode(query); auxiliary=[]
        for layer in self.layers:
            blocked=output['mask_logits'][:, memory_index].detach()<0
            blocked[blocked.all(1)]=False
            query=layer(query, memory, blocked); output=decode(query); auxiliary.append(output)
        return {**result, **output, 'seed_index':seed_index,
                'aux_outputs':auxiliary[:-1] if self.training else []}


def set_mask_losses(output, ids):
    """Hungarian matching on known points; unmatched unknown seeds are ignored."""
    zero=output['embedding'].sum()*0.
    valid=ids>=0
    if not len(output['seed_index']) or not valid.any():
        return dict(mask_loss=zero, embedding_loss=discriminative_loss(output['embedding'], ids), mask_iou=zero)
    gt_ids, counts=torch.unique(ids[ids>0], return_counts=True)
    gt_ids=gt_ids[counts>=4]
    gt=(gt_ids[:, None]==ids[None, valid]).float()
    logits=output['mask_logits'][:, valid].float()
    with torch.no_grad():
        probability=logits.sigmoid()
        cost=1.-(2.*(probability @ gt.T)+1.)/(probability.sum(1)[:, None]+gt.sum(1)[None]+1.)
        qi, gi=linear_sum_assignment(cost.cpu().numpy())
        qi=torch.as_tensor(qi,device=ids.device); gi=torch.as_tensor(gi,device=ids.device)
    targets=torch.zeros_like(logits)
    targets[qi]=gt[gi]
    supervised=(ids[output['seed_index']]>=0)
    supervised[qi]=True

    def layer_loss(prediction):
        logit=prediction['mask_logits'][:, valid].float(); probability=logit.sigmoid()
        if len(qi):
            p,t=probability[qi],targets[qi]
            dice=1.-(2.*(p*t).sum(1)+1.)/(p.sum(1)+t.sum(1)+1.)
            bce=F.binary_cross_entropy_with_logits(logit[qi],t,reduction='none')
            pt=p*t+(1-p)*(1-t)
            focal=((.25*t+.75*(1-t))*(1-pt).square()*bce).mean(1)
            mask=(2*dice+2*focal).mean()
        else: mask=zero
        with torch.no_grad():
            quality=torch.zeros(len(logit),device=ids.device)
            if len(qi):
                inter=(probability[qi]*targets[qi]).sum(1)
                quality[qi]=inter/(probability[qi].sum(1)+targets[qi].sum(1)-inter).clamp_min(1e-6)
        score=(prediction['object_logits'].sigmoid()-quality).square()
        return (mask+.5*score[supervised].mean() if supervised.any() else mask,
                quality[qi].mean() if len(qi) else zero)
    total,iou=layer_loss(output)
    for aux in output.get('aux_outputs',[]): total=total+.5*layer_loss(aux)[0]
    return dict(mask_loss=total,embedding_loss=discriminative_loss(output['embedding'], ids),mask_iou=iou)
