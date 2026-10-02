"""Two output branches sharing one frozen LitePT-S feature extractor.

The legacy branch is unchanged. The second branch uses multiscale features,
ISA seed embeddings and a SATv2-inspired masked cross-attention decoder.
Tree IDs are per-scene instances after merging masks, not dataset-wide classes.
"""
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


def pooled_context(features, coord, size):
    grid = torch.floor((coord - coord.amin(0)) / size).long()
    _, inverse = torch.unique(grid, dim=0, return_inverse=True)
    count = torch.bincount(inverse).to(features.dtype)
    pooled = features.new_zeros((len(count), features.shape[1]))
    pooled.index_add_(0, inverse, features)
    return (pooled / count[:, None].clamp_min(1))[inverse]


class DensePointDecoder(nn.Module):
    def __init__(self, feature_dim=72, hidden_dim=128, scales=(0.75, 2.0, 6.0)):
        super().__init__()
        self.scales = tuple(scales)
        self.input = nn.Sequential(
            nn.Linear(feature_dim * (1 + len(scales)) + 8, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.GELU(),
        )
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 2 * hidden_dim),
                          nn.GELU(), nn.Linear(2 * hidden_dim, hidden_dim)) for _ in range(2)
        ])
        self.foreground = nn.Linear(hidden_dim, 2)
        self.center_residual = nn.Linear(hidden_dim, 3)
        self.quality = nn.Linear(hidden_dim, 1)
        # The new branch starts at the trained legacy predictions.
        for layer in (self.foreground, self.center_residual):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, features, data, logits, offset):
        coord = data['coord']
        context = [pooled_context(features, coord, s) for s in self.scales]
        geometry = torch.cat((data['feat'] / data['feat'].new_tensor([10., 10., 30., 1.]),
                              logits.softmax(1)[:, 1:2], offset / 10.), 1)
        hidden = self.input(torch.cat([features, *context, geometry], 1))
        for block in self.blocks:
            hidden = hidden + block(hidden)
        return dict(
            semantic_logits=logits + self.foreground(hidden),
            offset_m=offset + 5.0 * torch.tanh(self.center_residual(hidden)),
            quality_logits=self.quality(hidden).squeeze(1),
            features=hidden,
        )


class MaskAttentionLayer(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.cross = nn.MultiheadAttention(dim, 4, batch_first=True)
        self.self_attention = nn.MultiheadAttention(dim, 4, batch_first=True)
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(3)])
        self.ffn = nn.Sequential(nn.Linear(dim, 4 * dim), nn.ReLU(), nn.Linear(4 * dim, dim))

    def forward(self, query, memory, mask):
        q, m = query[None], memory[None]
        q = self.norms[0](q + self.cross(q, m, m, attn_mask=mask, need_weights=False)[0])
        q = self.norms[1](q + self.self_attention(q, q, q, need_weights=False)[0])
        return self.norms[2](q + self.ffn(q))[0]


class TreeMaskDecoder(DensePointDecoder):
    """SATv2-inspired ISA + asymmetric tree-only mask decoding, sized for 6 GB."""
    def __init__(self, hidden_dim=128, queries=96, layers=3, memory_tokens=1024):
        super().__init__(hidden_dim=hidden_dim)
        self.queries, self.memory_tokens = queries, memory_tokens
        self.epoch = 0
        self.embedding = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 5))
        self.attention_memory = nn.Linear(hidden_dim, hidden_dim)
        self.mask_memory = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim),
                                         nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.position = nn.Linear(3, hidden_dim)
        self.layers = nn.ModuleList([MaskAttentionLayer(hidden_dim) for _ in range(layers)])
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.mask_query = nn.Linear(hidden_dim, hidden_dim)
        self.score = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
        self.radius = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.radius.weight)
        nn.init.zeros_(self.radius.bias)

    @staticmethod
    @torch.no_grad()
    def fps(values, count):
        if not len(values):
            return torch.empty(0, dtype=torch.long, device=values.device)
        distance = values.new_full((len(values),), float('inf'))
        index = (values - values.mean(0)).square().sum(1).argmax()
        result = []
        for _ in range(min(count, len(values))):
            result.append(index)
            distance = torch.minimum(distance, (values - values[index]).square().sum(1))
            # Avoid repeated seeds in a collapsed embedding space.
            distance[torch.stack(result)] = -1.
            index = distance.argmax()
        return torch.stack(result)

    def forward(self, features, data, logits, offset):
        dense = super().forward(features, data, logits, offset)
        hidden, xyz = dense.pop('features'), data['coord']
        embedding = self.embedding(hidden)
        if self.training and 'tree_id' in data:
            foreground = data['tree_id'] > 0
        else:
            foreground = dense['semantic_logits'].softmax(1)[:, 1] >= .3
        indices = torch.nonzero(foreground, as_tuple=False).flatten()
        result = {**dense, 'embedding': embedding, 'foreground_index': indices}
        if not len(indices):
            return {**result, 'mask_logits': hidden.new_empty((0, len(xyz))),
                    'object_logits': hidden.new_empty(0), 'seed_index': indices, 'aux_outputs': []}
        # Bounded memory; masks are still evaluated at full voxel resolution.
        memory_local = torch.linspace(0, len(indices)-1, min(len(indices), self.memory_tokens), device=xyz.device).long()
        memory_index = indices[memory_local]
        # A short spatial-seed warm-up uses the already trained forest backbone.
        sampling = xyz[memory_index, :2] if self.training and self.epoch < 4 else embedding[memory_index].detach()
        seed_index = memory_index[self.fps(sampling, self.queries)]
        position = self.position(xyz / xyz.new_tensor([10., 10., 30.]))
        memory = self.attention_memory(hidden[memory_index]) + position[memory_index]
        mask_features = F.normalize(self.mask_memory(hidden[indices]) + position[indices], dim=1)
        query = hidden[seed_index] + position[seed_index]
        # Locality prior from the legacy votes; the learned masks refine this support.
        centers = xyz[seed_index, :2] + dense['offset_m'][seed_index, :2]
        distance2 = (xyz[indices, :2][None] - centers[:, None]).square().sum(2)

        def decode(q):
            normalized = self.query_norm(q)
            masks = 8. * (F.normalize(self.mask_query(normalized), dim=1) @ mask_features.T)
            radius = 1. + 5. * self.radius(normalized).sigmoid()
            masks = masks + (2.5 - .5 * distance2 / radius.square()).clamp(min=-30.)
            full = masks.new_full((len(q), len(xyz)), -100.)
            full[:, indices] = masks
            return {'mask_logits': full, 'object_logits': self.score(normalized).squeeze(1)}

        output = decode(query)
        auxiliary = []
        for layer in self.layers:
            blocked = output['mask_logits'][:, memory_index].detach() < 0
            blocked[blocked.all(1)] = False
            query = layer(query, memory, blocked)
            output = decode(query)
            auxiliary.append(output)
        return {**result, **output, 'seed_index': seed_index,
                'aux_outputs': auxiliary[:-1] if self.training else []}


class DualHeadLitePT(nn.Module):
    def __init__(self, patch_size=256, hidden_dim=128, legacy=None,
                 queries=96, decoder_layers=3, memory_tokens=1024, decoder_policy='legacy'):
        super().__init__()
        if legacy is None:
            from pointcloud.model import LitePTTreeInstance
            legacy = LitePTTreeInstance(patch_size=patch_size)
        self.legacy = legacy
        decoder_class = TreeMaskDecoder
        if decoder_policy == 'hybrid_v4':
            from pointcloud.decoder_v4 import HybridTreeMaskDecoder
            decoder_class = HybridTreeMaskDecoder
        elif decoder_policy == 'shared_v5':
            from pointcloud.shared_instance import SharedCrownDecoder
            decoder_class = SharedCrownDecoder
        elif decoder_policy != 'legacy':
            raise ValueError(decoder_policy)
        self.point_decoder = decoder_class(hidden_dim=hidden_dim, queries=queries,
                                             layers=decoder_layers, memory_tokens=memory_tokens)
        self.training_scope = 'mask'
        self.legacy.requires_grad_(False)
        self.legacy.eval()
        self.legacy.backbone.shuffle_orders = False

    @property
    def backbone(self):
        return self.legacy.backbone

    def train(self, mode=True):
        super().train(mode)
        # Keep BN statistics and stochastic backbone layers fixed on small crops.
        # eval() does not disable gradients through explicitly unfrozen weights.
        self.legacy.eval()
        self.legacy.backbone.shuffle_orders = False
        return self

    def configure_training(self, scope='mask'):
        if scope not in ('mask', 'heads', 'partial'):
            raise ValueError(f'Unknown training scope: {scope}')
        self.training_scope = scope
        self.legacy.requires_grad_(False)
        self.point_decoder.requires_grad_(True)
        if scope != 'mask':
            self.legacy.semantic_head.requires_grad_(True)
            self.legacy.offset_head.requires_grad_(True)
        if scope == 'partial':
            # Deepest attention stage plus feature upsampling; early encoder fixed.
            self.backbone.enc.enc4.requires_grad_(True)
            self.backbone.dec.requires_grad_(True)
        self.train(self.training)

    def forward(self, data):
        if len(data['offset']) != 1:
            raise ValueError('One spatial crop per forward is required')
        joint = self.training_scope != 'mask'
        with torch.set_grad_enabled(torch.is_grad_enabled() and joint):
            features = self.legacy.backbone(data).feat
            logits = self.legacy.semantic_head(features)
            offset = self.legacy.offset_head(features) * self.legacy.offset_scale_m
        dense = self.point_decoder(features, data, logits, offset)
        return dict(semantic_logits=logits, offset_m=offset,
                    point_semantic_logits=dense['semantic_logits'],
                    point_offset_m=dense['offset_m'], point_quality_logits=dense['quality_logits'],
                    instance_masks=dense)

    def initialize_legacy(self, path):
        payload = torch.load(Path(path), map_location='cpu', weights_only=False)
        self.legacy.load_state_dict(payload['model'], strict=True)


class PointBranchView(nn.Module):
    """Adapter for the existing, unchanged full-crown evaluation protocol."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    @property
    def backbone(self):
        return self.model.backbone

    def forward(self, data):
        result = self.model(data)
        return dict(semantic_logits=result['point_semantic_logits'], offset_m=result['point_offset_m'])


class MaskBranchView(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.auxiliary_losses = False

    @property
    def backbone(self):
        return self.model.backbone

    def forward(self, data):
        return self.model(data)['instance_masks']


def discriminative_loss(embedding, ids):
    positive = ids > 0
    if not positive.any():
        return embedding.sum() * 0.
    _, inverse, counts = torch.unique(ids[positive], return_inverse=True, return_counts=True)
    values = embedding[positive]
    means = values.new_zeros((len(counts), values.shape[1])).index_add_(0, inverse, values) / counts[:, None]
    valid = counts >= 2
    if not valid.any():
        return embedding.sum() * 0.
    distance = torch.linalg.vector_norm(values - means[inverse], dim=1)
    variance = values.new_zeros(len(counts)).index_add_(0, inverse, F.relu(distance - .5).square()) / counts
    means = means[valid]
    push = F.relu(5. - torch.pdist(means)).square().mean() if len(means) > 1 else means.sum() * 0.
    return variance[valid].mean() + push + .001 * means.norm(dim=1).mean()


def seed_mask_losses(output, ids):
    if output.get('mask_supervision') == 'hungarian_v4':
        from pointcloud.decoder_v4 import set_mask_losses
        return set_mask_losses(output, ids)
    zero = output['embedding'].sum() * 0.
    if not len(output['seed_index']):
        return {'mask_loss': zero, 'embedding_loss': discriminative_loss(output['embedding'], ids), 'mask_iou': zero}
    seed_ids = ids[output['seed_index']]
    valid_queries = seed_ids > 0
    valid_points = ids >= 0
    target = (seed_ids[:, None] == ids[None]).float()[:, valid_points]
    target[~valid_queries] = 0.
    gt_ids = torch.unique(ids[ids > 0])
    gt = (gt_ids[:, None] == ids[None]).float()[:, valid_points]

    def layer_loss(prediction):
        logits = prediction['mask_logits'][:, valid_points].float()
        prob = logits.sigmoid()
        dice = 1. - (2 * (prob * target).sum(1) + 1.) / (prob.sum(1) + target.sum(1) + 1.)
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
        pt = prob * target + (1-prob) * (1-target)
        focal = ((.25 * target + .75 * (1-target)) * (1-pt).square() * bce).mean(1)
        with torch.no_grad():
            intersection = prob @ gt.T
            iou = intersection / (prob.sum(1)[:, None] + gt.sum(1)[None] - intersection).clamp_min(1e-6)
            quality = iou.amax(1) if len(gt) else torch.zeros(len(prob), device=prob.device)
        score = (prediction['object_logits'].sigmoid() - quality).square()
        if not valid_queries.any():
            return zero, quality.mean()
        return (2 * dice + 2 * focal + .5 * score)[valid_queries].mean(), quality[valid_queries].mean()

    total, iou = layer_loss(output)
    for auxiliary in output.get('aux_outputs', []):
        total = total + .5 * layer_loss(auxiliary)[0]
    return {'mask_loss': total, 'embedding_loss': discriminative_loss(output['embedding'], ids), 'mask_iou': iou}


def dense_losses(prediction, batch):
    logits = prediction['point_semantic_logits']
    offset = prediction['point_offset_m']
    ids = batch['tree_id']
    valid = ids >= 0
    positive = ids > 0
    semantic_target = batch.get('semantic_target', torch.where(ids < 0, -1, positive.long()))
    semantic_valid = semantic_target >= 0
    zero = logits.sum() * 0.
    semantic = F.cross_entropy(logits[semantic_valid], semantic_target[semantic_valid].long(),
                               weight=logits.new_tensor([1., 1.2])) if semantic_valid.any() else zero
    probability = logits.softmax(1)[:, 1]
    semantic_positive = semantic_target > 0
    dice = 1. - (2 * probability[semantic_positive].sum() + 1.) / (
        probability[semantic_valid].sum() + semantic_positive.sum() + 1.)
    regression = compact = separation = quality = zero
    if positive.any():
        unique, inverse, counts = torch.unique(ids[positive], return_inverse=True, return_counts=True)
        target = batch['instance_offset'][positive, :2]
        votes = batch['coord'][positive, :2] + offset[positive, :2]
        target_votes = batch['coord'][positive, :2] + target
        # Give small and large trees equal aggregate weight.
        weight = 1. / counts[inverse].float()
        error = F.smooth_l1_loss(offset[positive, :2], target, beta=.5, reduction='none').mean(1)
        regression = (weight * error).sum() / len(unique)
        means = votes.new_zeros((len(unique), 2)).index_add_(0, inverse, votes) / counts[:, None]
        true_means = votes.new_zeros((len(unique), 2)).index_add_(0, inverse, target_votes) / counts[:, None]
        spread = (votes - means[inverse]).square().sum(1).clamp(max=25.)
        compact = (weight * spread).sum() / len(unique)
        if len(unique) > 1:
            pred_dist = torch.cdist(means, means)
            true_dist = torch.cdist(true_means, true_means)
            neighbors = (true_dist > .1) & (true_dist < 8.)
            if neighbors.any():
                margin = (.5 * true_dist).clamp(max=2.)
                separation = F.relu(margin[neighbors] - pred_dist[neighbors]).square().mean()
        quality_target = torch.zeros_like(probability)
        quality_target[positive] = torch.exp(-torch.linalg.vector_norm(offset[positive, :2].detach() - target, dim=1) / 2.)
        quality = F.binary_cross_entropy_with_logits(prediction['point_quality_logits'][valid], quality_target[valid])
    loss = semantic + .25 * dice + regression + .05 * compact + .1 * separation + .1 * quality
    masks = seed_mask_losses(prediction['instance_masks'], ids)
    loss = loss + masks['mask_loss'] + .1 * masks['embedding_loss']
    return dict(loss=loss, semantic=semantic, dice=dice, offset=regression,
                compact=compact, separation=separation, quality=quality, **masks)
