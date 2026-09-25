"""LitePT backbone with a compact query-based 3-D instance mask decoder."""

from __future__ import annotations

import math
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn


PROJECT_DIR = Path(__file__).resolve().parents[1]
LITEPT_ROOT = PROJECT_DIR / "vendor" / "LitePT"
if str(LITEPT_ROOT) not in sys.path:
    sys.path.insert(0, str(LITEPT_ROOT))

from litept.model import LitePT  # noqa: E402


class LitePTMaskDecoder(nn.Module):
    """Predict a fixed set of scored instance masks from LitePT point features."""

    def __init__(
        self,
        patch_size: int = 256,
        queries: int = 64,
        hidden_dim: int = 128,
        decoder_layers: int = 3,
        attention_heads: int = 4,
        memory_tokens: int = 2048,
        spatial_prior: bool = False,
        masked_attention: bool = False,
        auxiliary_losses: bool = False,
        semantic_head: bool = False,
        backbone_type: str = "litept",
        anchor_mode: str = "fps",
    ):
        super().__init__()
        self.queries = int(queries)
        self.memory_tokens = int(memory_tokens)
        self.spatial_prior = bool(spatial_prior)
        self.masked_attention = masked_attention
        self.auxiliary_losses = auxiliary_losses
        self.backbone_type = backbone_type
        self.anchor_mode = anchor_mode
        self.backbone = LitePT(
            in_channels=4,
            enc_patch_size=(patch_size,) * 5,
            dec_patch_size=(patch_size,) * 4,
        ) if backbone_type == "litept" else nn.Sequential(
            nn.Linear(4, 64), nn.LayerNorm(64), nn.GELU(),
            nn.Linear(64, 128), nn.LayerNorm(128), nn.GELU(), nn.Linear(128, 72)
        )
        self.foreground_head = nn.Linear(72, 1) if semantic_head else None
        self.memory_projection = nn.Linear(72, hidden_dim)
        self.position_projection = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.query_embedding = nn.Embedding(queries, hidden_dim)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=attention_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=decoder_layers,
            norm=nn.LayerNorm(hidden_dim),
        )
        self.object_head = nn.Linear(hidden_dim, 1)
        self.mask_query = nn.Linear(hidden_dim, hidden_dim)
        self.mask_point = nn.Sequential(
            nn.Linear(72, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.logit_scale = nn.Parameter(torch.tensor(math.log(8.0)))
        if self.spatial_prior:
            self.center_offset_head = nn.Linear(hidden_dim, 2)
            self.radius_head = nn.Linear(hidden_dim, 1)
            nn.init.zeros_(self.center_offset_head.weight)
            nn.init.zeros_(self.center_offset_head.bias)
            nn.init.zeros_(self.radius_head.weight)
            nn.init.constant_(self.radius_head.bias, -0.95)
        nn.init.constant_(self.object_head.bias, -2.0)

    def _memory_indices(self, count: int, device: torch.device) -> torch.Tensor:
        if count <= self.memory_tokens:
            return torch.arange(count, device=device)
        # Even sampling is deterministic at evaluation and covers the complete
        # serialized point sequence instead of favoring one spatial corner.
        return torch.linspace(0, count - 1, self.memory_tokens, device=device).long()

    def _farthest_anchors(self, coord: torch.Tensor, count: int) -> torch.Tensor:
        """Deterministic farthest-point anchors on a small memory token set."""
        count = min(count, len(coord))
        if count == 0:
            return torch.zeros(0, dtype=torch.long, device=coord.device)
        center = coord.mean(dim=0, keepdim=True)
        first = torch.linalg.vector_norm(coord - center, dim=1).argmax()
        anchors = [first]
        distance = torch.linalg.vector_norm(coord - coord[first], dim=1).square()
        for _ in range(1, count):
            next_index = distance.argmax()
            anchors.append(next_index)
            next_distance = torch.linalg.vector_norm(
                coord - coord[next_index], dim=1
            ).square()
            distance = torch.minimum(distance, next_distance)
        return torch.stack(anchors)

    def forward(self, data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        point_features = self.backbone(data).feat if self.backbone_type == "litept" else self.backbone(data["feat"])
        count = len(point_features)
        indices = self._memory_indices(count, point_features.device)
        all_coord = data["coord"]
        center = all_coord.mean(dim=0, keepdim=True)
        scale = (all_coord - center).abs().amax(dim=0, keepdim=True).clamp_min(1.0)
        normalized_coord = (all_coord - center) / scale
        coord = normalized_coord[indices]
        position = self.position_projection(coord)
        memory = self.memory_projection(point_features[indices]) + position
        anchor_index = self._farthest_anchors(coord[:, :2], self.queries)
        anchors = memory[anchor_index]
        anchor_coord = coord[anchor_index, :2]
        proposal_bias = None
        if self.anchor_mode == "height_peaks":
            peak_indices = self._height_peak_indices(all_coord)
            peak_count = len(peak_indices)
            # Real peak features followed by fallback FPS seeds. A fixed prior
            # distinguishes proposals from padding, but does not forbid trees
            # without a visible height maximum from being learned.
            fallback = indices[anchor_index]
            selected = torch.cat([peak_indices, fallback])[:self.queries]
            anchors = self.memory_projection(point_features[selected]) + self.position_projection(normalized_coord[selected])
            anchor_coord = normalized_coord[selected, :2]
            proposal_bias = torch.full((self.queries,), -2., device=coord.device)
            proposal_bias[:peak_count] = 1.
        if len(anchors) < self.queries:
            padding = self.query_embedding.weight[len(anchors) :]
            anchors = torch.cat((anchors, padding), dim=0)
            anchor_coord = torch.cat(
                (
                    anchor_coord,
                    torch.zeros(
                        self.queries - len(anchor_coord),
                        2,
                        device=coord.device,
                        dtype=coord.dtype,
                    ),
                ),
                dim=0,
            )
        queries = (self.query_embedding.weight + anchors).unsqueeze(0)
        if self.masked_attention:
            point_masks = self.mask_point(point_features) + self.position_projection(normalized_coord)
            point_masks = nn.functional.normalize(point_masks, dim=1)
            auxiliary = []
            decoded = queries[0]
            for layer in self.decoder.layers:
                prior = self._decode_masks(decoded, point_masks, normalized_coord, anchor_coord, proposal_bias)
                attention_mask = prior["mask_logits"][:, indices].detach() < 0
                # Keep a minimum local support even for initially empty masks.
                nearest = prior["mask_logits"][:, indices].detach().topk(min(32, len(indices)), dim=1).indices
                attention_mask.scatter_(1, nearest, False)
                decoded = layer(decoded.unsqueeze(0), memory.unsqueeze(0), memory_mask=attention_mask)[0]
                decoded_norm = self.decoder.norm(decoded)
                auxiliary.append(self._decode_masks(decoded_norm, point_masks, normalized_coord, anchor_coord, proposal_bias))
            result = auxiliary[-1]
            if self.auxiliary_losses:
                result["aux_outputs"] = auxiliary[:-1]
            if self.foreground_head is not None:
                result["semantic_logits"] = self.foreground_head(point_features).squeeze(1)
            return result
        decoded = self.decoder(queries, memory.unsqueeze(0))[0]
        query_masks = nn.functional.normalize(self.mask_query(decoded), dim=1)
        point_masks = self.mask_point(point_features) + self.position_projection(
            normalized_coord
        )
        point_masks = nn.functional.normalize(point_masks, dim=1)
        result = self._decode_masks(decoded, point_masks, normalized_coord, anchor_coord, proposal_bias)
        if self.foreground_head is not None:
            result["semantic_logits"] = self.foreground_head(point_features).squeeze(1)
        return result

    @torch.no_grad()
    def _height_peak_indices(self, coord):
        grid = torch.floor((coord[:, :2] - coord[:, :2].amin(0)) / .5).long()
        shape = grid.amax(0) + 1
        h, w = int(shape[0]), int(shape[1])
        flat = grid[:, 0] * w + grid[:, 1]
        heights = torch.full((h*w,), -float("inf"), device=coord.device)
        heights.scatter_reduce_(0, flat, coord[:, 2].float(), reduce="amax")
        local_max = nn.functional.max_pool2d(heights.reshape(1, 1, h, w), 5, stride=1, padding=2).flatten()
        candidates = torch.nonzero((coord[:, 2] == heights[flat]) & (coord[:, 2] >= local_max[flat]) & (coord[:, 2] >= 2.), as_tuple=False).flatten()
        if not len(candidates):
            return candidates
        # One source point per peak cell, including quantized-height ties.
        order = torch.argsort(flat[candidates], stable=True)
        candidates = candidates[order]
        first = torch.cat([torch.ones(1, dtype=torch.bool, device=coord.device), flat[candidates][1:] != flat[candidates][:-1]])
        candidates = candidates[first]
        return candidates[torch.argsort(coord[candidates, 2], descending=True)[:self.queries]]

    def _decode_masks(self, decoded, point_masks, normalized_coord, anchor_coord, proposal_bias=None):
        query_masks = nn.functional.normalize(self.mask_query(decoded), dim=1)
        scale_value = self.logit_scale.exp().clamp(max=30.0)
        mask_logits = scale_value * torch.einsum(
            "qd,nd->qn", query_masks, point_masks
        )
        result = {
            "object_logits": self.object_head(decoded).squeeze(1) + (proposal_bias if proposal_bias is not None else 0.),
            "mask_logits": mask_logits,
            "normalized_coord": normalized_coord,
        }
        if self.spatial_prior:
            # FPS gives every query a deterministic local anchor.  The decoder
            # may move it by up to roughly 3.5 m in a 20 m crop, while the
            # learned radius covers the observed crown-radius distribution.
            query_center = anchor_coord + 0.35 * torch.tanh(
                self.center_offset_head(decoded)
            )
            query_radius = 0.03 + 0.32 * torch.sigmoid(
                self.radius_head(decoded).squeeze(1)
            )
            delta = normalized_coord[None, :, :2] - query_center[:, None, :]
            distance_square = delta.square().sum(dim=2)
            spatial_logits = 2.5 - 0.5 * distance_square / query_radius[:, None].square()
            result.update(
                {
                    "mask_logits": mask_logits + spatial_logits.clamp(min=-20.0),
                    "query_center": query_center,
                    "query_radius": query_radius,
                }
            )
        return result

    def load_backbone_from_tree_checkpoint(self, checkpoint: Path) -> dict[str, int]:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = payload.get("model", payload.get("state_dict", payload))
        backbone = OrderedDict()
        prefix = "backbone."
        for key, value in state.items():
            key = key.removeprefix("module.")
            if key.startswith(prefix):
                backbone[key[len(prefix) :]] = value
        incompatible = self.backbone.load_state_dict(backbone, strict=False)
        return {
            "loaded": len(backbone) - len(incompatible.unexpected_keys),
            "missing": len(incompatible.missing_keys),
            "unexpected": len(incompatible.unexpected_keys),
        }


def _instance_targets(tree_id: torch.Tensor, minimum_points: int = 4) -> torch.Tensor:
    identifiers, counts = torch.unique(tree_id[tree_id > 0], return_counts=True)
    identifiers = identifiers[counts >= minimum_points]
    if len(identifiers) == 0:
        return torch.zeros((0, len(tree_id)), dtype=torch.float32, device=tree_id.device)
    return (tree_id.unsqueeze(0) == identifiers.unsqueeze(1)).float()


def _matching_cost(
    object_logits: torch.Tensor,
    mask_logits: torch.Tensor,
    targets: torch.Tensor,
    query_center: torch.Tensor | None = None,
    target_center: torch.Tensor | None = None,
    max_points: int = 4096,
) -> np.ndarray:
    count = mask_logits.shape[1]
    if count > max_points:
        indices = torch.linspace(0, count - 1, max_points, device=mask_logits.device).long()
        logits = mask_logits[:, indices]
        target = targets[:, indices]
    else:
        logits = mask_logits
        target = targets
    points = max(logits.shape[1], 1)
    # BCE(logit, target) = softplus(logit) - logit * target.
    bce = nn.functional.softplus(logits).mean(dim=1, keepdim=True)
    bce = bce - logits @ target.T / points
    probability = logits.sigmoid()
    intersection = probability @ target.T
    denominator = probability.sum(dim=1, keepdim=True) + target.sum(dim=1).unsqueeze(0)
    dice = 1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)
    object_cost = -object_logits.sigmoid().unsqueeze(1)
    cost = 0.5 * object_cost + 2.0 * bce + 2.0 * dice
    if query_center is not None and target_center is not None:
        cost = cost + 4.0 * torch.cdist(query_center, target_center)
    return cost.detach().float().cpu().numpy()


def mask_decoder_losses(
    prediction: dict[str, torch.Tensor], tree_id: torch.Tensor,
    object_positive_weight: bool = True,
) -> dict[str, torch.Tensor | int]:
    # Keep matching and loss accumulation in FP32 even when the backbone and
    # decoder run under AMP.  Rare very sparse crops can otherwise overflow
    # the pairwise mask-cost calculation in FP16.
    object_logits = prediction["object_logits"].float()
    mask_logits = prediction["mask_logits"].float()
    targets = _instance_targets(tree_id)
    object_target = torch.zeros_like(object_logits)
    center_loss = mask_logits.sum() * 0.0
    radius_loss = mask_logits.sum() * 0.0

    if len(targets):
        target_center = None
        target_radius = None
        if "query_center" in prediction:
            xy = prediction["normalized_coord"][:, :2].float()
            target_count = targets.sum(dim=1, keepdim=True).clamp_min(1.0)
            target_center = targets @ xy / target_count
            centered = xy.unsqueeze(0) - target_center.unsqueeze(1)
            mean_distance_square = (
                targets * centered.square().sum(dim=2)
            ).sum(dim=1) / target_count.squeeze(1)
            target_radius = (2.0 * mean_distance_square).sqrt().clamp(0.03, 0.35)
        cost = _matching_cost(
            object_logits,
            mask_logits,
            targets,
            (
                prediction["query_center"].float()
                if "query_center" in prediction
                else None
            ),
            target_center,
        )
        query_index_np, target_index_np = linear_sum_assignment(cost)
        query_index = torch.as_tensor(query_index_np, device=tree_id.device, dtype=torch.long)
        target_index = torch.as_tensor(target_index_np, device=tree_id.device, dtype=torch.long)
        object_target[query_index] = 1.0
        predicted_masks = mask_logits[query_index]
        target_masks = targets[target_index]
        positive = target_masks.sum(dim=1).clamp_min(1.0)
        negative = target_masks.shape[1] - positive
        positive_weight = (negative / positive).clamp(1.0, 12.0).unsqueeze(1)
        mask_bce = (
            nn.functional.softplus(predicted_masks) * (1.0 - target_masks)
            + nn.functional.softplus(-predicted_masks)
            * target_masks
            * positive_weight
        ).mean()
        probability = predicted_masks.sigmoid()
        intersection = (probability * target_masks).sum(dim=1)
        mask_dice = (
            1.0
            - (2.0 * intersection + 1.0)
            / (probability.sum(dim=1) + target_masks.sum(dim=1) + 1.0)
        ).mean()
        matched_iou = (
            ((probability >= 0.5) & (target_masks > 0.5)).sum(dim=1).float()
            / ((probability >= 0.5) | (target_masks > 0.5)).sum(dim=1).clamp_min(1).float()
        ).mean()
        if target_center is not None and target_radius is not None:
            center_loss = nn.functional.smooth_l1_loss(
                prediction["query_center"][query_index].float(),
                target_center[target_index],
            )
            radius_loss = nn.functional.smooth_l1_loss(
                prediction["query_radius"][query_index].float(),
                target_radius[target_index],
            )
    else:
        mask_bce = mask_logits.sum() * 0.0
        mask_dice = mask_logits.sum() * 0.0
        matched_iou = mask_logits.sum() * 0.0

    positives = max(int(object_target.sum().item()), 1)
    positive_weight = torch.tensor(
        min(float(len(object_target)) / positives, 8.0) if object_positive_weight else 1.,
        device=object_logits.device,
    )
    object_loss = nn.functional.binary_cross_entropy_with_logits(
        object_logits, object_target, pos_weight=positive_weight
    )
    loss = (
        object_loss
        + 2.0 * mask_bce
        + 2.0 * mask_dice
        + 3.0 * center_loss
        + 0.5 * radius_loss
    )
    if "semantic_logits" in prediction:
        loss = loss + nn.functional.binary_cross_entropy_with_logits(
            prediction["semantic_logits"].float(), (tree_id > 0).float()
        )
    if prediction.get("aux_outputs"):
        loss = loss + .3 * torch.stack([
            mask_decoder_losses(item, tree_id, object_positive_weight)["loss"]
            for item in prediction["aux_outputs"]
        ]).mean()
    return {
        "loss": loss,
        "object_loss": object_loss,
        "mask_bce_loss": mask_bce,
        "mask_dice_loss": mask_dice,
        "center_loss": center_loss,
        "radius_loss": radius_loss,
        "matched_mask_iou": matched_iou,
        "target_instances": int(len(targets)),
    }
