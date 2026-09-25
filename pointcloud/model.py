"""LitePT-S backbone with tree semantics and instance-center offset heads."""

from __future__ import annotations

import sys
from collections import OrderedDict
from pathlib import Path

import torch
from torch import nn


PROJECT_DIR = Path(__file__).resolve().parents[1]
LITEPT_ROOT = PROJECT_DIR / "vendor" / "LitePT"
if str(LITEPT_ROOT) not in sys.path:
    sys.path.insert(0, str(LITEPT_ROOT))

from litept.model import LitePT  # noqa: E402


class LitePTTreeInstance(nn.Module):
    """Predict tree probability and a 3-D offset to the tree centroid."""

    def __init__(self, patch_size: int = 256, offset_scale_m: float = 10.0):
        super().__init__()
        self.offset_scale_m = float(offset_scale_m)
        self.backbone = LitePT(
            in_channels=4,
            enc_patch_size=(patch_size,) * 5,
            dec_patch_size=(patch_size,) * 4,
        )
        self.semantic_head = nn.Sequential(
            nn.Linear(72, 72),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(72, 2),
        )
        self.offset_head = nn.Sequential(
            nn.Linear(72, 72),
            nn.GELU(),
            nn.Linear(72, 3),
        )
        nn.init.zeros_(self.offset_head[-1].weight)
        nn.init.zeros_(self.offset_head[-1].bias)

    def forward(self, data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        point = self.backbone(data)
        features = point.feat
        return {
            "semantic_logits": self.semantic_head(features),
            "offset_m": self.offset_head(features) * self.offset_scale_m,
        }

    def load_pretrained_backbone(self, checkpoint: Path) -> dict[str, int]:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = payload.get("state_dict", payload)
        backbone = OrderedDict()
        prefix = "module.backbone."
        for key, value in state.items():
            if key.startswith(prefix):
                backbone[key[len(prefix) :]] = value
        incompatible = self.backbone.load_state_dict(backbone, strict=False)
        return {
            "loaded": len(backbone) - len(incompatible.unexpected_keys),
            "missing": len(incompatible.missing_keys),
            "unexpected": len(incompatible.unexpected_keys),
        }


def instance_losses(
    prediction: dict[str, torch.Tensor],
    semantic_target: torch.Tensor,
    offset_target_m: torch.Tensor,
    class_weights: torch.Tensor | None = None,
    xy_only: bool = False,
) -> dict[str, torch.Tensor]:
    logits = prediction["semantic_logits"]
    offset = prediction["offset_m"]
    semantic_loss = nn.functional.cross_entropy(logits, semantic_target, weight=class_weights)
    mask = semantic_target == 1
    if torch.any(mask):
        predicted = offset[mask]
        target = offset_target_m[mask]
        if xy_only:
            predicted, target = predicted[:, :2], target[:, :2]
        offset_l1 = nn.functional.smooth_l1_loss(predicted, target, beta=0.5)
        cosine = 1.0 - nn.functional.cosine_similarity(predicted, target, dim=1, eps=1e-6)
        valid_direction = torch.linalg.vector_norm(target, dim=1) > 0.1
        offset_cosine = (
            cosine[valid_direction].mean()
            if torch.any(valid_direction)
            else torch.zeros((), device=logits.device)
        )
    else:
        offset_l1 = torch.zeros((), device=logits.device)
        offset_cosine = torch.zeros((), device=logits.device)
    loss = semantic_loss + offset_l1 + 0.5 * offset_cosine
    return {
        "loss": loss,
        "semantic_loss": semantic_loss,
        "offset_l1_loss": offset_l1,
        "offset_cosine_loss": offset_cosine,
    }
