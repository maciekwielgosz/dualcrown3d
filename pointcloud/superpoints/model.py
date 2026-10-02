"""Complete mask decoder on a frozen/cached LitePT feature extractor."""
import torch
from torch import nn
from pointcloud.dual_head import TreeMaskDecoder
from pointcloud.superpoints.decoder import SuperpointMaskDecoder


class CachedInstanceModel(nn.Module):
    def __init__(self, method='fixed', queries=96, memory_tokens=1024,
                 graph_width=96, graph_layers=3, wide_dim=0, wide_layers=2):
        super().__init__()
        self.method = method
        self.semantic_head = nn.Sequential(nn.Linear(72, 72), nn.GELU(), nn.Dropout(.1), nn.Linear(72, 2))
        self.offset_head = nn.Sequential(nn.Linear(72, 72), nn.GELU(), nn.Linear(72, 3))
        if method == 'retained':
            self.decoder = TreeMaskDecoder(queries=queries, memory_tokens=memory_tokens)
        else:
            self.decoder = SuperpointMaskDecoder(use_graph=method != 'control',
                                                 queries=queries, memory_tokens=memory_tokens,
                                                 graph_width=graph_width, graph_layers=graph_layers,
                                                 wide_dim=wide_dim, wide_layers=wide_layers)
        self.semantic_head.requires_grad_(False)
        self.offset_head.requires_grad_(False)

    def initialize(self, state):
        for name in ('semantic_head', 'offset_head'):
            prefix = f'legacy.{name}.'
            getattr(self, name).load_state_dict({k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)})
        prefix = 'point_decoder.'
        missing, extra = self.decoder.load_state_dict(
            {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}, strict=False)
        if extra or any(not key.startswith(('graph.', 'wide.')) for key in missing):
            raise ValueError(f'Unexpected decoder initialization mismatch: {missing}, {extra}')

    def forward(self, data):
        self.semantic_head.eval()
        feature = data['feature']
        with torch.no_grad():
            semantic = self.semantic_head(feature)
            offset = self.offset_head(feature) * 10.
        masks = self.decoder(feature, data, semantic, offset)
        return dict(instance_masks=masks, semantic_logits=semantic, offset_m=offset,
                    point_semantic_logits=masks['semantic_logits'], point_offset_m=masks['offset_m'],
                    point_quality_logits=masks['quality_logits'])
