#!/usr/bin/env python3
"""Compare core ownership versus full overlap-mask merging on validation only."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from pointcloud.dual_head import DualHeadLitePT
from scripts.evaluate_dual_head import evaluate_masks


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, default=PROJECT / 'outputs/dual_head_satv2_litept_v3')
    args = p.parse_args()
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('high')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    selected_path = args.run / 'selected.json'
    original = json.loads(selected_path.read_text())
    archive = args.run / 'selection_before_merge_calibration.json'
    if archive.exists():
        raise FileExistsError('Merge calibration already started')
    archive.write_text(json.dumps(original, indent=2) + '\n')
    checkpoint = torch.load(args.run / 'weights/best.pt', map_location='cpu', weights_only=False)
    model = DualHeadLitePT().to('cuda:0')
    model.load_state_dict(checkpoint['model'], strict=True)
    configurations = [dict(object_threshold=s, mask_threshold=m, minimum_voxels=12,
                           merge_overlap=overlap, owner_only=owner)
                      for s in (.1, .3) for m in (.4, .5) for overlap in (.15, .5) for owner in (True, False)]
    result = evaluate_masks(model, Path(checkpoint['config']['manifest']), args.run / 'merge_calibration', configs=configurations)
    original['config'] = result['config']
    original['validation'] = result
    original['merge_calibration'] = 'validation-only core ownership/full-mask merge grid; fixed trained checkpoint'
    selected_path.write_text(json.dumps(original, indent=2) + '\n')
    print(json.dumps({'config': result['config'], 'point_metrics': result['point_metrics'], 'crown_metrics': result['metrics']}, indent=2))


if __name__ == '__main__':
    main()
