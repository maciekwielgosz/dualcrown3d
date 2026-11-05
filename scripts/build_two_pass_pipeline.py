#!/usr/bin/env python3
"""Bundle a vote-field detector and an assignment network into one pipeline checkpoint.

The bundle lives in its own run folder as ``weights/best.pt`` with the selected
configuration inside, so the held-out evaluation and scene scripts use it like
any validated two-pass checkpoint. Only bundle what validation selected.
"""
import argparse
import json
import os
from pathlib import Path
import sys

import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from scripts.cache_stage1_predictions import DEFAULT_OUTPUT, sha256, write_json


def load_assign_network(payload, device='cuda:0'):
    """The assignment network stored in a pipeline checkpoint, or None."""
    if not payload.get('assign_net'):
        return None
    from pointcloud.assign_net import AssignNet
    item = payload['assign_net']
    network = AssignNet(k=item['k'], width=item['width'], levels=item['levels']).to(device)
    network.load_state_dict(item['network'])
    return network.eval()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--run', required=True, help='name of the new pipeline run')
    parser.add_argument('--detector', required=True, help='run/weights of the detector, e.g. vote_field_v7/last.pt')
    parser.add_argument('--assign', required=True, help='run/weights of the assignment network, e.g. assign_v2/best.pt')
    args = parser.parse_args()
    root = args.output.resolve()

    def locate(text):
        run, name = text.split('/')
        return root / 'runs' / run / 'weights' / name
    detector_path, assign_path = locate(args.detector), locate(args.assign)
    detector = torch.load(detector_path, map_location='cpu', weights_only=False)
    assign = torch.load(assign_path, map_location='cpu', weights_only=False)
    if 'pool_threshold' not in assign:
        raise ValueError('The assignment checkpoint carries no validation-selected configuration (use its best.pt)')
    if assign['configuration']['detector_sha256'] != sha256(detector_path):
        raise ValueError('The assignment network was trained on candidates of a different detector')
    selection = json.loads((assign_path.parents[1] / 'selected.json').read_text())['selected']
    config = dict(assign_net=dict(pool_threshold=assign['pool_threshold'], none_bias=assign['none_bias']))
    folder = root / 'runs' / args.run / 'weights'
    if (folder / 'best.pt').exists():
        raise FileExistsError(f'{folder / "best.pt"} exists')
    folder.mkdir(parents=True, exist_ok=True)
    bundle = {**detector, 'assign_net': dict(network=assign['network'], width=assign['width'], levels=assign['levels'], k=assign['k'],
                                             epoch=assign['epoch']),
              'selected_config': config, 'selected_config_name': f"assign_{selection['config_name']}",
              'sources': dict(detector=dict(path=str(detector_path), sha256=sha256(detector_path)),
                              assign=dict(path=str(assign_path), sha256=sha256(assign_path)))}
    temporary = folder / 'best.tmp.pt'
    torch.save(bundle, temporary)
    os.replace(temporary, folder / 'best.pt')
    write_json(root / 'runs' / args.run / 'selected.json', dict(selected=selection, pipeline=bundle['sources'], config=config,
                                                                 best_sha256=sha256(folder / 'best.pt'), test_used_for_selection=False))
    print(f'pipeline -> {folder / "best.pt"} ({bundle["selected_config_name"]})')


if __name__ == '__main__':
    main()
