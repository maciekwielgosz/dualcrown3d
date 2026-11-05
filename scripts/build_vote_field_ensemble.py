#!/usr/bin/env python3
"""Combine trained vote-field detectors into one ensemble checkpoint (outputs averaged).

The ensemble lives in its own run folder so it is validated, selected and tested
with the same scripts as a single detector.
"""
import argparse
import os
from pathlib import Path
import sys

import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from scripts.cache_stage1_predictions import DEFAULT_OUTPUT, sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--run', required=True, help='name of the new ensemble run')
    parser.add_argument('--members', nargs='+', required=True, help='run/weights pairs, e.g. vote_field_v7/last.pt')
    args = parser.parse_args()
    root = args.output.resolve()
    folder = root / 'runs' / args.run / 'weights'
    if (folder / 'last.pt').exists():
        raise FileExistsError(f'{folder / "last.pt"} exists')
    folder.mkdir(parents=True, exist_ok=True)
    members, sources = [], []
    for member in args.members:
        path = root / 'runs' / member.split('/')[0] / 'weights' / member.split('/')[1]
        payload = torch.load(path, map_location='cpu', weights_only=False)
        members.append(dict(network=payload['network'], width=payload['width'], dropout=payload.get('dropout', 0.),
                            epoch=payload['epoch']))
        sources.append(dict(path=str(path), sha256=sha256(path), epoch=payload['epoch']))
        base = payload
    ensemble = dict(members=members, model='VoteFieldNet ensemble', grid=base['grid'], epoch=max(m['epoch'] for m in members),
                    stage1_checkpoint=base['stage1_checkpoint'], stage1_sha256=base['stage1_sha256'],
                    stage1_config=base['stage1_config'], symmetries=base['symmetries'], sources=sources)
    temporary = folder / 'last.tmp.pt'
    torch.save(ensemble, temporary)
    os.replace(temporary, folder / 'last.pt')
    write_json(root / 'runs' / args.run / 'configuration.json', dict(run=args.run, ensemble_of=sources,
                                                                      combination='mean of heat probability, offset and log-extent'))
    print(f'{len(members)} members -> {folder / "last.pt"}')


if __name__ == '__main__':
    main()
