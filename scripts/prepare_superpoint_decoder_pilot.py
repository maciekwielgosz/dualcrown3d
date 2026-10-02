#!/usr/bin/env python3
"""Build matched Stage-3 crops; run with isolated Stage-2 packages on PYTHONPATH."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import time
import geopandas as gpd
import numpy as np
import shapely
from shapely.geometry import box
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from pointcloud.data import load_npz, prepare_crop
from pointcloud.superpoints.decoder import group_geometry
from pointcloud.superpoints.instance_embedding import InstanceBoundaryEmbedding, edge_affinity
from pointcloud.superpoints.partition import geometric_partition
from scripts.benchmark_superpoint_algorithms import load_methods, PartitionContext, budget_match, sha256, save_json
from scripts.train_superpoint_affinity import eligibility
from scripts.train_supervision_v4 import INITIAL, REAL

ROOT = PROJECT / 'outputs/dualcrown3d_superpoint_v1'
DEFAULT = ROOT / 'stage3_decoder_pilot'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=DEFAULT)
    p.add_argument('--max-plots', type=int, default=0)
    args = p.parse_args()
    root = args.output.resolve()
    if root.exists():
        raise FileExistsError(f'Protected experiment directory: {root}')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    torch.set_num_threads(4)
    torch.manual_seed(20261001)
    _, csr, ezsp, shim = load_methods()
    embedding_path = ROOT / 'stage2_instance_embedding/embedding.pt'
    embedding = InstanceBoundaryEmbedding().cuda().eval()
    embedding.load_state_dict(torch.load(embedding_path, map_location='cpu', weights_only=False)['model'])
    old_config = json.loads((ROOT / 'stage2_affinity_pilot/config.json').read_text())
    train, val = eligibility()
    root.mkdir(parents=True)
    entries = []
    for split, rows in [('train', train), ('val', val)]:
        for index, row in enumerate(rows):
            if args.max_plots and index >= args.max_plots:
                break
            start = time.monotonic()
            cached = ROOT / 'stage2_affinity_pilot/cache' / split / f"{row['dataset_id']}.npz"
            with np.load(cached) as f:
                old = {k: f[k] for k in f.files}
            arrays = load_npz(row['output'])
            rng = np.random.default_rng(old_config['seed'] + index + (0 if split == 'train' else 100000))
            anchor = int(rng.integers(len(arrays['coord'])))
            crop = prepare_crop(arrays, rng, 20., old_config['max_points'], False,
                                preserve_height=bool(row.get('height_normalization')),
                                anchor_index=anchor, return_transform=True)
            coord = crop['coord'].numpy()
            if not np.array_equal(coord, old['coord']) or not np.array_equal(crop['tree_id'].numpy(), old['tree_id']):
                raise ValueError(f"Stage-2 crop parity failed: {row['dataset_id']}")
            world = coord[:, :2].astype(np.float64) + crop['local_to_world'].numpy()[:2, 2]
            center = arrays['coord'][anchor, :2].astype(np.float64) + arrays['source_origin'][:2]
            bounds = box(*(center - 10.), *(center + 10.))
            frame = gpd.read_file(row['gt_vector'])
            for key in ('complete', 'evaluation_eligible'):
                if key in frame:
                    frame = frame[frame[key].astype(bool)]
            frame = frame[frame.geometry.intersects(bounds)].copy()
            frame.geometry = frame.geometry.intersection(bounds)
            frame = frame[(~frame.geometry.is_empty) & (frame.geometry.area >= .1)]
            identifier = 'treeID' if 'treeID' in frame else 'tree_id'
            ignore = shapely.GeometryCollection()
            if row.get('ignore_vector'):
                ignore = shapely.from_wkb(Path(row['ignore_vector']).read_bytes()).intersection(bounds)
            metadata = {key: row.get(key, '') for key in (
                'dataset_id', 'source_dataset', 'collection', 'group_id', 'annotation_method')}
            metadata.update(split=split, crs=str(frame.crs) if frame.crs else None,
                            source_cache=str(cached), source_cache_sha256=sha256(cached),
                            gt_ids=frame[identifier].astype(int).tolist(),
                            gt_wkb=[geometry.wkb_hex for geometry in frame.geometry],
                            ignore_wkb=ignore.wkb_hex, bounds=list(bounds.bounds))
            del arrays
            started = time.monotonic()
            fixed = geometric_partition(coord, .5)
            fixed_seconds = time.monotonic() - started
            torch.cuda.synchronize()
            started = time.monotonic()
            with torch.no_grad():
                features = torch.from_numpy(old['feature'].astype(np.float32)).cuda()
                edges = torch.from_numpy(old['edge'].astype(np.int64)).cuda()
                latent = embedding(features)
                weight = edge_affinity(latent, edges).cpu().numpy().astype(np.float32)
                latent = latent.cpu().numpy().astype(np.float32)
            torch.cuda.synchronize()
            embedding_seconds = time.monotonic() - started
            if len(old['edge']):
                context = PartitionContext(coord, latent, old['edge'], weight, csr)
                learned, chosen, search = budget_match(lambda reg: context.run_ez(reg, ezsp),
                    len(fixed) / len(np.unique(fixed)), lower=1e-5, upper=20., steps=8)
                del context
            else:
                learned = np.arange(len(coord), dtype=np.int32)
                chosen = dict(regularization=None, final_partition_seconds=0., total_search_seconds=0.)
                search = []
            data = dict(coord=coord, feature=old['feature'], world_xy=world,
                        feat=crop['feat'].numpy(), tree_id=crop['tree_id'].numpy(),
                        semantic_target=crop['semantic_target'].numpy(),
                        instance_offset=crop['instance_offset'].numpy())
            for name, partition in [('fixed', fixed), ('ezsp', learned)]:
                before = time.monotonic()
                groups, centers, graph = group_geometry(coord, partition)
                metadata[f'{name}_graph_seconds'] = time.monotonic() - before
                data.update({f'{name}_groups': groups, f'{name}_centers': centers, f'{name}_edges': graph})
                metadata[f'{name}_compression'] = len(coord) / len(centers)
            metadata.update(fixed_partition_seconds=fixed_seconds, embedding_seconds=embedding_seconds,
                            ezsp_partition_seconds=chosen['final_partition_seconds'],
                            ezsp_budget_search_seconds=chosen['total_search_seconds'],
                            regularization=chosen['regularization'], search=search,
                            preparation_seconds=time.monotonic() - start)
            folder = root / 'cache' / split
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / f"{row['dataset_id']}.npz"
            np.savez_compressed(path, **data)
            save_json(path.with_suffix('.json'), metadata)
            entries.append({**metadata, 'cache': str(path), 'cache_sha256': sha256(path)})
            save_json(root / 'progress.json', dict(entries=entries))
            print(f"prepare {split} {index+1}/{min(len(rows), args.max_plots or len(rows))} "
                  f"{row['dataset_id']} n={len(coord)} {metadata['preparation_seconds']:.1f}s", flush=True)
    config = dict(entries=entries, seed=20261001, initial_checkpoint=str(INITIAL),
                  initial_sha256=sha256(INITIAL), manifest_sha256=sha256(REAL),
                  embedding_checkpoint=str(embedding_path), embedding_sha256=sha256(embedding_path),
                  encoder='frozen retained LitePT; previous HELIOS exposure',
                  protocol='one deterministic 20m crop per parent plot; native ALS training and validation; no test',
                  ezsp_budget='per-crop label-free search matching fixed 0.5m groups; research overhead included in cost logs',
                  graph='centroid 3m/k16 undirected', torch_scatter_compat=shim,
                  crown_protocol='complete native crown polygons clipped to 20m crop; explicit annotation ignore mask',
                  code_sha256={str(Path(__file__).relative_to(PROJECT)): sha256(Path(__file__))})
    save_json(root / 'prepared.json', config)
    print(f'Prepared {len(entries)} crops: {root}', flush=True)


if __name__ == '__main__':
    main()
