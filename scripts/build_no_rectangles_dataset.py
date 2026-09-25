#!/usr/bin/env python3
"""Create an independent combined-data version without IDTREES box labels."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[2]


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha256(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            value.update(block)
    return value.hexdigest()


def copy_plot(task):
    old, new, row = task
    if not new.exists():
        new.parent.mkdir(parents=True, exist_ok=True)
        # Reflinks are independent copy-on-write files where supported and
        # ordinary copies elsewhere. Never use symlinks or hardlinks.
        subprocess.run(["cp", "-a", "--reflink=auto", str(old), str(new)], check=True)
    metadata = new / "metadata.json"
    if metadata.exists():
        content = json.loads(metadata.read_text())
        content.update({key: row[key] for key in ("source_las", "output", "gt_vector", "chm")})
        metadata.write_text(json.dumps(content, indent=2) + "\n")
    return row["dataset_id"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=WORKSPACE / "combined_als_crowns_v1")
    parser.add_argument("--output", type=Path, default=WORKSPACE / "combined_als_crowns_no_rectangles_v2")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    original = read_csv(source / "manifest.csv")
    removed = [r for r in original if r["collection"] == "IDTREES"]
    rows = [dict(r) for r in original if r["collection"] != "IDTREES"]
    for row in rows:
        old_folder = Path(row["output"]).parent
        new_folder = output / row["model_split"] / row["dataset_id"]
        row.update(source_las=str(new_folder / "points.las"),
                   output=str(new_folder / "points_0p25m.npz"),
                   gt_vector=str(new_folder / "crowns_full.gpkg"),
                   chm=str(new_folder / Path(row["chm"]).name))
    tasks = [(Path(old["output"]).parent, output / new["model_split"] / new["dataset_id"], new)
             for old, new in zip([r for r in original if r["collection"] != "IDTREES"], rows, strict=True)]
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for index, identifier in enumerate(executor.map(copy_plot, tasks), 1):
            if index % 20 == 0 or index == len(tasks):
                print(f"Copied {index}/{len(tasks)}: {identifier}", flush=True)
    write_csv(output / "manifest.csv", rows)
    for split in ("train", "val", "test"):
        write_csv(output / f"manifest_{split}.csv", (r for r in rows if r["model_split"] == split))
    kept_ids = {r["dataset_id"] for r in rows}
    links = read_csv(source / "spatial_links.csv")
    write_csv(output / "spatial_links.csv", (r for r in links if r["a"] in kept_ids and r["b"] in kept_ids))
    for name in ("missing_sources.csv", "grid_repair_report.json"):
        if (source / name).exists():
            shutil.copy2(source / name, output / name)
    inventory = []
    for item in json.loads((source / "split_inventory.json").read_text()):
        if item["collection"] == "IDTREES":
            continue
        folder = output / item["model_split"] / item["dataset_id"]
        item.update(source_las=str(folder / "points.las"), output=str(folder / "points_0p25m.npz"),
                    gt_vector=str(folder / "crowns_full.gpkg"), chm=str(folder / Path(item["chm"]).name))
        inventory.append(item)
    (output / "split_inventory.json").write_text(json.dumps(inventory, indent=2)+"\n")
    split_counts = Counter(r["model_split"] for r in rows)
    crown_counts = {split: sum(int(r["instances"]) for r in rows if r["model_split"] == split)
                    for split in ("train", "val", "test")}
    report = dict(created_utc=datetime.now(timezone.utc).isoformat(), source_dataset=str(source),
                  policy="Exclude the entire IDTREES collection because its released ITC labels are bounding rectangles, not detailed crown masks.",
                  retained_plots=len(rows), removed_plots=len(removed), removed_collection="IDTREES",
                  retained_split_counts=dict(split_counts), retained_crown_counts=crown_counts,
                  split_policy="Preserve v1 spatial groups and train/val/test assignments; never move an exposed test plot into training.",
                  source_manifest_sha256=sha256(source / "manifest.csv"),
                  removed_dataset_ids=[r["dataset_id"] for r in removed],
                  note="Other native point-label sources are retained even when a coarse 0.5 m projection happens to be rectangular; those are not released bounding-box annotations.")
    (output / "exclusion_report.json").write_text(json.dumps(report, indent=2)+"\n")
    summary = dict(seed=20260925, counts=dict(split_counts), by_source=dict(Counter(f"{r['model_split']}/{r['source_dataset']}/{r['collection']}" for r in rows)),
                   target="full crowns without IDTREES weak rectangular supervision", removed_collection="IDTREES",
                   test_status="same historically exposed spatial groups as v1; not a pristine independent test",
                   selection="source-balanced full-crown PQ@0.50 on validation; test not for tuning")
    (output / "split_summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    manifest_hash = sha256(output / "manifest.csv")
    (output / "READY.json").write_text(json.dumps(dict(plots=len(rows), manifest_sha256=manifest_hash,
            parent_manifest_sha256=report["source_manifest_sha256"], excluded="IDTREES"), indent=2)+"\n")
    (output / "README.md").write_text(f"""# Combined full-crown dataset without rectangular IDTREES labels, v2

Independent filtered copy of `combined_als_crowns_v1`. The complete IDTREES
collection was excluded because all 1763 retained labels were released bounding
rectangles rather than detailed crown masks. Source data and v1 remain unchanged.

Plots: **{len(rows)}** — **{split_counts['train']} train / {split_counts['val']} val /
{split_counts['test']} test**. Reference crowns: **{crown_counts['train']} /
{crown_counts['val']} / {crown_counts['test']}**. Existing spatial-group split
assignments are preserved, so no previously exposed test plot enters training.

`exclusion_report.json` lists every removed plot and the parent checksum.
Directories contain independent files (copy-on-write reflinks where supported),
not symlinks or hardlinks. The class-3 annotation-ignore and full-crown IoU
evaluation rules from v1 remain applicable. This is still a historically exposed,
heterogeneous benchmark, not a pristine publication holdout.
""")
    print(json.dumps({"output": str(output), "plots": len(rows), "splits": dict(split_counts), "crowns": crown_counts, "manifest_sha256": manifest_hash}, indent=2))


if __name__ == "__main__":
    main()
