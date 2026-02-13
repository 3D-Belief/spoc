#!/usr/bin/env python3
import argparse
import itertools
import math
import random
from pathlib import Path
from typing import List

def list_scenes(base_dir: Path, only_dirs: bool = True, glob: str = "*") -> List[Path]:
    """Return sorted list of scene paths under base_dir."""
    if not base_dir.exists():
        raise FileNotFoundError(f"Base dir not found: {base_dir}")
    items = sorted(base_dir.glob(glob))
    if only_dirs:
        items = [p for p in items if p.is_dir()]
    return items

def parse_gpu_list(gpus_str: str) -> List[int]:
    """
    Parse GPU list like '0-7' or '0,1,3,5' or '0-3,6,7'.
    """
    result = []
    for chunk in gpus_str.split(","):
        chunk = chunk.strip()
        if "-" in chunk:
            a, b = chunk.split("-", 1)
            result.extend(range(int(a), int(b) + 1))
        else:
            result.append(int(chunk))
    # Deduplicate while preserving order
    seen = set()
    out = []
    for g in result:
        if g not in seen:
            seen.add(g)
            out.append(g)
    return out

def split_round_robin(items: List[Path], num_bins: int) -> List[List[Path]]:
    """Distribute items into num_bins bins as evenly as possible (round-robin)."""
    bins = [[] for _ in range(num_bins)]
    for idx, item in enumerate(items):
        bins[idx % num_bins].append(item)
    return bins

def chunk_by_size(items: List[Path], chunk_size: int) -> List[List[Path]]:
    """Split items into chunks of size <= chunk_size."""
    return [list(x) for x in (itertools.islice(items, i, i + chunk_size) for i in range(0, len(items), chunk_size))]

def write_groups(
    groups: List[List[Path]],
    gpus: List[int],
    groups_per_gpu: int,
    out_dir: Path
):
    out_dir.mkdir(parents=True, exist_ok=True)
    assert len(groups) == len(gpus) * groups_per_gpu

    group_idx = 0
    for gpu in gpus:
        for k in range(groups_per_gpu):
            group = groups[group_idx]
            group_idx += 1
            out_path = out_dir / f"gpu_{gpu}_group_{k}.txt"
            with out_path.open("w") as f:
                for p in group:
                    f.write(str(p.resolve()) + "\n")

def main():
    p = argparse.ArgumentParser(
        description="Split scenes into gpu_{id}_group_{k}.txt files for GPUs 0-7."
    )
    p.add_argument("--base_dir", required=True, type=str,
                   help="Directory that contains all scene folders (houses).")
    p.add_argument("--house_groups_dir", type=str, default="house_groups",
                   help="Output directory to write group files.")
    p.add_argument("--gpus", type=str, default="2-5,7",
                   help="GPU list: '0-7' or '0,1,3,5' or '0-3,6,7'. Default: 0-7")
    group = p.add_mutually_exclusive_group()
    group.add_argument("--groups_per_gpu", type=int, default=None,
                       help="Number of group files per GPU (evenly balanced).")
    group.add_argument("--max_scenes_per_group", type=int, default=None,
                       help="Max scenes per group file (derives number of groups).")
    p.add_argument("--shuffle", action="store_true",
                   help="Shuffle scenes before splitting.")
    p.add_argument("--seed", type=int, default=42,
                   help="RNG seed used when --shuffle is set.")
    p.add_argument("--glob", type=str, default="*",
                   help="Glob for selecting scene folders under base_dir (default '*').")
    p.add_argument("--include_files", action="store_true",
                   help="If set, include files too (default is directories only).")
    args = p.parse_args()

    base_dir = Path(args.base_dir)
    out_dir = Path(args.house_groups_dir)
    gpus = parse_gpu_list(args.gpus)

    scenes = list_scenes(base_dir, only_dirs=(not args.include_files), glob=args.glob)
    if not scenes:
        raise SystemExit(f"No scenes found under {base_dir} with glob '{args.glob}'")

    if args.shuffle:
        random.seed(args.seed)
        random.shuffle(scenes)

    # Determine total groups and per-GPU grouping
    if args.max_scenes_per_group is not None:
        total_groups = max(1, math.ceil(len(scenes) / args.max_scenes_per_group))
        # Make total groups a multiple of number of GPUs so each GPU gets same count
        if total_groups % len(gpus) != 0:
            total_groups = math.ceil(total_groups / len(gpus)) * len(gpus)
        groups_per_gpu = total_groups // len(gpus)
        # Make round-robin bins across total_groups
        bins = [[] for _ in range(total_groups)]
        for i, s in enumerate(scenes):
            bins[i % total_groups].append(s)
        groups = bins
    else:
        # Default or explicit groups_per_gpu
        groups_per_gpu = args.groups_per_gpu if args.groups_per_gpu is not None else 1
        total_groups = len(gpus) * groups_per_gpu
        groups = split_round_robin(scenes, total_groups)

    write_groups(groups, gpus, groups_per_gpu, out_dir)

    # Summary
    total = len(scenes)
    print(f"Found {total} scene(s) in: {base_dir}")
    print(f"GPUs: {gpus}")
    print(f"Groups per GPU: {groups_per_gpu}  (total groups: {len(groups)})")
    for i, gpu in enumerate(gpus):
        counts = [len(groups[i*groups_per_gpu + k]) for k in range(groups_per_gpu)]
        print(f"  GPU {gpu}: {counts}  (sum={sum(counts)})")
    print(f"Group files written to: {out_dir.resolve()}")
    print("Example file names match your runner: gpu_{id}_group_{k}.txt")

if __name__ == "__main__":
    main()
