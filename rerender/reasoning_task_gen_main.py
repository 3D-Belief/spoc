# generate_visibility_trajectories.py
import os
import sys
import json
import math
import gzip
import glob
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import importlib, pkgutil

from PIL import Image
from tqdm import tqdm

ROOT_DIR = str(Path(__file__).parent.parent)
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

os.environ["OBJAVERSE_DATA_DIR"] = "/home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/spoc/data"
os.environ["OBJAVERSE_HOUSES_DIR"] = "/home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/spoc/data/houses_2023_07_28"

from environment.stretch_controller import StretchController
from spoc_utils.constants.stretch_initialization_utils import STRETCH_ENV_ARGS
import rerender.embodied_tasks as plugins
for m in pkgutil.iter_modules(plugins.__path__, plugins.__name__ + "."):
    importlib.import_module(m.name)

from rerender.registry import get_embodied_task, all_embodied_tasks, get_constant, all_constants

# ---------------- house loading  (same pattern as rerendering script) ----------------

def load_house_from_prior(house_index=0):
    """Load house JSON by index from local gz JSONL. Falls back to a minimal house."""
    from spoc_utils.constants.objaverse_data_dirs import OBJAVERSE_HOUSES_DIR

    for split in ["val"]:
        houses_path = os.path.join(OBJAVERSE_HOUSES_DIR, f"{split}.jsonl.gz")
        if not os.path.exists(houses_path):
            continue
        try:
            cur = 0
            with gzip.open(houses_path, 'rt', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    house = json.loads(line)
                    if cur == house_index:
                        return house
                    cur += 1
        except Exception:
            pass

    # fallback
    return {
        "id": f"default_{house_index}",
        "objects": [],
        "rooms": [],
        "scene_bounds": {
            "center": {"x": 0, "y": 0, "z": 0},
            "size": {"x": 10, "y": 3, "z": 10},
        },
    }

def process_houses(
    task_name: str,
    output_base_dir: str,
    start_house: int = 0,
    num_houses: int = 10,
    max_objects_per_house: int = 5,
    rotate_dir: str = "random",
    deg_step: int = 3,
    resume: bool = True,
):
    """
    Iterate houses by index, create episodes per object. One episode dir per (house, object).
    """
    os.makedirs(output_base_dir, exist_ok=True)
    summary_path = os.path.join(output_base_dir, "summary.json")
    progress = {
        "processed": [],
        "errors": []
    }
    if resume and os.path.exists(summary_path):
        try:
            progress = json.load(open(summary_path, "r"))
        except Exception:
            pass

    for hi in tqdm(range(start_house, start_house + num_houses), desc="Houses"):
        house = load_house_from_prior(hi)
        if house is None:
            progress["errors"].append(f"House index {hi} could not be loaded.")
            continue
        house_id = house.get("id", f"index_{hi}")

        # Skip if this house already processed (optional)
        already = any(h.get("house_id") == house_id for h in progress["processed"])
        if resume and already:
            continue

        # init controller
        controller = StretchController(**STRETCH_ENV_ARGS)
        controller.reset(house)
        
        # List candidate objects visible at the spawn view (seed list). We'll try to expand if empty.
        entity_type = "objects" if "door" not in task_name else "doors"
        scene_graph = controller.current_scene_json.get(entity_type, [])
        obj_ids = [obj.get("id", "") for obj in scene_graph]
        # Exclude certain key words from object IDs
        task_constant = [c for c in all_constants() if c.startswith(task_name)][0]
        exclude = True if 'exclusion' in task_constant else False
        keys = get_constant(task_constant)
        want_include = not exclude
        obj_ids = [oid for oid in obj_ids if (any(k in oid.lower() for k in keys)) == want_include]
        
        if len(obj_ids) == 0:
            continue

        house_done = {"house_id": house_id, "episodes": [], "errors": []}

        # Limit how many objects per house
        tried = 0
        for oid in obj_ids:
            if tried >= max_objects_per_house:
                break
            ep_dir = os.path.join(output_base_dir, f"house_{house_id}_obj_{oid.replace('|','_').replace(':','_')}")
            if resume and os.path.exists(os.path.join(ep_dir, "trajectory_metadata.json")):
                # already exists
                continue

            # try:
            if rotate_dir == "random":
                rotate_dir_eff = np.random.choice(["left", "right"])
            else:
                rotate_dir_eff = rotate_dir
            os.makedirs(ep_dir, exist_ok=True)
            meta = get_embodied_task(task_name)(
                controller=controller,
                house_id=house_id,
                object_id=oid,
                out_dir=ep_dir,
                rotate_dir=rotate_dir_eff,
                deg_step=deg_step,
            )
            if meta is None:
                # failed; skip silently
                continue
            house_done["episodes"].append({
                "object_id": oid,
                "output_dir": ep_dir,
                "frames": meta.get("frames", 0)
            })
            tried += 1
            # except Exception as e:
            #     house_done["errors"].append(f"{oid}: {e}")

        controller.stop()

        progress["processed"].append(house_done)
        with open(summary_path, "w") as f:
            json.dump(progress, f, indent=2)

    return progress


# ---------------- CLI ----------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate visibility-based rotation trajectories (RGB/Depth/Semantic/Pose) per object")
    parser.add_argument("--task_name", type=str, default="visibility_trajectory", help="Name of the task to run")
    parser.add_argument("--output_dir", type=str, default="data/visibility_trajs", help="Output base directory")
    parser.add_argument("--start_house", type=int, default=0, help="Start house index in local JSONL")
    parser.add_argument("--num_houses", type=int, default=10, help="How many houses to process")
    parser.add_argument("--max_objects_per_house", type=int, default=5, help="Limit objects per house")
    parser.add_argument("--rotate_dir", type=str, default="random", choices=["left", "right", "random"], help="Rotation direction")
    parser.add_argument("--deg_step", type=int, default=6, help="Rotation step in degrees")
    parser.add_argument("--resume", action="store_true", help="Resume with existing outputs")
    args = parser.parse_args()

    summary = process_houses(
        task_name=args.task_name,
        output_base_dir=args.output_dir,
        start_house=args.start_house,
        num_houses=args.num_houses,
        max_objects_per_house=args.max_objects_per_house,
        rotate_dir=args.rotate_dir,
        deg_step=args.deg_step,
        resume=args.resume,
    )
    print("Done. Summary written to", os.path.join(args.output_dir, "visibility_summary.json"))
