#!/usr/bin/env python3
import os
import sys
import h5py
import json
import numpy as np
import cv2
import gzip
import glob
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

# --- Project paths / imports ---------------------------------------------------
ROOT_DIR = str(Path(__file__).parent.parent)
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

# Environment variables used by the loaders
os.environ["OBJAVERSE_DATA_DIR"] = "/home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/spoc/data"
os.environ["OBJAVERSE_HOUSES_DIR"] = "/home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/spoc/data/houses_2023_07_28"

from spoc_utils.constants.stretch_initialization_utils import STRETCH_ENV_ARGS
from spoc_utils.embodied_utils import find_object_node  # optional, only if needed
from environment.stretch_controller import StretchController


# --- Small utilities -----------------------------------------------------------

def create_video_from_frames(frames_dir, output_path, fps=10):
    frames = sorted([f for f in os.listdir(frames_dir) if f.startswith('frame_') and f.endswith('.png')])
    if not frames:
        return False
    first = cv2.imread(os.path.join(frames_dir, frames[0]))
    if first is None:
        return False
    h, w, _ = first.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vw = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    for fn in frames:
        img = cv2.imread(os.path.join(frames_dir, fn))
        if img is not None:
            vw.write(img)
    vw.release()
    return True


def find_object_info(scene_graph, target_id):
    """
    Recursively search for an object with a given id in a scene graph.

    Args:
        scene_graph (list[dict]): The scene graph (list of dicts, each may have children).
        target_id (str): The object id to search for.

    Returns:
        str or None: objectType (or name) if found, otherwise None.
    """
    for node in scene_graph:
        if node.get("id") == target_id:
            return node.get("objectType")

        children = node.get("children", [])
        if children:
            result = find_object_info(children, target_id)
            if result is not None:
                return result
    return None


def load_house_from_prior(house_index: int):
    """Lightweight loader for a house JSON from local gz JSONL."""
    from spoc_utils.constants.objaverse_data_dirs import OBJAVERSE_HOUSES_DIR
    path = os.path.join(OBJAVERSE_HOUSES_DIR, "val.jsonl.gz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing houses file: {path}")
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i == house_index:
                return json.loads(line)
    raise IndexError(f"House index {house_index} out of range in {path}")


def nav_camera_pose_from_event(event):
    """Return cam->world 4x4 for the navigation camera (right-handed)."""
    pos = event.metadata["cameraPosition"]
    agent = event.metadata["agent"]
    yaw = agent["rotation"]["y"]
    pitch = agent["cameraHorizon"]
    R_aw = R.from_euler('xyz', [0.0, yaw, 0.0], degrees=True).as_matrix()
    R_ca = R.from_euler('xyz', [pitch, 0.0, 0.0], degrees=True).as_matrix()
    R_cw = R_aw @ R_ca
    t_cw = np.array([pos["x"], pos["y"], pos["z"]], dtype=float)
    T = np.eye(4, dtype=float)
    T[:3, :3] = R_cw
    T[:3, 3] = t_cw
    F = np.diag([1, 1, -1, 1])  # LH->RH fix if needed by the engine
    return F @ T @ F


def is_forward_action(token: str) -> bool:
    """Heuristic: treat these as forward motion."""
    if token is None:
        return False
    t = token.lower()
    return t in ("m", "moveahead")


def is_rotation_action(token: str) -> bool:
    """Heuristic: tokens that represent pure rotation."""
    if token is None:
        return False
    t = token.lower()
    return t in ("l", "r", "ls", "rs", "rotateleft", "rotateright")


# --- Core: camera-shaking segment generator ------------------------------------

def replay_camera_shaking_segment(
    house_id: str,
    house_data: dict,
    trajectory_data: dict,
    output_dir: str,
    *,
    min_seq_len: int = 30,
    max_seq_len: int = 50,
    shake_deg: float = 15.0,
    shake_dir: str = "random",  # "left", "right", or "random"
):
    """
    For a given (house, episode) trajectory:
      - sample a subsequence of length in [min_seq_len, max_seq_len],
      - replay it with TeleportFull at recorded positions,
      - add camera shaking: during forward steps, gradually change yaw up to ±shake_deg,
        keeping rotation steps with original yaw.

    Saves RGB / Depth / Semantic / Pose for each frame in the window, plus metadata.
    Returns dict(frame_count=...).
    """

    os.makedirs(output_dir, exist_ok=True)
    rgb_dir = os.path.join(output_dir, "rgb")
    depth_dir = os.path.join(output_dir, "depth")
    sem_dir = os.path.join(output_dir, "semantic")
    pose_dir = os.path.join(output_dir, "pose")
    for d in (rgb_dir, depth_dir, sem_dir, pose_dir):
        os.makedirs(d, exist_ok=True)

    positions = trajectory_data.get('positions', [])
    rotations = trajectory_data.get('rotations', [])
    actions = trajectory_data.get('actions', [])

    T = min(len(positions), len(rotations), len(actions))
    if T < min_seq_len:
        print(f"[CameraShake] Episode too short (T={T}); skip.")
        return {"frame_count": 0}

    # --- Sample window [win_start, win_end) ------------------------------------
    max_len = min(max_seq_len, T)
    seq_len = np.random.randint(min_seq_len, max_len + 1)
    max_start = T - seq_len
    win_start = np.random.randint(0, max_start + 1) if max_start > 0 else 0
    win_end = win_start + seq_len

    window_indices = list(range(win_start, win_end))

    # --- Compute shaken yaw schedule -------------------------------------------
    # Default: yaw_new[t] = original yaw
    yaw_new = {i: float(rotations[i].get('y', 0.0)) for i in window_indices}

    # Pick shake direction
    if shake_dir == "random":
        sign = np.random.choice([+1.0, -1.0])
        shake_direction = "left" if sign > 0 else "right"
    elif shake_dir == "left":
        sign = +1.0
        shake_direction = "left"
    else:
        sign = -1.0
        shake_direction = "right"

    # Identify forward segments within the window
    forward_flags = [is_forward_action(actions[i]) for i in window_indices]

    idx = 0
    while idx < len(window_indices):
        if not forward_flags[idx]:
            idx += 1
            continue

        # start of forward block
        j = idx
        while j < len(window_indices) and forward_flags[j]:
            j += 1

        # forward block is window_indices[idx:j]
        seg_len = j - idx

        # We want jitter that starts and ends at 0 offset:
        # offset(k) = sign * shake_deg * sin(pi * phase), phase in [0,1]
        # so k=0 and k=seg_len-1 both have offset ~0.
        if seg_len >= 2:
            for k in range(seg_len):
                t = window_indices[idx + k]
                base_yaw = float(rotations[t].get('y', 0.0))
                phase = k / float(seg_len - 1)  # 0 .. 1
                offset = sign * shake_deg * np.sin(np.pi * phase)
                yaw_new[t] = base_yaw + offset
        # If seg_len == 1, we skip shaking to avoid a single-step jump.

        idx = j
    
    # --- Initialize controller & replay ----------------------------------------
    controller = StretchController(**STRETCH_ENV_ARGS)
    controller.reset(house_data)

    frame_count = 0
    W = H = None

    # helper: save current frame
    def save_frame(frame_idx: int):
        nonlocal frame_count, W, H
        rgb = controller.navigation_camera
        h, w = rgb.shape[:2]
        size = min(h, w)
        sx, sy = (w - size) // 2, (h - size) // 2
        rgb_sq = rgb[sy:sy+size, sx:sx+size]
        Image.fromarray(rgb_sq).save(os.path.join(rgb_dir, f"frame_{frame_idx:04d}.png"))

        # set dims
        H, W = size, size

        # depth
        if hasattr(controller, 'navigation_depth_frame'):
            dimg = controller.navigation_depth_frame
            if dimg.ndim == 2:
                d_sq = dimg[sy:sy+size, sx:sx+size]
            else:
                d_sq = dimg[sy:sy+size, sx:sx+size, :]
            np.save(os.path.join(depth_dir, f"frame_{frame_idx:04d}_raw.npy"), d_sq)
            dmax = float(np.max(d_sq)) if np.size(d_sq) > 0 else 0.0
            if dmax > 0:
                d_viz = (d_sq / dmax * 255).astype(np.uint8)
            else:
                d_viz = np.zeros_like(d_sq, dtype=np.uint8)
            cv2.imwrite(os.path.join(depth_dir, f"frame_{frame_idx:04d}.png"), d_viz)

        # semantic (raw + object metadata)
        if hasattr(controller, 'navigation_camera_segmentation'):
            seg = controller.navigation_camera_segmentation
            seg_sq = seg[sy:sy+size, sx:sx+size]
            Image.fromarray(seg_sq).save(os.path.join(sem_dir, f"frame_{frame_idx:04d}.png"))
            np.save(os.path.join(sem_dir, f"frame_{frame_idx:04d}_raw.npy"), seg_sq)

            frame_objects = {}
            colors = np.unique(seg_sq.reshape(-1, 3), axis=0)
            colors = {tuple(int(c) for c in color) for color in colors}
            color_to_id = getattr(controller.controller.last_event, "color_to_object_id", {})

            for obj_color in colors:
                obj_id = color_to_id.get(obj_color, None)
                if obj_id is None:
                    continue
                if obj_id not in frame_objects:
                    obj_name = "Unknown"
                    if hasattr(controller, 'current_scene_json') and 'objects' in controller.current_scene_json:
                        scene_graph = controller.current_scene_json['objects']
                        obj_name = find_object_info(scene_graph, obj_id) or "Unknown"
                        if "Obja" in obj_name:
                            obj_name = obj_name.replace("Obja", "")

                    frame_objects[obj_id] = {
                        "color": tuple(obj_color),
                        "rgb_string": f"rgb({obj_color[0]}, {obj_color[1]}, {obj_color[2]})",
                        "hex_color": f"#{obj_color[0]:02x}{obj_color[1]:02x}{obj_color[2]:02x}",
                        "name": obj_name,
                        "instance_id": obj_id
                    }

            metadata = {
                "scene_metadata": {
                    "total_objects": len(frame_objects),
                    "object_categories": list(set(obj['name'] for obj in frame_objects.values()))
                },
                "objects": frame_objects
            }
            meta_path = os.path.join(sem_dir, f"frame_{frame_idx:04d}_meta.json")
            with open(meta_path, 'w') as f:
                json.dump(metadata, f, indent=2)

        # pose
        Tcw = nav_camera_pose_from_event(controller.controller.last_event)
        np.save(os.path.join(pose_dir, f"frame_{frame_idx:04d}_pose.npy"), Tcw)

        frame_count += 1

    # Teleport through the window with shaken yaws
    for frame_idx, t in enumerate(window_indices):
        pos = positions[t]
        rot = rotations[t]
        yaw = float(yaw_new[t])
        controller.step(
            action="TeleportFull",
            position={"x": pos['x'], "y": pos['y'], "z": pos['z']},
            rotation={"x": rot.get('x', 0.0), "y": yaw, "z": rot.get('z', 0.0)},
            horizon=0.0,
            standing=True,
        )
        save_frame(frame_idx)

    controller.stop()

    # --- Metadata --------------------------------------------------------------
    meta = {
        "house_id": house_id,
        "frames": int(frame_count),
        "width": int(W) if W is not None else None,
        "height": int(H) if H is not None else None,
        "window_start_step": int(win_start),
        "window_end_step": int(win_end),
        "seq_len": int(seq_len),
        "shake_deg": float(shake_deg),
        "shake_direction": shake_direction,
        "min_seq_len": int(min_seq_len),
        "max_seq_len": int(max_seq_len),
        "task_type": "camera_shaking",
    }
    with open(os.path.join(output_dir, "trajectory_metadata.json"), 'w') as f:
        json.dump(meta, f, indent=2)

    return {"frame_count": frame_count}


# --- Batch processing for camera shaking --------------------------------------

def process_all_camera_shaking(
    base_dir: str,
    output_base_dir: str,
    *,
    min_seq_len: int = 30,
    max_seq_len: int = 50,
    shake_deg: float = 15.0,
    shake_dir: str = "random",
    start_house: str = None,
    end_house: str = None,
    houses_file: str = None,
    make_videos: bool = True,
):
    os.makedirs(output_base_dir, exist_ok=True)

    # Discover house dirs
    if houses_file and os.path.exists(houses_file):
        with open(houses_file, 'r') as f:
            house_dirs = [line.strip() for line in f if line.strip()]
    else:
        house_dirs = sorted(glob.glob(os.path.join(base_dir, "*")))

    # Optional range filter (by numeric folder name)
    if start_house is not None or end_house is not None:
        filtered = []
        for hdir in house_dirs:
            hid = os.path.basename(hdir)
            if not os.path.isdir(hdir) or not hid.isdigit():
                continue
            if start_house is not None and hid < start_house:
                continue
            if end_house is not None and hid > end_house:
                continue
            filtered.append(hdir)
        house_dirs = filtered

    total_eps = 0
    total_frames = 0
    skipped = 0

    for house_dir in tqdm(house_dirs, desc="Houses"):
        house_id = os.path.basename(house_dir)
        if not os.path.isdir(house_dir) or not house_id.isdigit():
            continue
        hdf5_path = os.path.join(house_dir, "hdf5_sensors.hdf5")
        if not os.path.exists(hdf5_path):
            print(f"[CameraShake/Batch] Missing HDF5 for house {house_id}, skip.")
            skipped += 1
            continue

        house_index = int(house_id)
        try:
            house_data = load_house_from_prior(house_index)
        except Exception as e:
            print(f"[CameraShake/Batch] Could not load house {house_id}: {e}")
            skipped += 1
            continue

        # enumerate episode indices in HDF5
        try:
            with h5py.File(hdf5_path, 'r') as f:
                episode_keys = sorted(
                    [k for k in f.keys() if k.isdigit() or (k.startswith('0') and k[1:].isdigit())],
                    key=lambda x: int(x)
                )
        except Exception as e:
            print(f"[CameraShake/Batch] Could not read episodes in {hdf5_path}: {e}")
            skipped += 1
            continue

        for ep in tqdm(episode_keys, leave=False, desc=f"Episodes/{house_id}"):
            try:
                with h5py.File(hdf5_path, 'r') as f:
                    if f'{ep}/last_agent_location' not in f or f'{ep}/last_action_str' not in f:
                        continue
                    traj = {'positions': [], 'rotations': [], 'actions': []}

                    locs = f[f'{ep}/last_agent_location'][:]
                    for loc in locs:
                        if len(loc) >= 6:
                            traj['positions'].append({'x': float(loc[0]), 'y': float(loc[1]), 'z': float(loc[2])})
                            traj['rotations'].append({'x': 0.0, 'y': float(loc[4]), 'z': 0.0})

                    act_bytes = f[f'{ep}/last_action_str'][:]
                    for ab in act_bytes:
                        s = ''
                        for b in ab:
                            if b == 0:
                                break
                            s += chr(b)
                        traj['actions'].append(s)

                out_dir = os.path.join(output_base_dir, f"house_{house_id}_episode_{ep}")
                os.makedirs(out_dir, exist_ok=True)

                info = replay_camera_shaking_segment(
                    house_id=str(house_id),
                    house_data=house_data,
                    trajectory_data=traj,
                    output_dir=out_dir,
                    min_seq_len=min_seq_len,
                    max_seq_len=max_seq_len,
                    shake_deg=shake_deg,
                    shake_dir=shake_dir,
                )

                if info.get('frame_count', 0) > 0 and make_videos:
                    vids = os.path.join(out_dir, "videos")
                    os.makedirs(vids, exist_ok=True)
                    for mdir, vname in (("rgb", "rgb.mp4"), ("depth", "depth.mp4"), ("semantic", "semantic.mp4")):
                        dpath = os.path.join(out_dir, mdir)
                        if os.path.isdir(dpath) and os.listdir(dpath):
                            create_video_from_frames(dpath, os.path.join(vids, vname))

                total_eps += 1
                total_frames += int(info.get('frame_count', 0))

            except Exception as e:
                print(f"[CameraShake/Batch] Error on house {house_id} ep {ep}: {e}")
                continue

    print(f"[CameraShake/Batch] Done. Episodes processed: {total_eps}, frames saved: {total_frames}, skipped: {skipped}")


# --- Single-episode helper -----------------------------------------------------

def process_one_episode_camera_shaking(
    house_index: int,
    episode_idx: str,
    hdf5_path: str,
    output_dir_root: str,
    *,
    min_seq_len: int = 30,
    max_seq_len: int = 50,
    shake_deg: float = 15.0,
    shake_dir: str = "random",
    make_videos: bool = True,
):
    house_data = load_house_from_prior(house_index)
    if hdf5_path is None:
        raise ValueError("Please provide --hdf5_path for single-episode mode.")
    with h5py.File(hdf5_path, 'r') as f:
        if episode_idx not in f:
            raise ValueError(f"Episode {episode_idx} not found in HDF5: {hdf5_path}")
        traj = {'positions': [], 'rotations': [], 'actions': []}
        locs = f[f'{episode_idx}/last_agent_location'][:]
        for loc in locs:
            if len(loc) >= 6:
                traj['positions'].append({'x': float(loc[0]), 'y': float(loc[1]), 'z': float(loc[2])})
                traj['rotations'].append({'x': 0.0, 'y': float(loc[4]), 'z': 0.0})
        act_bytes = f[f'{episode_idx}/last_action_str'][:]
        for ab in act_bytes:
            s = ''
            for b in ab:
                if b == 0:
                    break
                s += chr(b)
            traj['actions'].append(s)

    out_dir = os.path.join(output_dir_root, f"house_{house_index}_episode_{episode_idx}")
    os.makedirs(out_dir, exist_ok=True)

    info = replay_camera_shaking_segment(
        house_id=str(house_index),
        house_data=house_data,
        trajectory_data=traj,
        output_dir=out_dir,
        min_seq_len=min_seq_len,
        max_seq_len=max_seq_len,
        shake_deg=shake_deg,
        shake_dir=shake_dir,
    )

    if info.get('frame_count', 0) > 0 and make_videos:
        vids = os.path.join(out_dir, "videos")
        os.makedirs(vids, exist_ok=True)
        for mdir, vname in (("rgb", "rgb.mp4"), ("depth", "depth.mp4"), ("semantic", "semantic.mp4")):
            dpath = os.path.join(out_dir, mdir)
            if os.path.isdir(dpath) and os.listdir(dpath):
                create_video_from_frames(dpath, os.path.join(vids, vname))

    print(f"[CameraShake/Single] Frames saved: {info.get('frame_count', 0)} at {out_dir}")


# --- CLI ----------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build camera-shaking observation segments from existing action data.")
    # Modes
    parser.add_argument('--process_all', action='store_true', help='Process all houses/episodes under base_dir.')
    # IO
    parser.add_argument('--base_dir', type=str,
                        default="/home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/spoc/data/all/ObjectNavType/val",
                        help='Base directory containing house folders (each with hdf5_sensors.hdf5).')
    parser.add_argument('--output_dir', type=str, default="data/camera_shaking_segments_val",
                        help='Output base directory.')
    parser.add_argument('--houses_file', type=str, default=None,
                        help='Optional file listing house directories to process (one per line).')
    parser.add_argument('--start_house', type=str, default=None, help='Start house id (inclusive, numeric as string).')
    parser.add_argument('--end_house', type=str, default=None, help='End house id (inclusive, numeric as string).')
    parser.add_argument('--no_videos', action='store_true', help='Do not create MP4s.')

    # Camera shaking params
    parser.add_argument('--min_seq_len', type=int, default=30, help='Minimum subsequence length.')
    parser.add_argument('--max_seq_len', type=int, default=50, help='Maximum subsequence length.')
    parser.add_argument('--shake_deg', type=float, default=15.0, help='Max yaw offset for camera shaking (degrees).')
    parser.add_argument('--shake_dir', type=str, default="random",
                        choices=["left", "right", "random"],
                        help='Direction of shaking.')

    # Single-episode
    parser.add_argument('--house_index', type=int, default=None, help='House index for single episode.')
    parser.add_argument('--episode_idx', type=str, default=None, help='Episode index (string) for single episode.')
    parser.add_argument('--hdf5_path', type=str, default=None, help='Path to hdf5_sensors.hdf5 for single episode.')

    args = parser.parse_args()

    if args.process_all:
        process_all_camera_shaking(
            base_dir=args.base_dir,
            output_base_dir=args.output_dir,
            min_seq_len=args.min_seq_len,
            max_seq_len=args.max_seq_len,
            shake_deg=args.shake_deg,
            shake_dir=args.shake_dir,
            start_house=args.start_house,
            end_house=args.end_house,
            houses_file=args.houses_file,
            make_videos=not args.no_videos,
        )
    else:
        if args.house_index is None or args.episode_idx is None or args.hdf5_path is None:
            print("Single-episode mode requires --house_index, --episode_idx and --hdf5_path.")
            sys.exit(1)
        process_one_episode_camera_shaking(
            house_index=args.house_index,
            episode_idx=args.episode_idx,
            hdf5_path=args.hdf5_path,
            output_dir_root=args.output_dir,
            min_seq_len=args.min_seq_len,
            max_seq_len=args.max_seq_len,
            shake_deg=args.shake_deg,
            shake_dir=args.shake_dir,
            make_videos=not args.no_videos,
        )
