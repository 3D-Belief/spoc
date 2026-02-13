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

# External helpers assumed available in the codebase
from spoc_utils.embodied_utils import find_agent_room
from spoc_utils.constants.stretch_initialization_utils import STRETCH_ENV_ARGS
from environment.stretch_controller import StretchController

def _door_tangent_and_normal(x0, z0, x1, z1):
    """
    Returns unit tangent (along wall) and unit normal (perpendicular, pointing to one side).
    normal is chosen as rotate90(tangent): n = (-dz, +dx) normalized in xz-plane.
    """
    dx, dz = (x1 - x0), (z1 - z0)
    t = np.array([dx, dz], dtype=float)
    nrm = np.linalg.norm(t)
    if nrm < 1e-6:
        return None, None
    t /= nrm
    # perpendicular (left-hand turn): n = (-dz, +dx)
    n = np.array([-t[1], t[0]], dtype=float)
    return t, n

def _door_center_and_normal(door_node):
    """
    From door node json:
      - center from assetPosition (x,z)
      - wall from wall0 (or wall1) -> wall tangent & normal in xz-plane
    Returns (center_xyz, normal_xz) or (None, None) if unavailable.
    """
    ap = door_node.get("assetPosition", None)
    if not ap:
        return None, None
    x0 = door_node["holePolygon"][0]["x"]
    z0 = door_node["holePolygon"][0]["z"]
    x1 = door_node["holePolygon"][1]["x"]
    z1 = door_node["holePolygon"][1]["z"]

    _, nn = _door_tangent_and_normal(x0, z0, x1, z1)
    if nn is None:
        cc = np.array([ap["x"], ap.get("y", 1.0), ap.get("z", 0.0)], dtype=float)
        return cc, None

    cc = np.array([ap["x"], ap.get("y", 1.0), ap.get("z", 0.0)], dtype=float)
    return cc, nn  # n is 2D in xz-plane

# --- Small utilities -----------------------------------------------------------
EPS = 1e-6

def convert_action(token: str):
    """Map abbreviated token to AI2-THOR action."""
    table = {
        'm': ('MoveAhead', {'moveMagnitude': 0.2}),
        'b': ('MoveBack', {'moveMagnitude': 0.2}),
        'l': ('RotateLeft', {'degrees': 30}),
        'r': ('RotateRight', {'degrees': 30}),
        'ls': ('RotateLeft', {'degrees': 6}),
        'rs': ('RotateRight', {'degrees': 6}),
        'end': ('Done', {}),
        'sub_done': ('SubDone', {}),
        'p': ('PickupObject', {}),
        'd': ('DropObject', {}),
    }
    return table.get(token, ('Pass', {}))

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
        tuple[str, str] or None: (objectType, assetId) if found, otherwise None.
    """
    for node in scene_graph:
        if node.get("id") == target_id:
            return node.get("objectType")

        # search recursively in children if they exist
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

def _dist2d(ax, az, bx, bz):
    return float(np.hypot(ax - bx, az - bz))

def _normalize_room_label(r):
    if not r:
        return 'unknown'
    lab = r.lower() if isinstance(r, str) else str(r).lower()
    if 'living' in lab: return 'living room'
    if 'kitchen' in lab: return 'kitchen'
    if 'bed' in lab: return 'bedroom'
    if 'bath' in lab: return 'bathroom'
    return lab

# --- Core: door-crossing segment generator ------------------------------------
def replay_door_crossing_segment(
    house_id: str,
    house_data: dict,
    trajectory_data: dict,
    output_dir: str,
    *,
    pre_steps: int = 20,
    post_steps: int = 40,
    min_start_pixels: int = 80,
    extra_rot_steps: int = 10, 
):
    """
    Detect a door crossing and save ONLY a window around it.
    Returns dict(frame_count=...).
    """

    # Prepare output dirs
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

    if not positions:
        print("[DoorSegment] No positions; skip.")
        return {'frame_count': 0}

    # Initialize controller
    controller = StretchController(**STRETCH_ENV_ARGS)
    controller.reset(house_data)

    # --- Filter 1: multi-room only
    scene_rooms = (controller.current_scene_json or {}).get("rooms", [])
    if len(scene_rooms) < 2:
        print("[DoorSegment] Single-room scene; skip.")
        controller.stop()
        return {'frame_count': 0}

    # --- Expand rotations to 5x small steps to match the original replay
    expanded_positions = [positions[0]]
    expanded_rotations = [rotations[0]]
    processed_actions = []
    action_to_original = {}
    new_idx = 1
    for orig_step, token in enumerate(actions[1:], 1):
        if not token:
            processed_actions.append("")
            action_to_original[new_idx] = orig_step
            if orig_step < len(positions) and orig_step < len(rotations):
                expanded_positions.append(positions[orig_step])
                expanded_rotations.append(rotations[orig_step])
            new_idx += 1
            continue

        if token in ('l', 'r'):
            if orig_step < len(rotations):
                start_rot = rotations[orig_step-1] if orig_step > 0 else rotations[0]
                start_yaw = float(start_rot.get('y', 0.0))
                sign = -1 if token == 'l' else +1  # the original had - for left; keep consistent with camera convention used
                for i in range(5):
                    processed_actions.append('ls' if token == 'l' else 'rs')
                    action_to_original[new_idx] = orig_step
                    if orig_step < len(positions):
                        sp = positions[orig_step-1] if orig_step > 0 else positions[0]
                        ep = positions[orig_step]
                        t = (i + 1) / 5.0
                        expanded_positions.append({
                            'x': sp.get('x', 0.0) + t * (ep.get('x', 0.0) - sp.get('x', 0.0)),
                            'y': sp.get('y', 0.0) + t * (ep.get('y', 0.0) - sp.get('y', 0.0)),
                            'z': sp.get('z', 0.0) + t * (ep.get('z', 0.0) - sp.get('z', 0.0)),
                        })
                    expanded_rotations.append({
                        'x': start_rot.get('x', 0.0),
                        'y': start_yaw + sign * (i + 1) * 6.0,
                        'z': start_rot.get('z', 0.0),
                    })
                    new_idx += 1
        else:
            processed_actions.append(token)
            action_to_original[new_idx] = orig_step
            if orig_step < len(positions) and orig_step < len(rotations):
                expanded_positions.append(positions[orig_step])
                expanded_rotations.append(rotations[orig_step])
            new_idx += 1

    positions = expanded_positions
    rotations = expanded_rotations

    # --- Door topology helpers
    def _find_doors_between(room_label_a, room_label_b):
        """Pick door nodes whose room0/room1 labels match (order-insensitive)."""
        doors = (controller.current_scene_json or {}).get("doors", [])
        out = []
        for d in doors:
            _, r0, r1 = d.get("id", "").split("|")
            if r0 is None or r1 is None:
                continue
            # Normalize labels (may store canonical labels already)
            if {r0, r1} == { room_label_a.split("|")[1], room_label_b.split("|")[1] }:
                out.append(d)
        return out

    def _center_normal(dnode):
        c, n = _door_center_and_normal(dnode)  # provided helper
        cx, cz = float(c[0]), float(c[2])
        if len(n) == 2:
            nx, nz = float(n[0]), float(n[1])
        else:
            nx, nz = float(n[0]), float(n[2])
        return cx, cz, nx, nz

    # --- Detection pass (no saving): first room change == crossing step
    print("[DoorSegment] Detection pass (room-change heuristic)...")
    controller.step(
        action="TeleportFull",
        position=positions[0],
        rotation=rotations[0],
        horizon=0.0,
        standing=True,
    )

    def _room_tuple_of_pos(px, pz):
        """
        We expect find_agent_room(...) to return something like (room_id, room_label).
        Fall back gracefully if it returns only a label.
        """
        r = find_agent_room(controller.current_scene_json, px, pz)
        if r is None:
            return (None, None)
        if isinstance(r, (list, tuple)):
            if len(r) >= 2:
                return (r[0], r[1])
            # single element
            return (r[0], r[0])
        # string label only
        return (r, r)

    # initial room (id, label)
    init_room = _room_tuple_of_pos(positions[0]['x'], positions[0]['z'])
    last_room = init_room
    cross_step = None
    dest_room_id, dest_room_label = None, None

    for step_idx, token in enumerate(processed_actions, 1):
        if not token:
            continue
        act, params = convert_action(token)
        try:
            event = controller.step(action=act, **params)
        except Exception as e:
            print(f"[DoorSegment] detect step {step_idx} action {act} failed: {e}")
            continue

        curp = event.metadata["agent"]["position"]
        cur_room = _room_tuple_of_pos(curp["x"], curp["z"])

        # If both have ids (or labels) and changed → declare crossing at this step
        if last_room[0] is not None and cur_room[0] is not None and cur_room[0] != last_room[0]:
            print(f"[DoorSegment] Room change detected: {last_room} -> {cur_room} at step {step_idx}")
            cross_step = step_idx
            dest_room_id, dest_room_label = cur_room[0], cur_room[1]
            break

        last_room = cur_room

    if cross_step is None:
        print("[DoorSegment] No room change found; skip episode.")
        controller.stop()
        return {'frame_count': 0}

    # window around the detected step (inclusive start on actions, exclusive end)
    win_start = max(0, cross_step - pre_steps)
    win_end = min(len(processed_actions), cross_step + post_steps)

    print(f"[DoorSegment] Saving pass window = [{win_start}, {win_end}) based on cross_step={cross_step}")

        # Map room IDs -> labels (best-effort)
    ROOM_ID2LABEL = {}
    rooms_j = (controller.current_scene_json or {}).get("rooms", [])
    for r in rooms_j:
        rid = r.get("id") or r.get("roomId") or r.get("name")
        rlab = r.get("name") or r.get("label") or r.get("room_name") or r.get("room_label")
        if rid is not None:
            ROOM_ID2LABEL[str(rid)] = rlab

    def _node_room_label(node: dict):
        # Try IDs first, then embedded labels/names; fall back to raw field
        rid = node.get("room") or node.get("roomId")
        if rid is not None:
            rid_str = str(rid)
            if rid_str in ROOM_ID2LABEL:
                return ROOM_ID2LABEL[rid_str]
        # direct label/name fields
        return node.get("room_name") or node.get("room_label") or node.get("room") or node.get("roomId")

    EXCLUDE_TYPES = {"wall", "ceiling", "floor", "door"}

    scene_graph_objs = (controller.current_scene_json or {}).get("objects", [])
    _dest_obj_ids = []
    if dest_room_id is not None:
        for obj in scene_graph_objs:
            oid = obj.get("id", "")
            if not oid:
                continue
            if (any(k in oid.lower() for k in EXCLUDE_TYPES)):
                continue
            ap = obj.get("position", {})
            o_room_lab = find_agent_room(controller.current_scene_json, ap["x"], ap["z"])
            if o_room_lab is None:
                continue
            if o_room_lab[0] == dest_room_id:
                _dest_obj_ids.append(oid)

    def _visible_pixel_count(last_event, oid: str) -> int:
        masks = getattr(last_event, "instance_masks", None)
        if masks is None or oid not in masks:
            return 0
        return int(np.sum(masks[oid]))

    def _any_dest_room_object_visible(last_event, min_start_pixels) -> bool:
        if not _dest_obj_ids:
            return False
        for oid in _dest_obj_ids:
            if _visible_pixel_count(last_event, oid) > min_start_pixels:
                return True
        return False

    # --- Saving pass 
    # Reset for saving pass
    controller.step(
        action="TeleportFull",
        position=positions[0],
        rotation=rotations[0],
        horizon=0.0,
        standing=True,
    )

    # frame bookkeeping
    frame_count = 0
    W = H = None
    distance_traveled = 0.0
    angle_turned = 0.0
    rooms_during_window = []

    # helper: save current frame
    def save_frame(idx: int):
        nonlocal frame_count, W, H
        rgb = controller.navigation_camera
        h, w = rgb.shape[:2]
        size = min(h, w)
        sx, sy = (w - size) // 2, (h - size) // 2
        rgb_sq = rgb[sy:sy+size, sx:sx+size]
        Image.fromarray(rgb_sq).save(os.path.join(rgb_dir, f"frame_{idx:04d}.png"))

        # set dims
        H, W = size, size

        # depth
        if hasattr(controller, 'navigation_depth_frame'):
            dimg = controller.navigation_depth_frame
            if dimg.ndim == 2:
                d_sq = dimg[sy:sy+size, sx:sx+size]
            else:
                d_sq = dimg[sy:sy+size, sx:sx+size, :]
            np.save(os.path.join(depth_dir, f"frame_{idx:04d}_raw.npy"), d_sq)
            dmax = float(np.max(d_sq)) if np.size(d_sq) > 0 else 0.0
            if dmax > 0:
                d_viz = (d_sq / dmax * 255).astype(np.uint8)
            else:
                d_viz = np.zeros_like(d_sq, dtype=np.uint8)
            cv2.imwrite(os.path.join(depth_dir, f"frame_{idx:04d}.png"), d_viz)

        # semantic (raw + object mask)
        if hasattr(controller, 'navigation_camera_segmentation'):
            seg = controller.navigation_camera_segmentation
            seg_sq = seg[sy:sy+size, sx:sx+size]
            Image.fromarray(seg_sq).save(os.path.join(sem_dir, f"frame_{idx:04d}.png"))
            np.save(os.path.join(sem_dir, f"frame_{idx:04d}_raw.npy"), seg_sq)

            frame_objects = {}

            # fast mask of "known" instances using color→id map
            colors = np.unique(seg_sq.reshape(-1, 3), axis=0)
            colors = {tuple(int(c) for c in color) for color in colors}
            for obj_color in colors:
                obj_id = controller.controller.last_event.color_to_object_id.get(obj_color, None)
                if obj_id is None:
                    continue
                # If we haven't processed this object yet in this frame
                if obj_id not in frame_objects:
                    # Try to match with scene objects if possible
                    if hasattr(controller, 'current_scene_json') and 'objects' in controller.current_scene_json:
                        scene_graph = controller.current_scene_json['objects']
                        obj_name = find_object_info(scene_graph, obj_id)
                        if obj_name is None:
                            obj_name = "Unknown"
                        if "Obja" in obj_name:
                            obj_name = obj_name.replace("Obja", "")
                    
                    # Store object info
                    frame_objects[obj_id] = {
                        "color": tuple(obj_color),
                        "rgb_string": f"rgb({obj_color[0]}, {obj_color[1]}, {obj_color[2]})",
                        "hex_color": f"#{obj_color[0]:02x}{obj_color[1]:02x}{obj_color[2]:02x}",
                        "name": obj_name,
                        "instance_id": obj_id
                    }
            
            # Save object metadata
            metadata = {
                "scene_metadata": {
                    "total_objects": len(frame_objects),
                    "object_categories": list(set(obj['name'] for obj in frame_objects.values()))
                },
                "objects": frame_objects
            }

            meta_path = os.path.join(sem_dir, f"frame_{idx:04d}_meta.json")
            with open(meta_path, 'w') as f:
                json.dump(metadata, f, indent=2)

        # pose
        Tcw = nav_camera_pose_from_event(controller.controller.last_event)
        np.save(os.path.join(pose_dir, f"frame_{idx:04d}_pose.npy"), Tcw)

        frame_count += 1

    # # Save initial frame only if the window starts at 0
    # if win_start == 0:
    #     save_frame(0)

    # main loop over actions; save only in window
    started = False  # begin saving only after a dest-room (non-structural) object becomes visible

    for step_idx, token in enumerate(processed_actions, 1):
        act, params = convert_action(token)
        try:
            event = controller.step(action=act, **params)
        except Exception as e:
            print(f"[DoorSegment] save step {step_idx} action {act} failed: {e}")
            continue

        # metrics (unchanged)
        if act in ("MoveAhead", "MoveBack"):
            distance_traveled += float(params.get('moveMagnitude', 0.0))
        elif act in ("RotateLeft", "RotateRight"):
            angle_turned += abs(float(params.get('degrees', 0.0)))

        if win_start <= step_idx < win_end:
            if not started:
                # New rule: start saving only when we see ANY non-structural object in the DESTINATION room
                if _any_dest_room_object_visible(controller.controller.last_event, min_start_pixels=min_start_pixels):
                    started = True
                else:
                    # not visible yet → don’t save this step, keep rolling forward
                    continue

            # from here on, save frames as usual
            ap = event.metadata["agent"]["position"]
            rid, rlab = _room_tuple_of_pos(ap["x"], ap["z"])
            rooms_during_window.append(rlab)

            save_frame(step_idx)

        if step_idx >= win_end:
            break

    # ------------------ Post-window rotation sweep (depth heuristic) ------------------
    # Use current navigation depth frame from the last event to choose rotation direction.
    # We crop to the same square as frames, split L/R, compare means (over >0 values).
    try:
        dimg = getattr(controller, 'navigation_depth_frame', None)
        if dimg is not None and extra_rot_steps > 0:
            # Square crop like save_frame
            if dimg.ndim == 2:
                h, w = dimg.shape[:2]
                size = min(h, w)
                sx, sy = (w - size) // 2, (h - size) // 2
                d_sq = dimg[sy:sy+size, sx:sx+size]
            else:
                h, w = dimg.shape[:2]
                size = min(h, w)
                sx, sy = (w - size) // 2, (h - size) // 2
                d_sq = dimg[sy:sy+size, sx:sx+size, 0]  # take first channel if any

            mid = d_sq.shape[1] // 2
            left = d_sq[:, :mid]
            right = d_sq[:, mid:]

            # means over positive depths only
            left_vals = left[left > 0]
            right_vals = right[right > 0]
            left_mean = float(left_vals.mean()) if left_vals.size > 0 else 0.0
            right_mean = float(right_vals.mean()) if right_vals.size > 0 else 0.0

            # Choose direction with larger average depth (more free space)
            # If tie, prefer right.
            rot_token = 'ls' if left_mean > right_mean else 'rs'

            # Continue index after the last action index we could have saved
            next_idx = max(win_end, step_idx if 'step_idx' in locals() else 0)

            for k in range(extra_rot_steps):
                act, params = convert_action(rot_token)  # small 6° step
                try:
                    event = controller.step(action=act, **params)
                except Exception as e:
                    print(f"[DoorSegment] extra-rot step {k+1}/{extra_rot_steps} failed: {e}")
                    break

                # keep metrics
                if act in ("RotateLeft", "RotateRight"):
                    angle_turned += abs(float(params.get('degrees', 0.0)))

                # update rooms list (not required but consistent)
                ap = event.metadata["agent"]["position"]
                rid, rlab = _room_tuple_of_pos(ap["x"], ap["z"])
                rooms_during_window.append(rlab)

                # Save a frame for each rotation step
                save_frame(next_idx + k + 1)

            # stash for metadata
            extra_rot_dir = 'left' if rot_token == 'ls' else 'right'
            extra_rot_left_mean = left_mean
            extra_rot_right_mean = right_mean
        else:
            extra_rot_dir = None
            extra_rot_left_mean = None
            extra_rot_right_mean = None

    except Exception as e:
        print(f"[DoorSegment] extra rotation heuristic failed: {e}")
        extra_rot_dir = None
        extra_rot_left_mean = None
        extra_rot_right_mean = None

    controller.stop()

    # --- Metadata
    uniq_raw = [r for r in set(rooms_during_window) if r]
    uniq_norm = list({_normalize_room_label(r) for r in uniq_raw})
    room_annotation = 'cross-room' if len(uniq_norm) > 1 else f"single-room: {uniq_norm[0] if uniq_norm else 'unknown'}"
    extra_rot_dir = None
    extra_rot_left_mean = None
    extra_rot_right_mean = None

    meta = {
        'house_id': house_id,
        'frames': int(frame_count),
        'width': int(W) if W is not None else None,
        'height': int(H) if H is not None else None,
        'window_start_step': int(win_start),
        'window_end_step': int(win_end),
        'cross_step': int(cross_step),
        'distance_traveled_m': float(distance_traveled),
        'angle_turned_deg': float(angle_turned),
        'rooms_visited_window': rooms_during_window,
        'unique_rooms_window': uniq_norm,
        'room_annotation': room_annotation,
        'door_crossing_only': True,
    }
    meta['dest_room_id'] = dest_room_id
    meta['dest_room_label'] = dest_room_label
    meta['extra_rotation_steps'] = int(extra_rot_steps)
    meta['extra_rotation_direction'] = extra_rot_dir
    meta['depth_left_mean'] = extra_rot_left_mean
    meta['depth_right_mean'] = extra_rot_right_mean

    with open(os.path.join(output_dir, "trajectory_metadata.json"), 'w') as f:
        json.dump(meta, f, indent=2)

    return {'frame_count': frame_count}

# --- Batch processing (minimal, dedicated to door-crossing) --------------------
def process_all_door_crossings(
    base_dir: str,
    output_base_dir: str,
    *,
    pre_steps: int = 20,
    post_steps: int = 40,
    start_house: str = None,
    end_house: str = None,
    houses_file: str = None,
    make_videos: bool = True,
    min_start_pixels: int = 80,
    extra_rot_steps: int = 10,
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
            print(f"[Batch] Missing HDF5 for house {house_id}, skip.")
            skipped += 1
            continue

        house_index = int(house_id)
        try:
            house_data = load_house_from_prior(house_index)
        except Exception as e:
            print(f"[Batch] Could not load house {house_id}: {e}")
            skipped += 1
            continue

        # enumerate episode indices in HDF5
        try:
            with h5py.File(hdf5_path, 'r') as f:
                episode_keys = sorted([k for k in f.keys() if k.isdigit() or (k.startswith('0') and k[1:].isdigit())],
                                      key=lambda x: int(x))
        except Exception as e:
            print(f"[Batch] Could not read episodes in {hdf5_path}: {e}")
            skipped += 1
            continue
        
        episode_keys = episode_keys[:1]

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
                            if b == 0: break
                            s += chr(b)
                        traj['actions'].append(s)

                out_dir = os.path.join(output_base_dir, f"house_{house_id}_episode_{ep}")
                os.makedirs(out_dir, exist_ok=True)

                info = replay_door_crossing_segment(
                    house_id=str(house_id),
                    house_data=house_data,
                    trajectory_data=traj,
                    output_dir=out_dir,
                    pre_steps=pre_steps,
                    post_steps=post_steps,
                    min_start_pixels=min_start_pixels,
                    extra_rot_steps=args.extra_rot_steps,
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
                print(f"[Batch] Error on house {house_id} ep {ep}: {e}")
                continue

    print(f"[Batch] Done. Episodes processed: {total_eps}, frames saved: {total_frames}, skipped: {skipped}")

# --- Single episode helper -----------------------------------------------------
def process_one_episode_door_crossing(
    house_index: int,
    episode_idx: str,
    hdf5_path: str,
    output_dir_root: str,
    *,
    pre_steps: int = 20,
    post_steps: int = 40,
    make_videos: bool = True,
    extra_rot_steps: int = 10,
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
                if b == 0: break
                s += chr(b)
            traj['actions'].append(s)

    out_dir = os.path.join(output_dir_root, f"house_{house_index}_episode_{episode_idx}")
    os.makedirs(out_dir, exist_ok=True)

    info = replay_door_crossing_segment(
        house_id=str(house_index),
        house_data=house_data,
        trajectory_data=traj,
        output_dir=out_dir,
        pre_steps=pre_steps,
        post_steps=post_steps,
        extra_rot_steps=args.extra_rot_steps,
    )

    if info.get('frame_count', 0) > 0 and make_videos:
        vids = os.path.join(out_dir, "videos")
        os.makedirs(vids, exist_ok=True)
        for mdir, vname in (("rgb", "rgb.mp4"), ("depth", "depth.mp4"), ("semantic", "semantic.mp4")):
            dpath = os.path.join(out_dir, mdir)
            if os.path.isdir(dpath) and os.listdir(dpath):
                create_video_from_frames(dpath, os.path.join(vids, vname))

    print(f"[Single] Frames saved: {info.get('frame_count', 0)} at {out_dir}")

# --- CLI ----------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build door-crossing observation segments from existing action data.")
    # Modes
    parser.add_argument('--process_all', action='store_true', help='Process all houses/episodes under base_dir.')
    # IO
    parser.add_argument('--base_dir', type=str,
                        default="/home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/spoc/data/all/ObjectNavType/val",
                        help='Base directory containing house folders (each with hdf5_sensors.hdf5).')
    parser.add_argument('--output_dir', type=str, default="data/door_crossing_segments_val",
                        help='Output base directory.')
    parser.add_argument('--houses_file', type=str, default=None,
                        help='Optional file listing house directories to process (one per line).')
    parser.add_argument('--start_house', type=str, default=None, help='Start house id (inclusive, numeric as string).')
    parser.add_argument('--end_house', type=str, default=None, help='End house id (inclusive, numeric as string).')
    # Window / detection params
    parser.add_argument('--door_pre_steps', type=int, default=20, help='Steps before crossing to keep.')
    parser.add_argument('--door_post_steps', type=int, default=40, help='Steps after crossing to keep.')
    parser.add_argument('--no_videos', action='store_true', help='Do not create MP4s.')
    # Single-episode
    parser.add_argument('--house_index', type=int, default=None, help='House index for single episode.')
    parser.add_argument('--episode_idx', type=str, default=None, help='Episode index (string) for single episode.')
    parser.add_argument('--hdf5_path', type=str, default=None, help='Path to hdf5_sensors.hdf5 for single episode.')
    parser.add_argument('--min_start_pixels', type=int, default=80, help='Min pixels of dest-room object to start saving.')
    parser.add_argument('--extra_rot_steps', type=int, default=10,
                    help='Number of small rotation steps to execute after the window based on depth heuristic.')

    args = parser.parse_args()

    if args.process_all:
        process_all_door_crossings(
            base_dir=args.base_dir,
            output_base_dir=args.output_dir,
            pre_steps=args.door_pre_steps,
            post_steps=args.door_post_steps,
            start_house=args.start_house,
            end_house=args.end_house,
            houses_file=args.houses_file,
            make_videos=not args.no_videos,
            min_start_pixels=args.min_start_pixels
        )
    else:
        if args.house_index is None or args.episode_idx is None:
            print("Single-episode mode requires --house_index, --episode_idx and --hdf5_path.")
            sys.exit(1)
        process_one_episode_door_crossing(
            house_index=args.house_index,
            episode_idx=args.episode_idx,
            hdf5_path=args.hdf5_path,
            output_dir_root=args.output_dir,
            pre_steps=args.door_pre_steps,
            post_steps=args.door_post_steps,
            make_videos=not args.no_videos,
        )
