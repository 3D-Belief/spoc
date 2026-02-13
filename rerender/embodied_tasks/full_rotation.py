import os
import json
import math
from typing import Dict, List, Tuple, Optional

import numpy as np
from PIL import Image
import cv2

from scipy.spatial.transform import Rotation as R
from spoc_utils.embodied_utils import find_object_node
from rerender.registry import register_embodied_task, register_constant

OBJECT_EXCLUSION_LIST = ['wall', 'floor', 'ceiling', 'window', 'door', 'room']

register_constant("full_rotation_object_exclusion_list", OBJECT_EXCLUSION_LIST, overwrite=True)

# ---------- helpers shared with the main script style ----------

def _nav_camera_pose(event) -> np.ndarray:
    """Return cam->world 4x4 for the navigation camera (right-handed)."""
    pos = event.metadata["cameraPosition"]            # dict x,y,z (world)
    agent = event.metadata["agent"]
    yaw = agent["rotation"]["y"]                      # degrees (around Y)
    pitch = agent["cameraHorizon"]
    R_aw = R.from_euler('xyz', [0.0, yaw, 0.0], degrees=True).as_matrix()
    R_ca = R.from_euler('xyz', [pitch, 0.0, 0.0], degrees=True).as_matrix()
    R_cw = R_aw @ R_ca
    t_cw = np.array([pos["x"], pos["y"], pos["z"]], dtype=float)
    extrinsic_cam2world = np.eye(4, dtype=float)
    extrinsic_cam2world[:3, :3] = R_cw
    extrinsic_cam2world[:3, 3] = t_cw
    # left-handed to right-handed flip
    F = np.diag([1, 1, -1, 1])
    extrinsic_cam2world = F @ extrinsic_cam2world @ F
    return extrinsic_cam2world


def _square_crop(img_np: np.ndarray) -> np.ndarray:
    h, w = img_np.shape[:2]
    size = min(h, w)
    sx = (w - size) // 2
    sy = (h - size) // 2
    return img_np[sy:sy+size, sx:sx+size]


def _save_modalities(controller, out_dirs: Dict[str, str], step_idx: int):
    """Save RGB / Depth / Semantic (color-coded) / Pose, square-cropped, matching the rerender script."""
    # RGB
    rgb = controller.navigation_camera
    rgb_square = _square_crop(rgb)
    Image.fromarray(rgb_square).save(os.path.join(out_dirs["rgb"], f"frame_{step_idx:04d}.png"))

    # Depth → mm → 16bit PNG
    if hasattr(controller, "navigation_depth_frame"):
        depth = controller.navigation_depth_frame
        if depth.ndim == 3:
            depth = depth[..., 0]
        depth = depth.astype(np.float32)
        depth_sq = _square_crop(depth)
        invalid = ~np.isfinite(depth_sq) | (depth_sq < 0)
        depth_u16 = np.round(depth_sq * 1000.0).astype(np.int64)
        depth_u16[invalid] = 0
        depth_u16 = np.clip(depth_u16, 0, 65535).astype(np.uint16)
        cv2.imwrite(os.path.join(out_dirs["depth"], f"frame_{step_idx:04d}.png"), depth_u16)

    # Semantic (instance color image) + simple metadata + binary-objects mask
    if hasattr(controller, "navigation_camera_segmentation"):
        seg = controller.navigation_camera_segmentation
        seg_sq = _square_crop(seg)
        Image.fromarray(seg_sq).save(os.path.join(out_dirs["semantic"], f"frame_{step_idx:04d}.png"))

        # Build simple per-frame object color list
        colors = np.unique(seg_sq.reshape(-1, 3), axis=0)
        colors = [tuple(int(c) for c in row) for row in colors]
        frame_meta = {
            "total_colors": len(colors),
            "colors_rgb": [list(c) for c in colors],
        }
        with open(os.path.join(out_dirs["semantic"], f"frame_{step_idx:04d}_meta.json"), "w") as f:
            json.dump(frame_meta, f, indent=2)

        # Simple “object mask” = any non-background color (background assumed near black)
        seg_codes = (
            seg_sq[..., 0].astype(np.uint32) << 16
            | seg_sq[..., 1].astype(np.uint32) << 8
            | seg_sq[..., 2].astype(np.uint32)
        )
        mask_bool = seg_codes != 0  # heuristic
        Image.fromarray((mask_bool.astype(np.uint8) * 255)).save(
            os.path.join(out_dirs["semantic"], f"frame_{step_idx:04d}_object_binary_mask.png")
        )

    # Pose (4x4)
    pose = _nav_camera_pose(controller.controller.last_event)
    np.save(os.path.join(out_dirs["pose"], f"frame_{step_idx:04d}_pose.npy"), pose)

def _visible_pixel_count(last_event, oid: str) -> int:
    masks = last_event.instance_masks
    if masks is None or oid not in masks:
        return 0
    return int(np.sum(masks[oid]))


def _yaw_facing(from_pos: Dict[str, float], to_world_pos: Dict[str, float]) -> float:
    dx = to_world_pos["x"] - from_pos["x"]
    dz = to_world_pos["z"] - from_pos["z"]
    # atan2(dx, dz) to match AI2-THOR yaw convention
    return float((math.degrees(math.atan2(dx, dz)) % 360.0))


def generate_full_rotation_trajectory(
    controller,
    house_id: str,
    object_id: str,
    near_radius_m: Tuple[float, float] = (1.0, 3.0),
    rotate_step_deg: int = 3,
    rotate_direction: str = "left",   # or "right"
    total_degrees: float = 360.0,
    max_search_teleports: int = 100,
    saver: Optional[callable] = None,
) -> Dict:
    """
    For a given object, pick a nearby position, face the object once,
    then rotate a full 360° from that pose.

    No visibility metric is tracked; instead we record:
      - positions
      - rotations
      - relative_yaw_degrees (angle w.r.t. starting yaw, in [0, 360))
      - actions ("ls"/"rs" per step)

    Frames are saved in lockstep via the `saver(step_idx)` callback.
    """

    # 1) Reachable positions around the current house
    reachable = controller.step(action="GetReachablePositions").metadata["actionReturn"]
    if not reachable:
        return {}

    # Find object world position from current scene graph
    scene_graph = controller.current_scene_json.get("objects", [])
    node = find_object_node(scene_graph, object_id)
    obj_worldpos = node.get("position", None) if node else None
    if obj_worldpos is None:
        return {}

    # Build candidate positions within [rmin, rmax]
    rmin, rmax = near_radius_m
    cand_positions = []
    for p in reachable:
        dx, dz = (p["x"] - obj_worldpos['x']), (p["z"] - obj_worldpos['z'])
        dist = math.hypot(dx, dz)
        if rmin <= dist <= rmax:
            cand_positions.append(p)

    if not cand_positions:
        # No good nearby positions; skip this object
        return {}

    # 2) Choose a candidate position (random among nearby ones) and face the object
    pos = np.random.choice(cand_positions)
    base_yaw = _yaw_facing(pos, obj_worldpos)

    controller.step(
        action="TeleportFull",
        position=pos,
        rotation={"x": 0.0, "y": base_yaw, "z": 0.0},
        horizon=0.0,
        standing=True,
    )
    ev = controller.controller.last_event
    if not ev.metadata.get("lastActionSuccess", True):
        return {}

    # Starting pose after teleport
    agent_rot = ev.metadata["agent"]["rotation"]
    start_yaw = float(agent_rot["y"])

    positions: List[Dict[str, float]] = []
    rotations: List[Dict[str, float]] = []
    actions: List[str] = []               # 'ls' or 'rs'
    rel_yaw_degrees: List[float] = []     # relative yaw in [0, 360)

    # Choose rotation token and sign / action
    if rotate_direction == "left":
        token = "ls"
        rotate_action = "RotateLeft"
        step_sign = +1
    else:
        token = "rs"
        rotate_action = "RotateRight"
        step_sign = -1

    step_deg = abs(int(rotate_step_deg))
    if step_deg <= 0:
        raise ValueError("rotate_step_deg must be non-zero")

    # How many steps to approximately cover total_degrees
    n_steps = int(round(float(total_degrees) / float(step_deg)))

    def record_step():
        ev_local = controller.controller.last_event
        ag = ev_local.metadata["agent"]

        pos_local = ag["position"]
        rot_local = ag["rotation"]

        positions.append({"x": pos_local["x"], "y": pos_local["y"], "z": pos_local["z"]})
        rotations.append({"x": rot_local["x"], "y": rot_local["y"], "z": rot_local["z"]})

        # Relative yaw (wrap into [0, 360))
        curr_yaw = float(rot_local["y"])
        rel = (curr_yaw - start_yaw) * step_sign  # positive in chosen direction
        rel = rel % 360.0
        rel_yaw_degrees.append(float(rel))

        if saver is not None:
            saver(len(positions) - 1)

    # Initial frame (angle = 0)
    record_step()

    # Perform rotation steps
    for _ in range(n_steps):
        ev = controller.step(action=rotate_action, degrees=step_deg)
        if not ev.metadata.get("lastActionSuccess", True):
            break

        actions.append(token)
        record_step()

    return {
        "house_id": house_id,
        "object_id": object_id,
        "positions": positions,
        "rotations": rotations,
        "actions": actions,                      # length = frames - 1
        "relative_yaw_degrees": rel_yaw_degrees, # length = frames
        "total_degrees_target": float(total_degrees),
        "rotate_direction": rotate_direction,
        "deg_step": step_deg,
    }

# ---------- main task entry ----------

@register_embodied_task(name="full_rotation")
def full_rotation_episode(
    controller,
    house_id: str,
    object_id: str,
    out_dir: str,
    rotate_dir: str = "left",
    deg_step: int = 3,
    total_deg: float = 360.0,
) -> Optional[Dict]:
    """
    Full-rotation task: for a given object, move to a nearby position, face it once,
    then rotate ~360° in the chosen direction.

    No visibility label; supervision is purely the rotation angle.
    """

    frame_info = {"frame_count": 0}

    def saver_cb(step_idx: int):
        # step_idx matches positions index (0-based); write modalities in lockstep
        save_step_modalities_and_meta(controller, out_dir, step_idx, frame_info, record_modalities=True)

    traj = generate_full_rotation_trajectory(
        controller=controller,
        house_id=house_id,
        object_id=object_id,
        near_radius_m=(1.0, 3.0),
        rotate_step_deg=deg_step,
        rotate_direction=rotate_dir,
        total_degrees=total_deg,
        saver=saver_cb,
    )

    if not traj or not traj.get("positions"):
        return None

    # Save metadata (trajectory + frame stats)
    meta = {
        "house_id": traj["house_id"],
        "object_id": traj["object_id"],
        "frames": int(frame_info["frame_count"]),
        "width": int(frame_info["width"]),
        "height": int(frame_info["height"]),
        "positions": traj["positions"],
        "rotations": traj["rotations"],
        "actions": traj["actions"],
        "relative_yaw_degrees": [float(a) for a in traj["relative_yaw_degrees"]],
        "task_type": "full-rotation",
        "rotate_direction": traj["rotate_direction"],
        "deg_step": traj["deg_step"],
        "total_degrees_target": traj["total_degrees_target"],
    }

    with open(os.path.join(out_dir, "trajectory_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    return meta

def save_step_modalities_and_meta(
    controller,
    out_dir: str,
    step_idx: int,
    frame_info: Dict[str, int],
    record_modalities: bool = True,
):
    """Create dirs if needed, save RGB/Depth/Semantic/Pose like the rerender script."""
    os.makedirs(out_dir, exist_ok=True)
    rgb = os.path.join(out_dir, "rgb")
    depth = os.path.join(out_dir, "depth")
    semantic = os.path.join(out_dir, "semantic")
    pose = os.path.join(out_dir, "pose")
    for d in [rgb, depth, semantic, pose]:
        os.makedirs(d, exist_ok=True)

    out_dirs = {"rgb": rgb, "depth": depth, "semantic": semantic, "pose": pose}

    if record_modalities:
        _save_modalities(controller, out_dirs, step_idx)

    if frame_info.get("frame_count", 0) == 0:
        frame_info["width"] = controller.navigation_camera.shape[1]
        frame_info["height"] = controller.navigation_camera.shape[0]
    frame_info["frame_count"] = frame_info.get("frame_count", 0) + 1