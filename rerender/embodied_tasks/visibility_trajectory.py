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

register_constant("visibility_trajectory_object_exclusion_list", OBJECT_EXCLUSION_LIST, overwrite=True)

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


# ---------- main task entry ----------

def generate_visibility_trajectory_for_object(
    controller,
    house_id: str,
    object_id: str,
    near_radius_m: Tuple[float, float] = (1.0, 3.0),
    coarse_face_sweep_deg: int = 30,
    fine_face_sweep_deg: int = 5,
    rotate_step_deg: int = 3,
    rotate_direction: str = "left",   # or "right"
    min_start_pixels: int = 80,
    max_search_teleports: int = 100,
    ensure_border_margin: int = 2,
    saver: Optional[callable] = None,
) -> Dict:
    """
    For a given object, pick a nearby position, face the object, ensure strong visibility,
    then rotate in one direction in small steps until invisible. Save per-step visibility%
    and (optionally) call `saver(step_idx)` to write modalities like the rerenderer.

    Returns a dict with trajectory fields (positions, rotations, actions, visibility_percent).
    """

    # 1) Reachable positions (BUGFIX: use keyword arg for 'action')
    reachable = controller.step(action="GetReachablePositions").metadata["actionReturn"]
    if not reachable:
        return {}
    # Keep a current event for masks, etc.
    event = controller.controller.last_event

    # Find object world position from current event
    scene_graph = controller.current_scene_json['objects']
    node = find_object_node(scene_graph, object_id)
    obj_worldpos = node.get("position", None) if node else None

    if obj_worldpos is None:
        return {}

    # Build candidate positions within given [rmin, rmax]
    rmin, rmax = near_radius_m
    cand_positions = []
    for p in reachable:
        dx, dz = (p["x"] - obj_worldpos['x']), (p["z"] - obj_worldpos['z'])
        dist = math.hypot(dx, dz)
        if rmin <= dist <= rmax:
            cand_positions.append(p)
    if not cand_positions:
        return {}

    # 2) Try candidates: face object and ensure strong visibility (find best yaw around facing)
    best = None  # (visible_pixels, pos, yaw)
    trials = 0
    for pos in np.random.permutation(cand_positions):
        if trials >= max_search_teleports:
            break
        trials += 1

        base_yaw = _yaw_facing(pos, obj_worldpos)
        best_local_pixels, best_local_yaw = -1, base_yaw
        for delta in range(-coarse_face_sweep_deg, coarse_face_sweep_deg + 1, fine_face_sweep_deg):
            yaw_try = (base_yaw + delta) % 360.0
            controller.step(
                action="TeleportFull",
                position=pos,
                rotation={"x": 0.0, "y": yaw_try, "z": 0.0},
                horizon=0.0,
                standing=True,
            )
            ev = controller.controller.last_event
            if not ev.metadata.get("lastActionSuccess", True):
                continue
            pix = _visible_pixel_count(ev, object_id)
            if pix > best_local_pixels:
                best_local_pixels, best_local_yaw = pix, yaw_try

        if best_local_pixels >= min_start_pixels:
            best = (best_local_pixels, pos, best_local_yaw)
            break

    if best is None:
        return {}

    start_pixels, start_pos, start_yaw = best

    # Go to start pose and validate mask
    controller.step(
        action="TeleportFull",
        position=start_pos,
        rotation={"x": 0.0, "y": start_yaw, "z": 0.0},
        horizon=0.0,
        standing=True,
    )
    ev = controller.controller.last_event
    if not ev.metadata.get("lastActionSuccess", True):
        return {}

    masks = ev.instance_masks
    if masks is None or object_id not in masks:
        return {}

    # 3) Rotate in small steps until invisible — RECORD & SAVE EACH STEP
    positions: List[Dict[str, float]] = []
    rotations: List[Dict[str, float]] = []
    actions: List[str] = []  # 'ls' or 'rs'
    vis_percent: List[float] = []

    def record_step():
        nonlocal start_pixels

        # pose
        ev = controller.controller.last_event
        agent_pos = ev.metadata["agent"]["position"]
        agent_rot = ev.metadata["agent"]["rotation"]
        positions.append({"x": agent_pos["x"], "y": agent_pos["y"], "z": agent_pos["z"]})
        rotations.append({"x": agent_rot["x"], "y": agent_rot["y"], "z": agent_rot["z"]})

        vis = 0.0  # default

        if hasattr(controller, "navigation_camera_segmentation"):
            seg = controller.navigation_camera_segmentation  # HxWx3 (uint8)
            H, W = seg.shape[:2]
            size = min(H, W)
            sx = (W - size) // 2
            sy = (H - size) // 2
            seg_sq = seg[sy:sy+size, sx:sx+size]

            # Build object mask in the cropped square
            color_map = getattr(ev, "color_to_object_id", None)
            obj_mask_sq = None
            if isinstance(color_map, dict):
                obj_colors = [c for c, oid in color_map.items() if oid == object_id]
                if obj_colors:
                    obj_mask_sq = np.zeros((size, size), dtype=bool)
                    for c in obj_colors:
                        obj_mask_sq |= np.all(seg_sq == c, axis=-1)

            # Fallback to instance_masks (then crop)
            if obj_mask_sq is None:
                masks = ev.instance_masks
                if masks is not None and object_id in masks:
                    full_mask = masks[object_id].astype(bool)
                    obj_mask_sq = full_mask[sy:sy+size, sx:sx+size]
                else:
                    obj_mask_sq = np.zeros((size, size), dtype=bool)

            obj_pixels = int(np.count_nonzero(obj_mask_sq))

            if obj_pixels == 0:
                # Truly invisible in the cropped frame
                vis = 0.0
            else:
                # Check only left/right borders (with margin)
                m = max(int(ensure_border_margin), 1)
                m = min(m, size)
                touch_left  = bool(np.any(obj_mask_sq[:, :m]))
                touch_right = bool(np.any(obj_mask_sq[:, -m:]))
                border_touch = (touch_left or touch_right)

                if not border_touch:
                    # Fully visible (no LR border contact)
                    start_pixels = max(start_pixels, obj_pixels)
                    vis = 1.0
                else:
                    # Border-touching: use ratio against baseline
                    if start_pixels <= 0:
                        # initialize baseline if needed
                        start_pixels = obj_pixels
                        vis = 1.0
                    else:
                        vis = obj_pixels / float(start_pixels)
        else:
            # No segmentation available: fall back to global instance mask count (uncropped)
            pix = _visible_pixel_count(ev, object_id)
            if pix <= 0:
                vis = 0.0
            else:
                if start_pixels <= 0:
                    start_pixels = pix
                    vis = 1.0
                else:
                    vis = pix / float(start_pixels)

        vis_percent.append(max(0.0, min(1.0, float(vis))))

        # trigger external saver with the current step index
        if saver is not None:
            saver(len(positions) - 1)

    # initial frame (index 0)
    record_step()

    # Choose rotation token and sign
    if rotate_direction == "left":
        token, sign = "ls", +1
        rotate_action = "RotateLeft"
    else:
        token, sign = "rs", -1
        rotate_action = "RotateRight"

    # keep stepping until fully invisible (no mask)
    while True:
        # perform rotation step (BUGFIX: use keyword arg for 'action')
        ev = controller.step(action=rotate_action, degrees=abs(rotate_step_deg))
        if not ev.metadata.get("lastActionSuccess", True):
            break

        actions.append(token)
        record_step()

        if _visible_pixel_count(controller.controller.last_event, object_id) < 1:
            break

    return {
        "house_id": house_id,
        "object_id": object_id,
        "positions": positions,
        "rotations": rotations,
        "actions": actions,               # length = frames-1
        "visibility_percent": vis_percent # length = frames
    }

@register_embodied_task(name="visibility_trajectory")
def visibility_episode_for_object(
    controller,
    house_id: str,
    object_id: str,
    out_dir: str,
    rotate_dir: str = "left",
    deg_step: int = 3,
) -> Optional[Dict]:
    """
    Builds the visibility trajectory for a single object and writes per-step modalities & metadata.
    Now we save frames INSIDE the task as it rotates, via a saver callback, so states match frames 1:1.
    """
    frame_info = {"frame_count": 0}

    def saver_cb(step_idx: int):
        # step_idx matches positions index (0-based); write modalites in lockstep
        save_step_modalities_and_meta(controller, out_dir, step_idx, frame_info, record_modalities=True)

    traj = generate_visibility_trajectory_for_object(
        controller=controller,
        house_id=house_id,
        object_id=object_id,
        rotate_direction=rotate_dir,
        rotate_step_deg=deg_step,
        saver=saver_cb,   # <<< frames saved per-step here
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
        "visibility_percent": [float(v) for v in traj["visibility_percent"]],
        "task_type": "visibility-rotation-trajectory",
        "rotate_direction": rotate_dir,
        "deg_step": deg_step,
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