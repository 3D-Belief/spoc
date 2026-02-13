from typing import List, Union
from pathlib import Path
import json
import os
import re
from PIL import Image, ImageDraw, ImageFont
import random
from collections import defaultdict
import numpy as np
import math
import cv2
from shapely.geometry import Polygon
import itertools
from contextlib import suppress
import subprocess
import tempfile
from pathlib import Path
from PIL import Image
import math

from ai2thor.controller import Controller
# from ai2thor.util.metrics import get_shortest_path_to_object_type  # noqa: F401

def preview_html(qa_json, out_dir, html_out, preview_width=200):
    if not qa_json.exists():
        print("No QA JSON to preview.")
        return
    with open(qa_json, "r") as f:
        all_im_qas = json.load(f)

    html = [
        "<html><head><meta charset='utf-8'>"
        "<style>pre{background:#f6f6f6;padding:8px;border-radius:6px;display:inline-block;white-space:pre}</style>"
        "</head><body>"
    ]
    html.append(f"<p>Num samples: {len(all_im_qas)}</p>")

    # Separate questions by type
    relative_questions = [item for item in all_im_qas if item["qa"].get("type") == "relative-direction"]
    occlusion_questions = [item for item in all_im_qas if item["qa"].get("type") == "egocentric-occlusion"]
    egocentric_distance_questions = [item for item in all_im_qas if item["qa"].get("type") == "egocentric-distance"]
    egocentric_direction_questions = [item for item in all_im_qas if item["qa"].get("type") == "egocentric-direction"]
    relative_distance_questions = [item for item in all_im_qas if item["qa"].get("type") == "relative-distance"]
    relative_distance_no_imagine_questions = [item for item in all_im_qas if item["qa"].get("type") == "relative-distance-no-imagine"]
    egocentric_direction_no_imagine_questions = [item for item in all_im_qas if item["qa"].get("type") == "egocentric-direction-no-imagine"]
    egocentric_occlusion_no_imagine_questions = [item for item in all_im_qas if item["qa"].get("type") == "egocentric-occlusion-no-imagine"]
    relative_direction_no_imagine_questions = [item for item in all_im_qas if item["qa"].get("type") == "relative-direction-no-imagine"]

    # Sample 20 questions from each type
    items_to_show = []
    
    # Add up to 20 samples from each question type
    for question_type, questions in [
        ("relative-direction", relative_questions),
        ("egocentric-occlusion", occlusion_questions), 
        ("egocentric-distance", egocentric_distance_questions),
        ("egocentric-direction", egocentric_direction_questions),
        ("relative-distance", relative_distance_questions),
        ("relative-distance-no-imagine", relative_distance_no_imagine_questions),
        ("egocentric-direction-no-imagine", egocentric_direction_no_imagine_questions),
        ("egocentric-occlusion-no-imagine", egocentric_occlusion_no_imagine_questions),
        ("relative-direction-no-imagine", relative_direction_no_imagine_questions),
    ]:
        sample_size = min(20, len(questions))
        if sample_size > 0:
            items_to_show.extend(random.sample(questions, sample_size))

    # html.append(f"<p>Showing all {len(relative_questions)} relative-direction questions, all {len(occlusion_questions)} occlusion questions, plus {len(items_to_show) - len(relative_questions) - len(occlusion_questions)} other questions out of {len(all_im_qas)} total</p>")

    for item in items_to_show:
        question = item["qa"].get("question", "")
        images = item["qa"].get("images", [])
        choices = item["qa"].get("choices", [])
        answer = item["qa"].get("answer", "")
        qtype = item["qa"].get("type", "")
        html.append(f"<p><b>[{qtype}]</b> {question}</p>")

        for im in images:
            if not im:
                continue
            rel_path = Path(im)
            if rel_path.exists():
                partial_path = rel_path.relative_to(out_dir)
                html.append(
                    f"<img src='{partial_path.as_posix()}' "
                    f"style='width:{preview_width}px;height:{preview_width}px;margin-right:6px;'>"
                )

        html.append("<div>")
        for ans in choices:
            html.append(f"<p>{ans}</p>")
        html.append(f"<p><b>Answer:</b> {answer}</p>")

        if "imagination_paths" in item:
            html.append("<div><p><b>Target Pose Representations:</b></p>")
            for key in ["rgb", "ego3d_bbox", "topdown_bbox"]:
                block = item["imagination_paths"].get(key, {})
                for tkey in ["t", "t+1"]:
                    p = block.get(tkey)
                    if isinstance(p, str) and p and Path(p).exists():
                        pp = Path(p).relative_to(out_dir)
                        html.append(
                            "<div style='display:inline-block;margin-right:6px;text-align:center'>"
                            f"<img src='{pp.as_posix()}' style='width:{preview_width}px;height:{preview_width}px;'>"
                            f"<div>{key} ({tkey})</div></div>"
                        )
            
            # Add text grid representation
            grid_text_block = item["imagination_paths"].get("grid_text", {})
            for tkey in ["t", "t+1"]:
                tstr = grid_text_block.get(tkey, "")
                if tstr:
                    html.append(
                        "<div style='display:inline-block;vertical-align:top;margin-left:8px;text-align:left'>"
                        f"<pre>{tstr}</pre><div style='text-align:center'>grid_text ({tkey})</div></div>"
                    )
            html.append("</div>")

        if "wm_per_choice" in item:
            html.append("<div><p><b>Per-Choice Representations:</b></p>")
            for block in item["wm_per_choice"]:
                html.append(f"<p><i>{block['choice']}</i></p>")
                reps = block.get("imagination_paths", {})
                for key in ["rgb", "ego3d_bbox", "topdown_bbox"]:
                    sub = reps.get(key, {})
                    for tkey in ["t", "t+1"]:
                        p = sub.get(tkey)
                        if isinstance(p, str) and p and Path(p).exists():
                            pp = Path(p).relative_to(out_dir)
                            html.append(
                                "<div style='display:inline-block;margin-right:6px;text-align:center'>"
                                f"<img src='{pp.as_posix()}' style='width:{preview_width}px;height:{preview_width}px;'>"
                                f"<div>{key} ({tkey})</div></div>"
                            )
                
                grid_text_block = reps.get("grid_text", {})
                for tkey in ["t", "t+1"]:
                    tstr = grid_text_block.get(tkey, "")
                    if tstr:
                        html.append(
                            "<div style='display:inline-block;vertical-align:top;margin-left:8px;text-align:left'>"
                            f"<pre>{tstr}</pre><div style='text-align:center'>grid_text ({tkey})</div></div>"
                        )
            html.append("</div>")

        html.append("<hr>")

    html.append("</body></html>")
    html_out.write_text("\n".join(html), encoding="utf-8")
    print(f"Wrote preview HTML to: {html_out.resolve()}")

def print_stats(qa_json: Path):
    if not qa_json.exists():
        print("No QA JSON for stats.")
        return
    with open(qa_json, "r") as f:
        all_im_qas = json.load(f)
    type_counts = {}
    for item in all_im_qas:
        category = item["qa"].get("category", "")
        qtype = item["qa"].get("type", "other")
        if qtype not in type_counts:
            type_counts[qtype] = {}
        if category not in type_counts[qtype]:
            type_counts[qtype][category] = 0
        type_counts[qtype][category] += 1
    print("Total number of questions:", len(all_im_qas))
    print("Type counts:", dict(type_counts))

def try_load_font(size: int = 15):
    try:
        return ImageFont.truetype("LiberationSans-Bold.ttf", size)
    except Exception:
        try:
            return ImageFont.truetype("arial.ttf", size)
        except Exception:
            return ImageFont.load_default()

def camel_to_words(s: str) -> str:
    return re.sub(r'(?<!^)(?=[A-Z])', ' ', s).lower()

def add_red_dot_with_text(image: Image.Image, position: tuple[int, int], text: str) -> Image.Image:
    font = try_load_font(15)
    draw = ImageDraw.Draw(image)
    x, y = position
    r = 10
    draw.ellipse((x - r, y - r, x + r, y + r), fill="red", outline="red")
    tw = draw.textlength(text, font=font)
    draw.text((x - tw / 2, y - 7), text, fill="white", font=font)
    return image

def build_assetid2desc_from_scene(controller: Controller) -> dict[str, str]:
    assetid2desc: dict[str, str] = {}
    for obj in controller.last_event.metadata.get("objects", []):
        asset_id = obj.get("assetId")
        if not asset_id:
            continue
        obj_type = (obj.get("objectType") or "").strip().lower()
        name = (obj.get("name") or "").strip().lower()
        desc_candidates = [x for x in [obj_type, name] if x]
        desc = random.choice(desc_candidates) if desc_candidates else "object"
        assetid2desc.setdefault(asset_id, desc)
    return assetid2desc

def get_current_state(controller: Controller,
                      objid2assetid: dict[str, str],
                      assetid2desc: dict[str, str]):
    nav_visible_objects = controller.step("GetVisibleObjects", maxDistance=5).metadata["actionReturn"]
    nav_visible_objects = [o for o in nav_visible_objects if objid2assetid.get(o, "")]
    bboxes = controller.last_event.instance_detections2D
    vis_obj_to_size = {oid: (bb[2] - bb[0]) * (bb[3] - bb[1]) for oid, bb in bboxes.items()}
    objid2info = {}
    objdesc2cnt = defaultdict(int)
    for entry in controller.last_event.metadata["objects"]:
        obj_id = entry["objectId"]
        obj_type = entry["objectType"]
        asset_id = entry.get("assetId", "")
        distance = entry.get("distance", float("inf"))
        pos = np.array([entry["position"]["x"], entry["position"]["y"], entry["position"]["z"]])
        rotation = entry.get("rotation", {})
        desc = assetid2desc.get(asset_id, obj_type)
        moveable = bool(entry.get("moveable") or entry.get("pickupable"))
        bb = bboxes.get(obj_id)
        size_xy = vis_obj_to_size.get(obj_id, 0)
        pos_xy = None
        if bb is not None:
            pos_xy = [(bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2]
        parent = entry.get("parentReceptacles")
        if parent:
            parent = parent[-1]
            parent = "Floor" if parent == "Floor" else objid2assetid.get(parent, parent)
        is_receptacle = bool(entry.get("receptacle"))
        aabb = entry.get("axisAlignedBoundingBox", None)
        if aabb is not None:
            obj_vol = aabb["size"]['x'] * aabb["size"]['y'] * aabb["size"]['z']
        else:
            obj_vol = 0
        objid2info[obj_id] = {
            "entry_name": entry["name"],
            "object_type": obj_type,
            "distance": distance,
            "position": pos,
            "rotation": rotation,
            "description": desc,
            "moveable": moveable,
            "parent": parent,
            "size_xy": size_xy,
            "is_receptacle": is_receptacle,
            "position_xy": pos_xy,
            "obj_vol": obj_vol,
            "aabb": aabb
        }
        objdesc2cnt[obj_type] += 1
    moveable_visible_objs = [oid for oid in nav_visible_objects if objid2info[oid]["moveable"] and objid2info[oid]["size_xy"] > 1600]
    return nav_visible_objects, objid2info, objdesc2cnt, moveable_visible_objs

def basis_from_yaw_pitch(yaw_deg: float, pitch_deg: float):
    y = math.radians(yaw_deg)
    x = math.radians(pitch_deg)
    fwd = np.array([math.sin(y) * math.cos(x), -math.sin(x), math.cos(y) * math.cos(x)], dtype=np.float32)
    up_world = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    right = np.cross(up_world, fwd)
    right = right / max(1e-6, np.linalg.norm(right))
    up = np.cross(fwd, right)
    up = up / max(1e-6, np.linalg.norm(up))
    return right, up, fwd

def project_points(world_pts: np.ndarray, cam_pos: dict, yaw_deg: float, pitch_deg: float, fov_v_deg: float, W: int, H: int):
    right, up, fwd = basis_from_yaw_pitch(yaw_deg, pitch_deg)
    d = world_pts - np.array([cam_pos["x"], cam_pos["y"], cam_pos["z"]], dtype=np.float32)
    x_cam = d @ right
    y_cam = d @ up
    z_cam = d @ fwd
    fy = (H / 2.0) / math.tan(math.radians(fov_v_deg) / 2.0)
    fx = fy * (W / H)
    cx, cy = W / 2.0, H / 2.0
    mask = z_cam > 1e-4
    u = fx * (x_cam / z_cam) + cx
    v = fy * (-y_cam / z_cam) + cy
    return np.stack([u, v, mask.astype(np.uint8)], axis=1)

def draw_ego3d_wireframe(img_size, colors, controller: Controller, obj_infos: list[dict], out_path: Path):
    base = Image.new("RGB", (img_size, img_size), "white")
    overlay = Image.new("RGBA", (img_size, img_size), (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)
    meta = controller.last_event.metadata
    agent = meta["agent"]
    cam_pos = meta.get("cameraPosition") or agent.get("cameraPosition") or agent["position"]
    yaw = agent.get("rotation", {}).get("y", 0.0)
    pitch = agent.get("cameraHorizon", 0.0)
    fov = meta.get("cameraFieldOfView", 90.0)

    for obj_idx, info in enumerate(obj_infos):
        aabb = info.get("aabb")
        if not aabb:
            continue

        color = colors[obj_idx % len(colors)]
        c = aabb["center"]; s = aabb["size"]
        hx, hy, hz = s["x"] / 2.0, s["y"] / 2.0, s["z"] / 2.0
        corners = np.array([
            [c["x"] - hx, c["y"] - hy, c["z"] - hz],
            [c["x"] + hx, c["y"] - hy, c["z"] - hz],
            [c["x"] + hx, c["y"] + hy, c["z"] - hz],
            [c["x"] - hx, c["y"] + hy, c["z"] - hz],
            [c["x"] - hx, c["y"] - hy, c["z"] + hz],
            [c["x"] + hx, c["y"] - hy, c["z"] + hz],
            [c["x"] + hx, c["y"] + hy, c["z"] + hz],
            [c["x"] - hx, c["y"] + hy, c["z"] + hz],
        ], dtype=np.float32)
        proj = project_points(corners, cam_pos, yaw, pitch, fov, img_size, img_size)
        edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
        for a, b in edges:
            if proj[a, 2] and proj[b, 2]:
                draw.line((proj[a,0], proj[a,1], proj[b,0], proj[b,1]), fill=color, width=3)
    Image.alpha_composite(base.convert("RGBA"), overlay).convert("RGB").save(out_path)

def world_rect_corners_xz(aabb: dict) -> np.ndarray:
    c = aabb["center"]; s = aabb["size"]
    hx, hz = s["x"] / 2.0, s["z"] / 2.0
    return np.array([
        [c["x"] - hx, c["z"] - hz],
        [c["x"] + hx, c["z"] - hz],
        [c["x"] + hx, c["z"] + hz],
        [c["x"] - hx, c["z"] + hz],
    ], dtype=np.float32)

def world_to_local_xz(points_xz: np.ndarray, me_pos: dict, yaw_deg: float) -> np.ndarray:
    rel = points_xz - np.array([me_pos["x"], me_pos["z"]], dtype=np.float32)
    y = math.radians(yaw_deg)
    right = np.array([math.cos(y), -math.sin(y)], dtype=np.float32)
    fwd = np.array([math.sin(y),  math.cos(y)], dtype=np.float32)
    x_local = rel @ right
    z_local = rel @ fwd
    return np.stack([x_local, z_local], axis=1)

def axis_aligned_bbox_xz(points_xz: np.ndarray) -> tuple[float, float, float, float]:
    minx, minz = float(points_xz[:, 0].min()), float(points_xz[:, 1].min())
    maxx, maxz = float(points_xz[:, 0].max()), float(points_xz[:, 1].max())
    return minx, minz, maxx, maxz

def compute_topdown_local(objid2info: dict, me_pos: dict, me_yaw: float):
    local_bboxes = {}
    for oid, info in objid2info.items():
        aabb = info.get("aabb")
        if not aabb:
            continue
        corners_w = world_rect_corners_xz(aabb)
        corners_l = world_to_local_xz(corners_w, me_pos, me_yaw)
        minx, minz, maxx, maxz = axis_aligned_bbox_xz(corners_l)
        cx, cz = (minx + maxx) / 2.0, (minz + maxz) / 2.0
        local_bboxes[oid] = {"minx": minx, "maxx": maxx, "minz": minz, "maxz": maxz, "cx": cx, "cz": cz}
    return local_bboxes

def pick_r(local_bboxes: dict, target_oid: list[str] | None, r_default: float = 3.0):
    r = r_default
    if target_oid:
        max_ext = 0
        for oid in target_oid:
            if oid in local_bboxes:
                bb = local_bboxes[oid]
                ext = max(abs(bb["minx"]), abs(bb["maxx"]), abs(bb["minz"]), abs(bb["maxz"]))
                max_ext = max(max_ext, ext)
        if max_ext > r_default:
            r = float(max_ext * 1.05)
    return r

def draw_topdown_local(img_size, colors, me_pos_before: dict, me_yaw_before: float, objid2info_before: dict, out_path_before: Path, me_pos_after: dict, me_yaw_after: float, objid2info_after: dict, out_path_after: Path, target_oid: list[str] | None = None, draw_others: bool = False):
    local_bboxes_before = compute_topdown_local(objid2info_before, me_pos_before, me_yaw_before)
    local_bboxes_after = compute_topdown_local(objid2info_after, me_pos_after, me_yaw_after)
    r_before = pick_r(local_bboxes_before, target_oid)
    r_after = pick_r(local_bboxes_after, target_oid)
    r = max(r_before, r_after)
    cx, cy = img_size // 2, img_size // 2
    me_size = 0.04 * img_size
    triangle_points = [
        (cx, cy - me_size // 2),
        (cx - me_size // 2, cy + me_size // 2),
        (cx + me_size // 2, cy + me_size // 2),
    ]
    scale = (img_size * 0.9) / (2.0 * r)
    px_centers = []

    for i, (local_bboxes, objid2info, out_path) in enumerate([(local_bboxes_before, objid2info_before, out_path_before), (local_bboxes_after, objid2info_after, out_path_after)]):
        img = Image.new("RGB", (img_size, img_size), "white")
        draw = ImageDraw.Draw(img)
        draw.polygon(triangle_points, outline="black", width=3)
        px_centers.append({})
        
        # Draw target objects with colors
        to_iterate = [(oid, local_bboxes.get(oid)) for oid in target_oid if target_oid and oid in local_bboxes] if target_oid else []
        for color_idx, (oid, bb) in enumerate(to_iterate):
            if not bb or oid not in objid2info or objid2info[oid]["object_type"] == "Wall":
                continue
            color = colors[color_idx % len(colors)]
            x1 = cx + bb["minx"] * scale
            y1 = cy - bb["maxz"] * scale
            x2 = cx + bb["maxx"] * scale
            y2 = cy - bb["minz"] * scale
            draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
            px = cx + bb["cx"] * scale
            py = cy - bb["cz"] * scale
            px_centers[i][oid] = (px, py)
        
        # Draw all other objects in black if draw_others is True
        if draw_others:
            target_set = set(target_oid) if target_oid else set()
            for oid, bb in local_bboxes.items():
                if oid in target_set or not bb or oid not in objid2info or objid2info[oid]["object_type"] == "Wall":
                    continue
                x1 = cx + bb["minx"] * scale
                y1 = cy - bb["maxz"] * scale
                x2 = cx + bb["maxx"] * scale
                y2 = cy - bb["minz"] * scale
                draw.rectangle((x1, y1, x2, y2), outline="black", width=3)
                px = cx + bb["cx"] * scale
                py = cy - bb["cz"] * scale
                px_centers[i][oid] = (px, py)

        
        img.save(out_path)
    return px_centers


def relocate_object(idx: int, px: float, py: float, ix: int, iy: int, cells: list, filled: set, labels: dict, oid: str, img_size: float, grid_objects: dict, depth: int = 0):
    """
    Relocate an object to an adjacent cell when its preferred position is occupied.
    Handles recursive displacement if the new position is also occupied.
    
    Args:
        grid_objects: dict mapping grid_idx -> (object_id, px, py) for tracking object positions
    """
    # Prevent infinite recursion
    if depth > 10:
        return

    cx, cy = img_size / 2.0, img_size / 2.0
    
    # Get the existing object's position at the target cell
    existing_obj_id, existing_px, existing_py = grid_objects.get(idx, (None, None, None))
    
    if existing_obj_id is None:
        # No existing object, this shouldn't happen but handle gracefully
        dx, dy = px - cx, py - cy
    else:
        # Calculate displacement direction based on relative position to existing object
        dx, dy = px - existing_px, py - existing_py
    
    adx, ady = abs(dx), abs(dy)
    ix_new, iy_new = ix, iy
    if adx >= ady:  # Move horizontally
        if dx < 0:
            ix_new = ix - 1  # Move left
        else:
            ix_new = ix + 1  # Move right
            
    else:  # Move vertically
        if dy < 0:
            iy_new = iy - 1  # Move up
        else:
            iy_new = iy + 1  # Move down
        
    new_idx = iy_new * 5 + ix_new

    if ix_new not in range(5) or iy_new not in range(5) or new_idx == 12:
        if existing_obj_id is not None:
            existing_ix = idx % 5
            existing_iy = idx // 5

            # Now place the current object in the freed position
            cells[idx] = f"[{labels.get(oid, 'obj').capitalize()}]"
            grid_objects[idx] = (oid, px, py)

            # Recursively relocate the displaced object
            relocate_object(idx, existing_px, existing_py, existing_ix, existing_iy, 
                            cells, filled, labels, existing_obj_id, img_size, grid_objects, depth + 1)
        return

    


    # Try to place in the preferred new position
    if new_idx not in filled:  # Position is free and not center
        cells[new_idx] = f"[{labels.get(oid, 'obj').capitalize()}]"
        filled.add(new_idx)
        grid_objects[new_idx] = (oid, px, py)
        return
    
    # If preferred position is occupied by another object, displace it recursively
    else:
        existing_obj_id, existing_px, existing_py = grid_objects[new_idx]
        
        # Extract grid coordinates for the displaced object
        existing_ix = new_idx % 5
        existing_iy = new_idx // 5
        
        # Now place the current object in the freed position
        cells[new_idx] = f"[{labels.get(oid, 'obj').capitalize()}]"
        grid_objects[new_idx] = (oid, px, py)
        
        # Recursively relocate the displaced object
        relocate_object(new_idx, existing_px, existing_py, existing_ix, existing_iy, 
                       cells, filled, labels, existing_obj_id, img_size, grid_objects, depth + 1)
        return
    

def build_text_grid_from_pixels(img_size, px_centers: dict[str, tuple[float, float]], labels: dict[str, str], sample_max: int = 8) -> str:
    # Create 5x5 grid (25 cells total)
    cells = ["[ ]"] * 25
    cells[12] = "[Me]"  # Center position in 5x5 grid (index 12)
    
    if not px_centers:
        return "\n".join([
            " ".join(cells[0:5]), 
            " ".join(cells[5:10]), 
            " ".join(cells[10:15]), 
            " ".join(cells[15:20]), 
            " ".join(cells[20:25])
        ])

    cx, cy = img_size / 2.0, img_size / 2.0

    # Adjust tolerances for 5x5 grid (smaller regions)
    TOL_X = img_size / 10
    TOL_Y = img_size / 10

    def bucket_x(px: float) -> int:
        # Map to 5 columns (0, 1, 2, 3, 4)
        if px < cx - 3 * TOL_X:
            return 0
        elif px < cx - TOL_X:
            return 1
        elif px > cx + 3 * TOL_X:
            return 4
        elif px > cx + TOL_X:
            return 3
        else:
            return 2  # Center column

    def bucket_y(py: float) -> int:
        # Map to 5 rows (0, 1, 2, 3, 4)
        if py < cy - 3 * TOL_Y:
            return 0
        elif py < cy - TOL_Y:
            return 1
        elif py > cy + 3 * TOL_Y:
            return 4
        elif py > cy + TOL_Y:
            return 3
        else:
            return 2  # Center row

    items = list(px_centers.items())
    random.shuffle(items)

    filled = set([12])  # Center position (Me) is already filled
    grid_objects = {}

    for oid, (px, py) in items:
        ix = bucket_x(px)
        iy = bucket_y(py)
        idx = iy * 5 + ix  # 5x5 grid indexing

        if idx in filled:
            relocate_object(idx, px, py, ix, iy, cells, filled, labels, oid, img_size, grid_objects, depth=0)
        else:
            cells[idx] = f"[{labels.get(oid, 'obj').capitalize()}]"
            filled.add(idx)
            grid_objects[idx] = (oid, px, py)

        if len(filled) >= 25 or len(filled) - 1 >= sample_max:
            break

    return "\n".join([
        " ".join(cells[0:5]), 
        " ".join(cells[5:10]), 
        " ".join(cells[10:15]), 
        " ".join(cells[15:20]), 
        " ".join(cells[20:25])
    ])

def generate_representations(
    img_size: int,
    colors: List[str],
    ctrl: Controller,
    goal_pos: dict,
    goal_yaw: float,
    oid: list[str] | None,
    objid2info_before: dict,
    base_dir: Path,
    draw_others: bool = False,
    before_pos: dict | None = None,
    before_yaw: float | None = None,
):
    base = base_dir
    base.mkdir(parents=True, exist_ok=True)

    if before_pos is not None and before_yaw is not None:
        ctrl.step(action="Teleport", position=before_pos, rotation=dict(x=0, y=before_yaw, z=0))

    rgb_t_path = (base / "t_rgb.png")
    Image.fromarray(ctrl.last_event.frame).save(rgb_t_path)

    ego_t_path = (base / "t_ego3d_bbox.png")
    if draw_others:
        # print("objid2info_before:", len([id for id in objid2info_before]))
        other_infos_t = [objid2info_before[id] for id in objid2info_before if objid2info_before[id].get("aabb") and objid2info_before[id]["object_type"] != "Wall" and id not in (oid if oid else [])]
    else:
        other_infos_t = []
    if oid:
        infos_t = [objid2info_before[id] for id in oid if (id in objid2info_before and objid2info_before[id].get("aabb"))]
    else:
        infos_t = []
    # if draw_others:
    #     print(f"Drawing {len(infos_t)} target objects and {len(other_infos_t)} other objects at t")
    draw_ego3d_wireframe(img_size, colors[:len(infos_t)] + ["black"] * len(other_infos_t), ctrl, [info for info in infos_t if info["object_type"] != "Wall"] + other_infos_t, ego_t_path)

    top_t_path = (base / "t_topdown_box.png")
    agent_meta = ctrl.last_event.metadata["agent"]
    me_pos_t = agent_meta.get("position", {})
    me_yaw_t = agent_meta.get("rotation", {}).get("y", 0.0)

    ctrl.step(action="Teleport", position=goal_pos, rotation=dict(x=0, y=goal_yaw, z=0))
    objid2assetid_after = {o["objectId"]: o.get("assetId", "") for o in ctrl.last_event.metadata["objects"]}
    assetid2desc_after = build_assetid2desc_from_scene(ctrl)
    _, objid2info_after, _, _ = get_current_state(ctrl, objid2assetid_after, assetid2desc_after)

    rgb_tp1_path = (base / "tp1_rgb.png")
    Image.fromarray(ctrl.last_event.frame).save(rgb_tp1_path)

    ego_tp1_path = (base / "tp1_ego3d_bbox.png")
    if draw_others:
        other_infos_tp1 = [objid2info_after[id] for id in objid2info_after if objid2info_after[id].get("aabb") and objid2info_after[id]["object_type"] != "Wall" and id not in (oid if oid else [])]
    else:
        other_infos_tp1 = []
    if oid:
        infos_tp1 = [objid2info_after[id] for id in oid if (id in objid2info_after and objid2info_after[id].get("aabb"))]
    else:
        infos_tp1 = []
    draw_ego3d_wireframe(img_size, colors[:len(infos_tp1)] + ["black"] * len(other_infos_tp1), ctrl, [info for info in infos_tp1 if info["object_type"] != "Wall"] + other_infos_tp1, ego_tp1_path)

    top_tp1_path = (base / "tp1_topdown_box.png")
    agent_meta_after = ctrl.last_event.metadata["agent"]
    me_pos_tp1 = agent_meta_after.get("position", {})
    me_yaw_tp1 = agent_meta_after.get("rotation", {}).get("y", 0.0)

    px_centers_t, px_centers_tp1 = draw_topdown_local(img_size, colors, me_pos_t, me_yaw_t, objid2info_before, top_t_path, me_pos_tp1, me_yaw_tp1, objid2info_after, top_tp1_path, target_oid=oid, draw_others=draw_others)
    labels_t = {k: objid2info_before[k]["object_type"] for k in px_centers_t.keys() if k in objid2info_before}

    grid_repr_t = build_text_grid_from_pixels(img_size, px_centers_t, labels_t, sample_max=len(px_centers_t) if draw_others else len(oid))
    labels_tp1 = {k: objid2info_after[k]["object_type"] for k in px_centers_tp1.keys() if k in objid2info_after}
    grid_repr_tp1 = build_text_grid_from_pixels(img_size, px_centers_tp1, labels_tp1, sample_max=len(px_centers_tp1) if draw_others else len(oid))

    return {
        "rgb": {"t": str(rgb_t_path), "t+1": str(rgb_tp1_path)},
        "ego3d_bbox": {"t": str(ego_t_path), "t+1": str(ego_tp1_path)},
        "topdown_bbox": {"t": str(top_t_path), "t+1": str(top_tp1_path)},
        "grid_text": {"t": grid_repr_t, "t+1": grid_repr_tp1}
    }

def project_3d_bbox_to_2d(controller: Controller, obj_info: dict, img_size: int = 512) -> dict:
    aabb = obj_info.get("aabb")
    if not aabb:
        return {
            'total_area': 0.0,
            'visible_area': 0.0, 
            'visibility_ratio': 0.0,
            'projected_polygon': None,
            'clipped_polygon': None
        }

    # base = Image.new("RGB", (img_size, img_size), "white")
    # overlay = Image.new("RGBA", (img_size, img_size), (255, 255, 255, 0))
    # draw = ImageDraw.Draw(overlay)
    meta = controller.last_event.metadata
    agent = meta["agent"]
    cam_pos = meta.get("cameraPosition") or agent.get("cameraPosition") or agent["position"]
    yaw = agent.get("rotation", {}).get("y", 0.0)
    pitch = agent.get("cameraHorizon", 0.0)
    fov = meta.get("cameraFieldOfView", 90.0)


    c = aabb["center"]; s = aabb["size"]
    hx, hy, hz = s["x"] / 2.0, s["y"] / 2.0, s["z"] / 2.0
    corners = np.array([
        [c["x"] - hx, c["y"] - hy, c["z"] - hz],
        [c["x"] + hx, c["y"] - hy, c["z"] - hz],
        [c["x"] + hx, c["y"] + hy, c["z"] - hz],
        [c["x"] - hx, c["y"] + hy, c["z"] - hz],
        [c["x"] - hx, c["y"] - hy, c["z"] + hz],
        [c["x"] + hx, c["y"] - hy, c["z"] + hz],
        [c["x"] + hx, c["y"] + hy, c["z"] + hz],
        [c["x"] - hx, c["y"] + hy, c["z"] + hz],
    ], dtype=np.float32)
    proj = project_points(corners, cam_pos, yaw, pitch, fov, img_size, img_size)
    # edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
    # for a, b in edges:
    #     if proj[a, 2] and proj[b, 2]:
    #         draw.line((proj[a,0], proj[a,1], proj[b,0], proj[b,1]), fill=color, width=3)
    # obj_id = obj_info.get('entry_name', f'obj_{random.randint(1000, 9999)}')
    # Image.alpha_composite(base.convert("RGBA"), overlay).convert("RGB").save(f"./debug_{obj_id}.png")
    # Only use points that are actually visible (proj[:, 2] == 1)
    valid_mask = proj[:, 2] == 1
    valid_points = proj[valid_mask]
    
    if len(valid_points) < 3:
        return {
            'total_area': 0.0,
            'visible_area': 0.0, 
            'visibility_ratio': 0.0,
            'projected_polygon': None,
            'clipped_polygon': None
        }
    
    # Get 2D coordinates of valid points
    points_2d = valid_points[:, :2]  # [u, v] coordinates
    
    try:
        # Create convex hull of projected visible points
        hull_indices = cv2.convexHull(points_2d.astype(np.float32), returnPoints=False).flatten()
        hull_points = points_2d[hull_indices]
        
        # Create Shapely polygon from hull points
        if len(hull_points) >= 3:
            projected_polygon = Polygon(hull_points)
            
            # For total area, we consider the full projected polygon area
            total_area = projected_polygon.area
            
            # Create clipping rectangle (image bounds)
            image_bounds = Polygon([(0, 0), (img_size, 0), (img_size, img_size), (0, img_size)])

            # Clip the projected polygon with image bounds
            clipped_polygon = projected_polygon.intersection(image_bounds)
            visible_area = clipped_polygon.area if hasattr(clipped_polygon, 'area') else 0.0
            
            # Calculate visibility ratio
            visibility_ratio = visible_area / total_area if total_area > 0 else 0.0
            
            return {
                'total_area': total_area,
                'visible_area': visible_area,
                'visibility_ratio': visibility_ratio,
                'projected_polygon': projected_polygon,
                'clipped_polygon': clipped_polygon
            }
        else:
            return {
                'total_area': 0.0,
                'visible_area': 0.0,
                'visibility_ratio': 0.0,
                'projected_polygon': None,
                'clipped_polygon': None
            }
            
    except Exception as e:
        obj_id = obj_info.get('entry_name', 'unknown_obj')
        print(f"Error in polygon calculation for {obj_id}: {e}")
        return {
            'total_area': 0.0,
            'visible_area': 0.0,
            'visibility_ratio': 0.0,
            'projected_polygon': None,
            'clipped_polygon': None
        }

def filter_objects_by_2d_visibility_polygon(controller: Controller, nav_vis: list[str], objid2info: dict, min_visibility: float = 0.7, img_size: int = 512, min_area: int = 700) -> list[str]:
    valid_objects = []
    
    for obj_id in nav_vis:
        if obj_id not in objid2info:
            continue
            
        obj_info = objid2info[obj_id]
        
        # Project 3D bbox to 2D and calculate visibility
        visibility_info = project_3d_bbox_to_2d(controller, obj_info, img_size=img_size)
        
        # print(f"{obj_id}: vis_ratio={visibility_info['visibility_ratio']:.2f}, area={visibility_info['visible_area']:.0f}, total_area={visibility_info['total_area']:.0f}")
        if (visibility_info and 
            visibility_info['visibility_ratio'] >= min_visibility and
            visibility_info['visible_area'] >= min_area):  # Minimum pixel area threshold
            valid_objects.append(obj_id)
    
    return valid_objects




def render_person_at_position(base_img, ctrl, person_px, person_py, person_yaw, prerendered_path, distance_to_person, person_height_px=150):
    """
    Load pre-rendered person image and overlay it at the specified 2D pixel position.
    
    Args:
        base_img: PIL Image to composite person onto
        ctrl: AI2-THOR controller
        person_px: X pixel position (center of person)
        person_py: Y pixel position (bottom/feet of person)
        person_yaw: Yaw rotation for person in degrees (direction person is facing)
        blend_file_path: Path to sophia.blend file (used to derive prerendered path)
        distance_to_person: Distance from camera to person in world units
        person_height_px: Base height of person in pixels before perspective scaling
    
    Returns:
        Tuple of (PIL Image with person composited, occlusion_detected boolean)
    """
    
    # Get camera parameters
    meta = ctrl.last_event.metadata
    agent = meta["agent"]
    cam_pos = agent.get("position", {})
    cam_yaw = agent.get("rotation", {}).get("y", 0.0)
    cam_horizon = agent.get("cameraHorizon", 0.0)  # Pitch angle in degrees
    fov = meta.get("cameraFieldOfView", 90.0)

    # Calculate person rotation relative to camera view
    relative_yaw = (person_yaw - cam_yaw) % 360
    
    # floor to nearest integer degree for prerendered lookup
    degree_idx = int(math.floor(relative_yaw))

    # Load pre-rendered image
    prerendered_path = prerendered_path / f"deg_{degree_idx}.png"
    if not prerendered_path.exists():
        raise FileNotFoundError(f"Pre-rendered image not found: {prerendered_path}. Please run prerender_sophia.py first.")
    
    person_img = Image.open(prerendered_path).convert("RGBA")
    
    # Perspective projection for scaling
    person_height_world = 1.7  # meters
    img_height = base_img.height
    img_width = base_img.width
    
    fov_v_rad = math.radians(fov)
    focal_length = (img_height / 2.0) / math.tan(fov_v_rad / 2.0)
    
    perspective_scale = 0.5
    base_distance = 1.8
    effective_distance = base_distance + (distance_to_person - base_distance) * perspective_scale
    scaled_height_px = int((person_height_world * focal_length) / max(effective_distance, 0.85))
    
    
    # Brighten the person image
    from PIL import ImageEnhance
    enhancer = ImageEnhance.Brightness(person_img)
    person_img = enhancer.enhance(2.5)

    # Resize to desired height while maintaining aspect ratio
    aspect_ratio = person_img.width / person_img.height
    new_width = int(scaled_height_px * aspect_ratio)
    person_img = person_img.resize((new_width, scaled_height_px), Image.Resampling.LANCZOS)
    
    # Composite person onto base image
    paste_x = int(person_px - person_img.width // 2)
    paste_y = int(person_py - person_img.height + 10)
    
    occlusion_detected = False

    # Extract person segmentation mask (alpha channel with non-zero values)
    person_array = np.array(person_img)
    person_mask = person_array[:, :, 3] > 0  # Alpha channel > 0
    
    # Create full-size mask at paste position
    full_person_mask = np.zeros((img_height, img_width), dtype=bool)
    
    # Calculate valid region for pasting
    src_y_start = max(0, -paste_y)
    src_y_end = min(person_img.height, img_height - paste_y)
    src_x_start = max(0, -paste_x)
    src_x_end = min(person_img.width, img_width - paste_x)
    
    dst_y_start = max(0, paste_y)
    dst_y_end = min(img_height, paste_y + person_img.height)
    dst_x_start = max(0, paste_x)
    dst_x_end = min(img_width, paste_x + person_img.width)
    
    # Place person mask in full image
    if src_y_end > src_y_start and src_x_end > src_x_start:
        full_person_mask[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = \
            person_mask[src_y_start:src_y_end, src_x_start:src_x_end]

    # Get all objects in the scene and find foreground objects
    instance_masks = ctrl.last_event.instance_masks
    
    for obj in meta.get("objects", []):
        obj_id = obj["objectId"]
        obj_distance = obj.get("distance", float("inf"))
        
        # Check if object is in front of person (closer to camera)
        if obj_distance < distance_to_person and obj_id in instance_masks:
            # Get object segmentation mask
            obj_mask = instance_masks[obj_id]
            
            # Check for overlap between person mask and object mask
            overlap = np.logical_and(full_person_mask, obj_mask)
            
            if np.any(overlap):
                occlusion_detected = True
                break

    result = base_img.convert("RGBA")
    result.paste(person_img, (paste_x, paste_y), person_img)

    return result.convert("RGB"), occlusion_detected