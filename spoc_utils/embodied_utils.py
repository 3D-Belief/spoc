from typing import Dict, List, Optional, Tuple, Any
import copy
from PIL import Image

def square_image(image):
    # consider both rgb and depth images
    height, width = image.shape[:2]
    size = min(height, width)
    start_x = (width - size) // 2
    start_y = (height - size) // 2
    return image[start_y:start_y+size, start_x:start_x+size]

def find_object_node(scene_graph, target_id):
    """
    Recursively search for an object with a given id in a scene graph.

    Args:
        scene_graph (list[dict]): The scene graph (list of dicts, each may have children).
        target_id (str): The object id to search for.

    Returns:
        node or None: return node if found, otherwise None.
    """
    for node in scene_graph:
        if node.get("id") == target_id:
            return node

        # search recursively in children if they exist
        children = node.get("children", [])
        if children:
            result = find_object_node(children, target_id)
            if result is not None:
                return result
    return None

def find_agent_room(
    scene_graph: Dict,
    pos_x: Optional[float] = None,
    pos_z: Optional[float] = None,
    *,
    return_all_matches: bool = False,
    eps: float = 1e-4,
) -> Any:
    def _point_on_segment(px, pz, x1, z1, x2, z2) -> bool:
        if not (min(x1, x2) - eps <= px <= max(x1, x2) + eps and
                min(z1, z2) - eps <= pz <= max(z1, z2) + eps):
            return False
        return abs((x2 - x1) * (pz - z1) - (z2 - z1) * (px - x1)) <= eps

    def _point_in_poly(px, pz, poly: List[Dict[str, float]]) -> bool:
        inside = False
        n = len(poly)
        for i in range(n):
            x1, z1 = poly[i]["x"], poly[i]["z"]
            x2, z2 = poly[(i + 1) % n]["x"], poly[(i + 1) % n]["z"]
            if _point_on_segment(px, pz, x1, z1, x2, z2):
                return True
            crosses = ((z1 > pz) != (z2 > pz))
            if crosses:
                x_cross = x1 + (x2 - x1) * (pz - z1) / (z2 - z1)
                if px <= x_cross - eps:
                    inside = not inside
        return inside

    if pos_x is None or pos_z is None:
        agent = (scene_graph.get("metadata") or {}).get("agent", {}).get("position", {})
        pos_x = float(agent.get("x", 0.0))
        pos_z = float(agent.get("z", 0.0))

    rooms = scene_graph.get("rooms", [])
    rooms_sorted = sorted(rooms, key=lambda r: r.get("id", ""))

    hits: List[Tuple[str, str]] = []
    for r in rooms_sorted:
        poly = r.get("floorPolygon") or []
        if not poly:
            continue
        inside = _point_in_poly(pos_x, pos_z, poly)

        # If your graph includes holes, subtract them here:
        for hole in r.get("floorPolygonHoles", []) or []:
            if _point_in_poly(pos_x, pos_z, hole):
                inside = False
                break

        if inside:
            hits.append((r.get("id", ""), r.get("roomType", "")))
            if not return_all_matches:
                return hits[0]

    return hits if return_all_matches else ("", "")

def get_top_down_frame(controller):
    # Setup the top-down camera
    event = controller.step(action="GetMapViewCameraProperties", raise_for_failure=True)
    pose = copy.deepcopy(event.metadata["actionReturn"])

    bounds = event.metadata["sceneBounds"]["size"]
    max_bound = max(bounds["x"], bounds["z"])

    pose["fieldOfView"] = 50
    pose["position"]["y"] += 1.1 * max_bound
    pose["orthographic"] = False
    pose["farClippingPlane"] = 50
    del pose["orthographicSize"]

    # add the camera to the scene
    event = controller.step(
        action="AddThirdPartyCamera",
        **pose,
        skyboxColor="white",
        raise_for_failure=True,
    )
    top_down_frame = event.third_party_camera_frames[-1]
    return Image.fromarray(top_down_frame)

def distance_traveled_step(action_name, action_params):
    distance_traveled, angle_turned = 0.0, 0.0
    if action_name == "MoveAhead" or action_name == "MoveBack":
        # Calculate distance between last two positions
        distance = action_params.get('moveMagnitude', 0.0)
        distance_traveled += distance
    elif action_name == "RotateLeft" or action_name == "RotateRight":
        # Track angle turned
        degrees = action_params.get('degrees', 0)
        angle_turned += abs(degrees)
    return distance_traveled, angle_turned