import os
import sys
import h5py
import json
import numpy as np
import math
import matplotlib.pyplot as plt
import cv2
import gzip
import jsonlines
import copy
from pathlib import Path
from PIL import Image
import glob
from tqdm import tqdm
import datetime
from scipy.spatial.transform import Rotation as R

ROOT_DIR = str(Path(__file__).parent.parent)
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

from spoc_utils.embodied_utils import find_agent_room

# Set environment variables
os.environ["OBJAVERSE_DATA_DIR"] = "/home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/spoc/data"
os.environ["OBJAVERSE_HOUSES_DIR"] = "/home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/spoc/data/houses_2023_07_28"
# Set current working directory

def convert_action(action_char):
    """
    Convert abbreviated action character/string to full action name and parameters
    
    Parameters:
    action_char (str): Character or string representing an action
    
    Returns:
    str: Full action name
    dict: Any additional parameters for the action
    """
    action_map = {
        # Navigation actions
        'm': ('MoveAhead', {'moveMagnitude': 0.2}),
        'b': ('MoveBack', {'moveMagnitude': 0.2}),
        'l': ('RotateLeft', {'degrees': 30}),
        'r': ('RotateRight', {'degrees': 30}),
        'ls': ('RotateLeft', {'degrees': 6}),
        'rs': ('RotateRight', {'degrees': 6}),
        
        # Arm manipulation actions
        'yp': ('MoveArmHeight', {'y': 0.1, 'coordinate_system': 'relative'}),
        'ym': ('MoveArmHeight', {'y': -0.1, 'coordinate_system': 'relative'}),
        'zp': ('MoveArmForward', {'z': 0.1, 'coordinate_system': 'relative'}),
        'zm': ('MoveArmForward', {'z': -0.1, 'coordinate_system': 'relative'}),
        'yps': ('MoveArmHeight', {'y': 0.02, 'coordinate_system': 'relative'}),
        'yms': ('MoveArmHeight', {'y': -0.02, 'coordinate_system': 'relative'}),
        'zps': ('MoveArmForward', {'z': 0.02, 'coordinate_system': 'relative'}),
        'zms': ('MoveArmForward', {'z': -0.02, 'coordinate_system': 'relative'}),
        
        # Wrist actions
        'wp': ('RotateWristRelative', {'degrees': 10}),
        'wm': ('RotateWristRelative', {'degrees': -10}),
        
        # Task completion actions
        'end': ('Done', {}),
        'sub_done': ('SubDone', {}),  # May need to be mapped to an appropriate AI2-THOR action
        
        # Object interaction actions
        'p': ('PickupObject', {}),  # Will need object ID from the scene
        'd': ('DropObject', {})
    }
    
    if action_char in action_map:
        action_name, params = action_map[action_char]
        return action_name, params
    else:
        return 'Pass', {}
    
def replay_trajectory_with_modalities(house_id, house_data, trajectory_data, output_dir):
    """
    Replay a complete trajectory and generate RGB, Depth, and Semantic modalities
    
    Parameters:
    house_id (str): Identifier for the house
    house_data (dict): House data from the assets file
    trajectory_data (dict): Trajectory data including positions, rotations, and actions
    output_dir (str): Directory to save the output
    
    Returns:
    dict: Information about the generated frames
    """
    from environment.stretch_controller import StretchController
    from spoc_utils.constants.stretch_initialization_utils import STRETCH_ENV_ARGS
    from scipy.spatial.transform import Rotation as R
    
    # Define a JSON encoder class for numpy types
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super(NumpyEncoder, self).default(obj)
    
    # Make sure output directories exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create subdirectories for different modalities
    rgb_dir = os.path.join(output_dir, "rgb")
    depth_dir = os.path.join(output_dir, "depth")
    semantic_dir = os.path.join(output_dir, "semantic")
    # topdown_dir = os.path.join(output_dir, "topdown")
    pose_dir = os.path.join(output_dir, "pose")
    
    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)
    os.makedirs(semantic_dir, exist_ok=True)
    # os.makedirs(topdown_dir, exist_ok=True)
    os.makedirs(pose_dir, exist_ok=True)
    
    # Extract trajectory components
    positions = trajectory_data.get('positions', [])
    rotations = trajectory_data.get('rotations', [])
    actions = trajectory_data.get('actions', [])
    object_ids = trajectory_data.get('object_ids', [])
    
    if not positions or len(positions) == 0:
        print("ERROR: No position data found for this trajectory")
        return {'frame_count': 0}
    
    # Get initial position and rotation
    initial_position = positions[0]
    initial_rotation = rotations[0]
    
    print("Initializing StretchController...")
    # Initialize the controller with the house
    controller = StretchController(**STRETCH_ENV_ARGS)
    event = controller.reset(house_data)
    
    # Set the robot to the correct initial position and rotation
    print(f"Teleporting agent to position {initial_position} and rotation {initial_rotation}")
    controller.step(
        action="TeleportFull",
        position=initial_position,
        rotation=initial_rotation,
        horizon=0.0,  # Level camera view
        standing=True  # Agent is standing
    )
    
    # Set up frame info tracking
    frame_info = {
        'width': controller.navigation_camera.shape[1],
        'height': controller.navigation_camera.shape[0],
        'frame_count': 0
    }
    
    def nav_camera_pose(event):
        """Return cam->world 4x4 for the navigation camera."""
        pos = event.metadata["cameraPosition"]                      # dict x,y,z (world)
        agent = event.metadata["agent"]
        yaw = agent["rotation"]["y"]                 # degrees (around Y)
        pitch = agent["cameraHorizon"]
        roll = 0.0

        R_aw = R.from_euler('xyz', [0.0, yaw, 0.0], degrees=True).as_matrix() # R_aw * R_ca = R_cw
        R_ca = R.from_euler('xyz', [pitch, 0.0, 0.0], degrees=True).as_matrix()
        R_cw = R_aw @ R_ca
        t_cw = np.array([pos["x"], pos["y"], pos["z"]], dtype=float)
        extrinsic_cam2world = np.eye(4, dtype=float)
        extrinsic_cam2world[:3,:3] = R_cw
        extrinsic_cam2world[:3, 3] = t_cw
        F = np.diag([1, 1, -1, 1])
        extrinsic_cam2world =  F @ extrinsic_cam2world @ F  # left-handed to right-handed
        return extrinsic_cam2world

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
        
    def save_frame(step_num):
        """
        Save RGB, depth, instance segmentation, and camera info data for the current frame.
        This function uses the expanded positions and rotations arrays that account for
        the interpolated small rotation steps.
        
        Args:
            step_num (int): The current step/frame number
        """
        
        # Get RGB image
        rgb_image = controller.navigation_camera
        
        # Calculate square crop dimensions
        height, width = rgb_image.shape[:2]
        size = min(height, width)
        start_x = (width - size) // 2
        start_y = (height - size) // 2
        
        # Crop RGB image to square
        rgb_image_square = rgb_image[start_y:start_y+size, start_x:start_x+size]
        
        # Save cropped RGB image
        rgb_path = os.path.join(rgb_dir, f"frame_{step_num:04d}.png")
        Image.fromarray(rgb_image_square).save(rgb_path)
        
        # Save Depth image - using navigation_depth_frame
        if hasattr(controller, 'navigation_depth_frame'):
            depth_image = controller.navigation_depth_frame
            
            # Crop depth image to the same square dimensions
            if len(depth_image.shape) == 2:  # Check if it's a 2D array
                depth_image_square = depth_image[start_y:start_y+size, start_x:start_x+size]
            else:
                depth_image_square = depth_image[start_y:start_y+size, start_x:start_x+size, :]
            
            # Save raw depth data (cropped)
            depth_raw_path = os.path.join(depth_dir, f"frame_{step_num:04d}_raw.npy")
            np.save(depth_raw_path, depth_image_square)
            
            # Normalize for visualization
            if np.max(depth_image_square) > 0:
                depth_norm = (depth_image_square / np.max(depth_image_square) * 255).astype(np.uint8)
            else:
                depth_norm = np.zeros_like(depth_image_square, dtype=np.uint8)
            
            # Save grayscale depth image (cropped)
            depth_gray_path = os.path.join(depth_dir, f"frame_{step_num:04d}.png")
            cv2.imwrite(depth_gray_path, depth_norm)
        
        # Save Instance Segmentation - using navigation_camera_segmentation
        if hasattr(controller, 'navigation_camera_segmentation'):
            # Get the segmentation data
            seg_frame = controller.navigation_camera_segmentation
            
            # Crop segmentation frame to square
            seg_frame_square = seg_frame[start_y:start_y+size, start_x:start_x+size]
            
            # Save raw data (cropped)
            seg_raw_path = os.path.join(semantic_dir, f"frame_{step_num:04d}_raw.npy")
            np.save(seg_raw_path, seg_frame_square)
            
            # Dictionary to store object info for this frame
            frame_objects = {}
            
            # Process object IDs directly
            colors = np.unique(seg_frame_square.reshape(-1, 3), axis=0)
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
            
            # Save the visualization
            seg_path = os.path.join(semantic_dir, f"frame_{step_num:04d}.png")
            Image.fromarray(seg_frame_square).save(seg_path)
            
            # Save object metadata
            metadata = {
                "scene_metadata": {
                    "total_objects": len(frame_objects),
                    "object_categories": list(set(obj['name'] for obj in frame_objects.values()))
                },
                "objects": frame_objects
            }

            meta_path = os.path.join(semantic_dir, f"frame_{step_num:04d}_meta.json")
            with open(meta_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            # Save object mask
            kept_colors = [info["color"] for info in frame_objects.values() if info["name"] != "Unknown"]

            H, W = seg_frame_square.shape[:2]
            mask_path = os.path.join(semantic_dir, f"frame_{step_num:04d}_object_binary_mask.png")

            if len(kept_colors) == 0:
                # No known objects this frame -> all zeros
                Image.fromarray(np.zeros((H, W), dtype=np.uint8)).save(mask_path)
            else:
                # Convert the color image to a single 24-bit code per pixel: (R<<16)|(G<<8)|B
                seg_codes = (
                    seg_frame_square[..., 0].astype(np.uint32) << 16
                    | seg_frame_square[..., 1].astype(np.uint32) << 8
                    | seg_frame_square[..., 2].astype(np.uint32)
                )

                # Codes we want to keep
                keep_codes = np.array(
                    [(r << 16) | (g << 8) | b for (r, g, b) in kept_colors], dtype=np.uint32
                )

                # Boolean mask: True where pixel color matches any kept color
                mask_bool = np.isin(seg_codes, keep_codes)

                # Save as 0/255 PNG
                mask_u8 = (mask_bool.astype(np.uint8) * 255)
                Image.fromarray(mask_u8).save(mask_path)
            
            print(f"Created segmentation visualization with {len(frame_objects)} objects")
        
        # # Save top down view
        # top_down_frame = get_top_down_frame()
        # top_down_path = os.path.join(topdown_dir, f"frame_{step_num:04d}_top_down.png")
        # top_down_frame.save(top_down_path)

        # Save camera information for the current frame
        if step_num < len(positions) and step_num < len(rotations):
            pose = nav_camera_pose(controller.controller.last_event)
            # save as npy array
            pose_path = os.path.join(pose_dir, f"frame_{step_num:04d}_pose.npy")
            np.save(pose_path, pose)
        
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

    def get_top_down_frame():
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
                
    # Save initial state (frame 0)
    save_frame(0)
    frame_info['frame_count'] += 1
    print(f"Saved initial frame (0)")
    
    # This list will store all actions we need to execute
    processed_actions = []
    
    # This will map each new action step to its original action in the trajectory
    # This helps with object ID lookup and for tracking the original poses
    action_to_original_map = {}
    
    # Calculate the new positions and rotations for all frames after breaking down rotations
    # We'll expand the positions and rotations arrays to match our processed actions
    expanded_positions = [positions[0]]  # Start with initial position
    expanded_rotations = [rotations[0]]  # Start with initial rotation
    
    # Process the actions to replace 'l' with five 'ls' and 'r' with five 'rs'
    new_frame_idx = 1  # Start at 1 since we already have frame 0
    for orig_step, action_str in enumerate(actions[1:], 1):  # Skip the first action as it's typically empty
        if not action_str:
            processed_actions.append("")
            action_to_original_map[new_frame_idx] = orig_step
            new_frame_idx += 1
            # Just copy the original position/rotation for empty actions
            if orig_step < len(positions) and orig_step < len(rotations):
                expanded_positions.append(positions[orig_step])
                expanded_rotations.append(rotations[orig_step])
            continue
        
        # Handle rotation actions specially
        if action_str == 'l':
            # Get original rotation before this action
            if orig_step < len(rotations):
                start_rotation = rotations[orig_step-1] if orig_step > 0 else rotations[0]
                end_rotation = rotations[orig_step]
                
                # Calculate 5 intermediate rotations between start and end
                # We need to handle the 6-degree increments manually
                start_yaw = float(start_rotation.get('y', 0.0))
                
                # For a left rotation, we're adding positive 30 degrees (6 degrees × 5)
                for i in range(5):
                    # Replace a single 'l' with five 'ls'
                    processed_actions.append('ls')
                    action_to_original_map[new_frame_idx] = orig_step
                    
                    # Calculate intermediate position
                    # For simplicity, we'll interpolate linearly between start and end positions
                    if orig_step < len(positions):
                        start_pos = positions[orig_step-1] if orig_step > 0 else positions[0]
                        end_pos = positions[orig_step]
                        
                        # Linear interpolation of position
                        interp_factor = (i + 1) / 5.0  # progress from 0.2 to 1.0
                        interp_pos = {
                            'x': start_pos.get('x', 0.0) + interp_factor * (end_pos.get('x', 0.0) - start_pos.get('x', 0.0)),
                            'y': start_pos.get('y', 0.0) + interp_factor * (end_pos.get('y', 0.0) - start_pos.get('y', 0.0)),
                            'z': start_pos.get('z', 0.0) + interp_factor * (end_pos.get('z', 0.0) - start_pos.get('z', 0.0))
                        }
                        expanded_positions.append(interp_pos)
                    
                    # Calculate intermediate rotation (each step is +6 degrees in y)
                    interp_rotation = {
                        'x': start_rotation.get('x', 0.0),
                        'y': start_yaw - (i + 1) * 6.0,  # Each step adds 6 degrees
                        'z': start_rotation.get('z', 0.0)
                    }
                    expanded_rotations.append(interp_rotation)
                    
                    new_frame_idx += 1
                    
        elif action_str == 'r':
            # Get original rotation before this action
            if orig_step < len(rotations):
                start_rotation = rotations[orig_step-1] if orig_step > 0 else rotations[0]
                end_rotation = rotations[orig_step]
                
                # Calculate 5 intermediate rotations between start and end
                start_yaw = float(start_rotation.get('y', 0.0))
                
                # For a right rotation, we're subtracting 30 degrees (6 degrees × 5)
                for i in range(5):
                    # Replace a single 'r' with five 'rs'
                    processed_actions.append('rs')
                    action_to_original_map[new_frame_idx] = orig_step
                    
                    # Calculate intermediate position
                    if orig_step < len(positions):
                        start_pos = positions[orig_step-1] if orig_step > 0 else positions[0]
                        end_pos = positions[orig_step]
                        
                        # Linear interpolation of position
                        interp_factor = (i + 1) / 5.0  # progress from 0.2 to 1.0
                        interp_pos = {
                            'x': start_pos.get('x', 0.0) + interp_factor * (end_pos.get('x', 0.0) - start_pos.get('x', 0.0)),
                            'y': start_pos.get('y', 0.0) + interp_factor * (end_pos.get('y', 0.0) - start_pos.get('y', 0.0)),
                            'z': start_pos.get('z', 0.0) + interp_factor * (end_pos.get('z', 0.0) - start_pos.get('z', 0.0))
                        }
                        expanded_positions.append(interp_pos)
                    
                    # Calculate intermediate rotation (each step is -6 degrees in y)
                    interp_rotation = {
                        'x': start_rotation.get('x', 0.0),
                        'y': start_yaw + (i + 1) * 6.0,  # Each step subtracts 6 degrees
                        'z': start_rotation.get('z', 0.0)
                    }
                    expanded_rotations.append(interp_rotation)
                    
                    new_frame_idx += 1
        else:
            # Keep other actions as they are
            processed_actions.append(action_str)
            action_to_original_map[new_frame_idx] = orig_step
            
            # Just copy the original position/rotation for non-rotation actions
            if orig_step < len(positions) and orig_step < len(rotations):
                expanded_positions.append(positions[orig_step])
                expanded_rotations.append(rotations[orig_step])
                
            new_frame_idx += 1
    
    print(f"Original actions: {len(actions)}, Processed actions: {len(processed_actions)}")
    print(f"Original positions/rotations: {len(positions)}/{len(rotations)}, Expanded: {len(expanded_positions)}/{len(expanded_rotations)}")
    
    # Replace the original positions and rotations with our expanded ones
    positions = expanded_positions
    rotations = expanded_rotations
    # Initialize current_room list and distance tracking
    current_room = []
    distance_traveled = 0.0
    angle_turned = 0.0
    # Save the initial top-down view
    top_down_image = get_top_down_frame()
    top_down_path = os.path.join(output_dir, "top_down_view_initial.png")
    top_down_image.save(top_down_path)
    
    # Execute all processed actions
    for step, action_str in enumerate(processed_actions, 1):  # Start at 1 to match the frame count
        if not action_str:
            continue
        
        # Convert action character to full action name
        action_name, action_params = convert_action(action_str)
        
        print(f"Step {step}: Executing action {action_name} ({action_str})")
        
        try:
            # Special handling for pickup and drop actions with object IDs
            if action_name == "PickupObject":
                # Get the corresponding original step for object ID lookup
                orig_step = action_to_original_map.get(step, step)
                obj_id = object_ids[orig_step] if object_ids and orig_step < len(object_ids) else None
                if obj_id:
                    action_params['objectId'] = obj_id
            
            # Execute the action
            event = controller.step(action=action_name, **action_params)

            agent_position = event.metadata["agent"]["position"]
            agent_x = agent_position["x"]
            agent_z = agent_position["z"]

            # Append current room info
            room_name = find_agent_room(controller.current_scene_json, agent_x, agent_z)
            current_room.append(room_name)

            # Calculate distance traveled
            dist, angle = distance_traveled_step(action_name, action_params)
            distance_traveled += dist
            angle_turned += angle
            
            # Save frames after action
            save_frame(step)
            frame_info['frame_count'] += 1
            print(f"Saved frame {step}")
            
        except Exception as e:
            print(f"WARNING: Error executing action '{action_name}' at step {step}: {e}")
            # Continue with the next action
        
    # Save the final top-down view
    top_down_image = get_top_down_frame()
    top_down_path = os.path.join(output_dir, "top_down_view_final.png")
    top_down_image.save(top_down_path)
    
    # Extract last position and rotation
    if len(positions) > 0:
        final_position = positions[-1]
    else:
        final_position = initial_position

    if len(rotations) > 0:
        final_rotation = rotations[-1]
    else:
        final_rotation = initial_rotation

    print(f"Trajectory replay complete. Generated {frame_info['frame_count']} frames.")
    
    # Create a metadata file with trajectory information
    # Derive annotations: unique rooms visited and whether single-room or cross-room
    def _normalize_room_label(r):
        if not r:
            return 'unknown'
        rl = r.lower()
        if 'living' in rl:
            return 'living room'
        if 'kitchen' in rl:
            return 'kitchen'
        if 'bed' in rl:
            return 'bedroom'
        if 'bath' in rl:
            return 'bathroom'
        return rl

    unique_rooms_raw = [r for r in set(current_room) if r]
    unique_rooms = list({_normalize_room_label(r[1]) for r in unique_rooms_raw})
    if len(unique_rooms) == 1:
        room_annotation = f"single-room: {unique_rooms[0]}"
    else:
        room_annotation = 'cross-room'

    action_count = len([a for a in processed_actions if a])
    if action_count < 50:
        length_annotation = 'short'
    elif action_count <= 200:
        length_annotation = 'med'
    else:
        length_annotation = 'long'

    metadata = {
        'house_id': house_id,
        'object_id': object_ids[0] if object_ids else None,
        'initial_position': initial_position,
        'initial_rotation': initial_rotation,
        'final_position': final_position,
        'final_rotation': final_rotation,
        'frames': int(frame_info['frame_count']),
        'action_count': int(action_count),
        'actions': [a for a in processed_actions if a],
        'rooms_visited': current_room,
        'unique_rooms': unique_rooms,
        'room_annotation': room_annotation,
        'length_annotation': length_annotation,
        'distance_traveled': float(distance_traveled),
        'angle_turned': float(angle_turned),
        'width': int(frame_info['width']),
        'height': int(frame_info['height'])
    }

    with open(os.path.join(output_dir, "trajectory_metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Clean up
    controller.stop()
    
    return frame_info

def create_video_from_frames(frames_dir, output_path, fps=10):
    """
    Create a video from the saved frames (which are already square)
    
    Parameters:
    frames_dir (str): Directory containing the frames
    output_path (str): Path for the output video
    fps (int): Frames per second
    
    Returns:
    bool: Success status
    """
    # Get all frame files sorted numerically
    frames = sorted([f for f in os.listdir(frames_dir) if f.startswith('frame_') and f.endswith('.png')])
    
    if not frames:
        print(f"No frames found in {frames_dir}")
        return False
    
    # Read first frame to get dimensions
    first_frame_path = os.path.join(frames_dir, frames[0])
    first_frame = cv2.imread(first_frame_path)
    if first_frame is None:
        print(f"Could not read first frame: {first_frame_path}")
        return False
    
    height, width, _ = first_frame.shape
    
    # Create video writer (frames are already square)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 codec
    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Add frames to video
    for frame in frames:
        img_path = os.path.join(frames_dir, frame)
        img = cv2.imread(img_path)
        if img is not None:
            video.write(img)
    
    video.release()
    print(f"Created video at {output_path} with dimensions {width}x{height}")
    return True

def load_house_from_prior(house_index=0):
    """
    Load a house directly from local JSONL files instead of using the prior dataset
    
    Parameters:
    house_index (int): Index of the house to load (default: 0)
    
    Returns:
    dict: House data as a dictionary
    """
    from spoc_utils.constants.objaverse_data_dirs import OBJAVERSE_HOUSES_DIR
    import gzip
    import json
    import jsonlines
    
    print(f"Loading house at index {house_index} from local dataset...")
    
    # If direct file not found, try to load from JSONL files
    for split in ["val"]:
        houses_path = os.path.join(OBJAVERSE_HOUSES_DIR, f"{split}.jsonl.gz")
        
        if not os.path.exists(houses_path):
            print(f"Warning: {houses_path} does not exist, trying next split")
            continue
        
        print(f"Load from {houses_path}")
        try:
            # Manual approach using gzip and line-by-line JSON parsing to handle errors
            current_index = 0
            with gzip.open(houses_path, 'rt', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        # Skip empty lines
                        line = line.strip()
                        if not line:
                            continue
                        
                        # Parse JSON
                        house = json.loads(line)
                        
                        # Check if this is the house we want
                        if current_index == house_index:
                            print(f"Successfully loaded house at index {house_index} from {split} split (line {line_num})")
                            if 'id' in house:
                                print(f"House ID: {house['id']}")
                            return house
                        
                        current_index += 1
                        
                    except json.JSONDecodeError:
                        print(f"Warning: Invalid JSON at line {line_num}, skipping")
                    except Exception as e:
                        print(f"Warning: Error processing line {line_num}: {e}, skipping")
            
            print(f"Reached end of file {houses_path} after processing {current_index} houses, but didn't find index {house_index}")
            
        except Exception as e:
            print(f"Error reading {houses_path}: {e}")
    
    # Fallback to a simple house creation approach
    print(f"Could not find house with index {house_index}, creating a default house")
    
    # Create a minimal default house that will work with the StretchController
    default_house = {
        "id": f"default_{house_index}",
        "objects": [],
        "rooms": [],
        "scene_bounds": {
            "center": {"x": 0, "y": 0, "z": 0},
            "size": {"x": 10, "y": 3, "z": 10}
        }
    }
    
    return default_house

def process_all_houses_and_episodes(base_dir, output_base_dir, resume=True, start_house=None, end_house=None, houses_file=None, outdir_prefix=""):
    """
    Process houses and their episodes in the specified directory within a given range
    
    Parameters:
    base_dir (str): Base directory containing house folders
    output_base_dir (str): Base directory for output
    resume (bool): If True, skip already completed houses/episodes (default: True)
    start_house (str): House ID to start processing from (default: None, start from the beginning)
    end_house (str): House ID to end processing at, inclusive (default: None, process until the end)
    houses_file (str): Optional path to a text file containing house directories to process (default: None)
    outdir_prefix (str): Optional prefix for output directory names (default: "")
    
    Returns:
    dict: Summary of processed houses and episodes
    """
    # Create output base directory
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Check for existing summary file to resume from
    summary_path = os.path.join(output_base_dir, "processing_summary.json")
    existing_summary = None
    
    if resume and os.path.exists(summary_path):
        try:
            with open(summary_path, 'r') as f:
                existing_summary = json.load(f)
            print(f"Found existing summary file. Previously processed {existing_summary['total_houses']} houses and {existing_summary['total_episodes']} episodes.")
            
            # Create a set of already processed house/episode combinations for quick lookup
            processed_items = set()
            for house in existing_summary['processed_houses']:
                house_id = house['house_id']
                for episode in house['episodes']:
                    episode_idx = episode['episode_idx']
                    processed_items.add(f"{house_id}_{episode_idx}")
                    
            summary = existing_summary
            
        except Exception as e:
            print(f"Error reading existing summary, starting fresh: {str(e)}")
            existing_summary = None
    
    # If no existing summary or not resuming, create a new one
    if existing_summary is None:
        summary = {
            'total_houses': 0,
            'total_episodes': 0,
            'processed_houses': [],
            'errors': []
        }
        processed_items = set()
    
    # Create a progress tracking file that's updated frequently
    progress_path = os.path.join(output_base_dir, "processing_progress.json")
    
    # Function to save progress
    def save_progress():
        with open(progress_path, 'w') as f:
            json.dump({
                'timestamp': str(datetime.datetime.now()),
                'total_houses_processed': summary['total_houses'],
                'total_episodes_processed': summary['total_episodes'],
                'last_house_id': house_id if 'house_id' in locals() else None,
                'last_episode_idx': episode_idx if 'episode_idx' in locals() else None
            }, f, indent=2)
    
    # Get the set of houses that are already fully processed
    fully_processed_houses = set()
    if existing_summary is not None:
        for house in existing_summary['processed_houses']:
            # If a house has been fully processed, all its episodes would be in processed_items
            fully_processed_houses.add(house['house_id'])
    
    # Find all house directories
    if houses_file is not None and os.path.exists(houses_file):
        # Read house directories from the specified file
        with open(houses_file, 'r') as f:
            house_dirs = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(house_dirs)} house directories from {houses_file}")
    else:
        # Find all house directories in the base_dir
        house_dirs = sorted(glob.glob(os.path.join(base_dir, "*")))
    
    # Filter house directories based on start_house and end_house if provided
    if start_house is not None or end_house is not None:
        filtered_house_dirs = []
        for house_dir in house_dirs:
            house_id = os.path.basename(house_dir)
            
            # Skip if not a directory or doesn't look like a house ID
            if not os.path.isdir(house_dir) or not house_id.isdigit():
                continue
                
            # Check if house_id is within the specified range
            if start_house is not None and house_id < start_house:
                continue
            if end_house is not None and house_id > end_house:
                continue
                
            filtered_house_dirs.append(house_dir)
        
        house_dirs = filtered_house_dirs
        print(f"Processing {len(house_dirs)} houses in range: {start_house or 'beginning'} to {end_house or 'end'}")
    
    # --- Generation targets and counters ---
    TARGETS = {
        'single_room_per_type': {
            'living room': 10,
            'kitchen': 10,
            'bedroom': 10,
            'bathroom': 10,
        },
        'cross_room': 20,
        'length_bins': {
            'short': 15,
            'med': 15,
            'long': 15
        }
    }

    counters = {
        'single_room': {k: 0 for k in TARGETS['single_room_per_type'].keys()},
        'cross_room': 0,
        'length_bins': {k: 0 for k in TARGETS['length_bins'].keys()},
        'total_valid_episodes': 0
    }

    def _update_counters_from_metadata(metadata):
        # metadata expected to have 'room_annotation' and 'length_annotation'
        ra = metadata.get('room_annotation', '')
        la = metadata.get('length_annotation', '')

        if ra.startswith('single-room'):
            # format: 'single-room: kitchen'
            parts = ra.split(':')
            if len(parts) > 1:
                room = parts[1].strip()
                if room in counters['single_room']:
                    counters['single_room'][room] += 1
        elif ra == 'cross-room':
            counters['cross_room'] += 1

        if la in counters['length_bins']:
            counters['length_bins'][la] += 1

        counters['total_valid_episodes'] += 1

    def _targets_met():
        # Check single room per type
        for room, tgt in TARGETS['single_room_per_type'].items():
            if counters['single_room'].get(room, 0) < tgt:
                return False
        if counters['cross_room'] < TARGETS['cross_room']:
            return False
        for b, tgt in TARGETS['length_bins'].items():
            if counters['length_bins'].get(b, 0) < tgt:
                return False
        return True

    # If there are already generated episodes in the output base dir, initialize counters from them
    if os.path.exists(output_base_dir):
        for root, dirs, files in os.walk(output_base_dir):
            if 'trajectory_metadata.json' in files:
                try:
                    meta_path = os.path.join(root, 'trajectory_metadata.json')
                    with open(meta_path, 'r') as mf:
                        meta = json.load(mf)
                        _update_counters_from_metadata(meta)
                except Exception:
                    continue

    print(f"Generation targets:")
    print(json.dumps(TARGETS, indent=2))
    print(f"Starting counters (from existing outputs if any): {counters}")
    
    # Process each house
    for house_dir in tqdm(house_dirs, desc="Processing houses"):
        # If targets met, break out early
        if _targets_met():
            print("All generation targets satisfied. Stopping processing.")
            break

        house_id = os.path.basename(house_dir)
        
        # Skip if not a directory or doesn't look like a house ID
        if not os.path.isdir(house_dir) or not house_id.isdigit():
            continue
        
        # Skip if this house is already fully processed
        if house_id in fully_processed_houses and resume:
            print(f"Skipping house {house_id} - already fully processed")
            continue
        
        house_index = int(house_id)
        hdf5_path = os.path.join(house_dir, "hdf5_sensors.hdf5")
        
        # Skip if no HDF5 file exists
        if not os.path.exists(hdf5_path):
            summary['errors'].append(f"No HDF5 file found for house {house_id}")
            save_progress()
            continue
        
        print(f"\nProcessing house {house_id} at {hdf5_path}")
        
        # Find if we already have a summary for this house
        existing_house_summary = None
        for house in summary['processed_houses']:
            if house['house_id'] == house_id:
                existing_house_summary = house
                break
        
        # If no existing summary for this house, create one
        if existing_house_summary is None:
            house_summary = {
                'house_id': house_id,
                'episodes': [],
                'errors': []
            }
        else:
            house_summary = existing_house_summary
        
        # Load house data
        house_data = load_house_from_prior(house_index)
        if not house_data:
            error_msg = f"Failed to load house data for house {house_id}"
            house_summary['errors'].append(error_msg)
            summary['errors'].append(error_msg)
            
            # Only add house to summary if it's not already there
            if existing_house_summary is None:
                summary['processed_houses'].append(house_summary)
            
            save_progress()
            continue
        
        # Get list of episodes in this house's HDF5 file
        try:
            with h5py.File(hdf5_path, 'r') as f:
                # Find all episode indices (keys in the root of the HDF5 file)
                episode_indices = []
                for key in f.keys():
                    if key.isdigit() or (key.startswith('0') and key[1:].isdigit()):
                        episode_indices.append(key)
                
                # Sort them numerically
                episode_indices.sort(key=lambda x: int(x))
        except Exception as e:
            error_msg = f"Error reading episodes from {hdf5_path}: {str(e)}"
            print(error_msg)
            house_summary['errors'].append(error_msg)
            summary['errors'].append(error_msg)
            
            # Only add house to summary if it's not already there
            if existing_house_summary is None:
                summary['processed_houses'].append(house_summary)
            
            save_progress()
            continue
        
        # Process each episode for this house
        for episode_idx in tqdm(episode_indices[:1], desc=f"Processing episodes for house {house_id}"):
            # Check if this episode is already processed
            if resume and f"{house_id}_{episode_idx}" in processed_items:
                print(f"  Skipping episode {episode_idx} - already processed")
                continue
            
            try:
                # Create episode-specific output directory
                episode_output_dir = os.path.join(output_base_dir, f"{outdir_prefix}house_{house_id}_episode_{episode_idx}")
                os.makedirs(episode_output_dir, exist_ok=True)
                
                # Check if this episode is already fully processed
                trajectory_metadata_path = os.path.join(episode_output_dir, "trajectory_metadata.json")
                videos_path = os.path.join(episode_output_dir, "videos")
                
                if resume and os.path.exists(trajectory_metadata_path) and os.path.exists(videos_path):
                    try:
                        with open(trajectory_metadata_path, 'r') as f:
                            metadata = json.load(f)
                        
                        # Check if videos for all modalities exist
                        all_videos_exist = True
                        for modality, video_name in [
                            ("rgb", "rgb_trajectory.mp4"),
                            ("depth", "depth_trajectory.mp4"), 
                            ("semantic", "semantic_trajectory.mp4")
                        ]:
                            if not os.path.exists(os.path.join(videos_path, video_name)):
                                all_videos_exist = False
                                break
                        
                        if all_videos_exist and metadata.get('frames', 0) > 0:
                            print(f"  Skipping episode {episode_idx} - already fully processed")
                            
                            # Add to the house summary if not already there
                            exists_in_summary = False
                            for ep in house_summary.get('episodes', []):
                                if ep['episode_idx'] == episode_idx:
                                    exists_in_summary = True
                                    break
                            
                            if not exists_in_summary:
                                house_summary['episodes'].append({
                                    'episode_idx': episode_idx,
                                    'frame_count': metadata.get('frames', 0),
                                    'output_dir': episode_output_dir
                                })
                                summary['total_episodes'] += 1
                            
                            processed_items.add(f"{house_id}_{episode_idx}")
                            save_progress()
                            continue
                    except Exception as e:
                        print(f"  Error checking if episode {episode_idx} is fully processed: {str(e)}")
                        # Continue with processing this episode
                
                print(f"  Processing episode {episode_idx} for house {house_id}")
                
                # Load trajectory data for this episode
                with h5py.File(hdf5_path, 'r') as f:
                    trajectory_data = {}
                    
                    # Extract house index for reference
                    if f'{episode_idx}/house_index' in f:
                        trajectory_data['house_index'] = f[f'{episode_idx}/house_index'][0]
                    
                    # Extract positions and rotations
                    if f'{episode_idx}/last_agent_location' in f:
                        trajectory_data['positions'] = []
                        trajectory_data['rotations'] = []
                        
                        locations = f[f'{episode_idx}/last_agent_location'][:]
                        for loc in locations:
                            if len(loc) >= 6:
                                position = {'x': float(loc[0]), 'y': float(loc[1]), 'z': float(loc[2])}
                                rotation = {'x': 0.0, 'y': float(loc[4]), 'z': 0.0}  # AI2-THOR uses y for yaw
                                trajectory_data['positions'].append(position)
                                trajectory_data['rotations'].append(rotation)
                    
                    # Extract actions
                    if f'{episode_idx}/last_action_str' in f:
                        actions = []
                        action_data = f[f'{episode_idx}/last_action_str'][:]
                        
                        for action_bytes in action_data:
                            # Convert bytes to string
                            action_str = ''
                            for b in action_bytes:
                                if b == 0:  # End of string
                                    break
                                action_str += chr(b)
                            actions.append(action_str)
                        
                        trajectory_data['actions'] = actions
                    
                    # Extract object IDs if available
                    if f'{episode_idx}/nav_task_relevant_object_bbox/oids_as_bytes' in f:
                        object_ids = []
                        object_bytes = f[f'{episode_idx}/nav_task_relevant_object_bbox/oids_as_bytes'][:]
                        
                        for obj_bytes in object_bytes:
                            # Convert bytes to string
                            obj_str = ''
                            for b in obj_bytes:
                                if b == 0:  # End of string
                                    break
                                obj_str += chr(b)
                            
                            # Clean up any JSON formatting
                            obj_str = obj_str.replace('[', '').replace(']', '').replace('"', '')
                            object_ids.append(obj_str)
                        
                        trajectory_data['object_ids'] = object_ids
                
                # Replay the trajectory with all modalities
                frame_info = replay_trajectory_with_modalities(house_id, house_data, trajectory_data, episode_output_dir)
                
                # Create videos if frames were generated
                if frame_info['frame_count'] > 0:
                    videos_dir = os.path.join(episode_output_dir, "videos")
                    os.makedirs(videos_dir, exist_ok=True)
                    
                    # Create videos for each modality
                    modality_dirs = [
                        ("rgb", "rgb_trajectory.mp4"),
                        ("depth", "depth_trajectory.mp4"),
                        ("semantic", "semantic_trajectory.mp4"),
                        # ("top_down", "top_down_view.mp4")
                    ]
                    
                    for modality_dir, video_name in modality_dirs:
                        dir_path = os.path.join(episode_output_dir, modality_dir)
                        if os.path.exists(dir_path) and os.listdir(dir_path):
                            create_video_from_frames(
                                dir_path, 
                                os.path.join(videos_dir, video_name)
                            )
                
                # Add episode to house summary if not already there
                exists_in_summary = False
                for ep in house_summary.get('episodes', []):
                    if ep['episode_idx'] == episode_idx:
                        # Update existing entry
                        ep['frame_count'] = frame_info.get('frame_count', 0)
                        ep['output_dir'] = episode_output_dir
                        exists_in_summary = True
                        break
                
                if not exists_in_summary:
                    house_summary['episodes'].append({
                        'episode_idx': episode_idx,
                        'frame_count': frame_info.get('frame_count', 0),
                        'output_dir': episode_output_dir
                    })
                    summary['total_episodes'] += 1
                
                # Update counters based on saved trajectory metadata (if present)
                meta_file = os.path.join(episode_output_dir, 'trajectory_metadata.json')
                if os.path.exists(meta_file):
                    try:
                        with open(meta_file, 'r') as mf:
                            meta = json.load(mf)
                            _update_counters_from_metadata(meta)
                    except Exception as e:
                        print(f"Warning: could not read metadata for counters: {e}")

                print(f"Updated counters: {counters}")

                # Check if we've met all targets and stop early if so
                if _targets_met():
                    print("All generation targets satisfied. Saving summary and stopping processing.")
                    # Save summary and progress
                    with open(summary_path, 'w') as f:
                        json.dump(summary, f, indent=2)
                    save_progress()
                    # Set processed flag for this episode and break the loops
                    processed_items.add(f"{house_id}_{episode_idx}")
                    all_done = True
                else:
                    all_done = False
                
                print(f"  Completed episode {episode_idx} with {frame_info.get('frame_count', 0)} frames")
                
            except Exception as e:
                error_msg = f"Error processing episode {episode_idx} for house {house_id}: {str(e)}"
                print(error_msg)
                import traceback
                traceback.print_exc()
                house_summary['errors'].append(error_msg)
                summary['errors'].append(error_msg)
                save_progress()
        
        # Only add house to summary if it's not already there
        if existing_house_summary is None:
            summary['processed_houses'].append(house_summary)
            summary['total_houses'] += 1
        
        # Save summary after each house
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Completed house {house_id} with {len(house_summary['episodes'])} episodes")
    
    # Final save of the summary
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Processing complete! Summary saved to {summary_path}")
    print(f"Processed {summary['total_houses']} houses and {summary['total_episodes']} episodes")
    if summary['errors']:
        print(f"Encountered {len(summary['errors'])} errors during processing")
    
    return summary

def process_one_episode(house_index=0, episode_idx="0", hdf5_path=None):
    """
    Function to process episodes and reconstruct their trajectories
    
    Parameters:
    house_index (int): Index of the house to load from prior dataset (default: 0)
    episode_idx (str): Episode index to process (default: "0")
    hdf5_path (str): Optional path to a specific HDF5 file for testing
    """
    # Define paths
    base_hdf5_dir = "path/to/hdf5/train"
    
    # Load house directly from prior dataset
    house_data = load_house_from_prior(house_index)
    if not house_data:
        raise ValueError(f"Failed to load house at index {house_index}")
    
    # Use the house ID if available, otherwise use the index
    house_id = house_data.get('id', f"index_{house_index}")
    
    # Use the provided HDF5 path if given, otherwise construct from house_id
    if hdf5_path is None:
        hdf5_path = os.path.join(base_hdf5_dir, house_id, "hdf5_sensors.hdf5")
        if not os.path.exists(hdf5_path):
            # Try a fallback path if the exact house ID path doesn't exist
            print(f"HDF5 file not found at {hdf5_path}")
            # Look for HDF5 files in the train directory
            hdf5_files = []
            for root, dirs, files in os.walk(base_hdf5_dir):
                if "hdf5_sensors.hdf5" in files:
                    hdf5_files.append(os.path.join(root, "hdf5_sensors.hdf5"))
            
            if hdf5_files:
                # Use the first available HDF5 file
                hdf5_path = hdf5_files[0]
                print(f"Using fallback HDF5 file: {hdf5_path}")
            else:
                raise ValueError("No HDF5 files found in the train directory")
    else:
        print(f"Using provided HDF5 file: {hdf5_path}")
    
    # Create output directory
    output_dir = f"spoc_reconstructed_trajectories/house_{house_id}_episode_{episode_idx}"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Starting trajectory reconstruction process for house index {house_index}, episode {episode_idx}...")
    
    # Load trajectory data
    print(f"Loading trajectory data from {hdf5_path}...")
    with h5py.File(hdf5_path, 'r') as f:
        trajectory_data = {}
        
        # Check if the episode exists
        if episode_idx not in f:
            raise ValueError(f"Episode {episode_idx} not found in HDF5 file {hdf5_path}")
        
        # Extract house index for reference
        if f'{episode_idx}/house_index' in f:
            trajectory_data['house_index'] = f[f'{episode_idx}/house_index'][0]
        
        # Extract positions and rotations
        if f'{episode_idx}/last_agent_location' in f:
            trajectory_data['positions'] = []
            trajectory_data['rotations'] = []
            
            locations = f[f'{episode_idx}/last_agent_location'][:]
            for loc in locations:
                if len(loc) >= 6:
                    position = {'x': float(loc[0]), 'y': float(loc[1]), 'z': float(loc[2])}
                    rotation = {'x': 0.0, 'y': float(loc[4]), 'z': 0.0}  # AI2-THOR uses y for yaw
                    trajectory_data['positions'].append(position)
                    trajectory_data['rotations'].append(rotation)
            
            print(f"Extracted {len(trajectory_data['positions'])} position/rotation pairs")
        
        # Extract actions
        if f'{episode_idx}/last_action_str' in f:
            actions = []
            action_data = f[f'{episode_idx}/last_action_str'][:]
            
            for action_bytes in action_data:
                # Convert bytes to string
                action_str = ''
                for b in action_bytes:
                    if b == 0:  # End of string
                        break
                    action_str += chr(b)
                actions.append(action_str)
            
            trajectory_data['actions'] = actions
            print(f"Extracted {len(actions)} actions")
        
        # Extract object IDs if available
        if f'{episode_idx}/nav_task_relevant_object_bbox/oids_as_bytes' in f:
            object_ids = []
            object_bytes = f[f'{episode_idx}/nav_task_relevant_object_bbox/oids_as_bytes'][:]
            
            for obj_bytes in object_bytes:
                # Convert bytes to string
                obj_str = ''
                for b in obj_bytes:
                    if b == 0:  # End of string
                        break
                    obj_str += chr(b)
                
                # Clean up any JSON formatting
                obj_str = obj_str.replace('[', '').replace(']', '').replace('"', '')
                object_ids.append(obj_str)
            
            trajectory_data['object_ids'] = object_ids
            print(f"Extracted {len(object_ids)} object IDs")
    
    # Add a JSON encoder class for numpy types
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super(NumpyEncoder, self).default(obj)
    
    # Replay the complete trajectory with RGB, Depth, and Semantic modalities
    print("Replaying complete trajectory with all modalities...")
    frame_info = replay_trajectory_with_modalities(house_id, house_data, trajectory_data, output_dir)
    
    # Create videos from the generated frames
    if frame_info['frame_count'] > 0:
        print("Creating videos from frames...")
        videos_dir = os.path.join(output_dir, "videos")
        os.makedirs(videos_dir, exist_ok=True)
        
        # Create videos for each modality
        modality_dirs = [
            ("rgb", "rgb_trajectory.mp4"),
            ("depth", "depth_trajectory.mp4"),
            ("semantic", "semantic_trajectory.mp4"),
        ]
        
        for modality_dir, video_name in modality_dirs:
            dir_path = os.path.join(output_dir, modality_dir)
            if os.path.exists(dir_path) and os.listdir(dir_path):
                create_video_from_frames(
                    dir_path, 
                    os.path.join(videos_dir, video_name)
                )
    
    print(f"Reconstruction complete! Generated {frame_info['frame_count']} frames.")
    print(f"Output saved to {output_dir}")
    
    if frame_info['frame_count'] > 0:
        print(f"Videos saved to {videos_dir}")
    
    # Return information about the reconstruction
    return {
        'house_id': house_id,
        'episode_idx': episode_idx,
        'frame_count': frame_info.get('frame_count', 0),
        'output_dir': output_dir,
        'videos_dir': videos_dir if frame_info.get('frame_count', 0) > 0 else None
    }

# Update the argument parser in the main section
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Reconstruct SPOC trajectories with additional modalities')
    parser.add_argument('--process_all', action='store_true', help='Process all houses and episodes')
    parser.add_argument('--base_dir', type=str, 
                       default="/home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/spoc/data/all/ObjectNavType/val",
                       help='Base directory containing house folders')
    parser.add_argument('--output_dir', type=str, 
                       default="data/spoc_trajectories_val",
                       help='Base directory for output')
    
    # Add start and end house arguments
    parser.add_argument('--start_house', type=str, help='House ID to start processing from', default=None)
    parser.add_argument('--end_house', type=str, help='House ID to end processing at (inclusive)', default=None)
    
    # Add houses file argument
    parser.add_argument('--houses_file', type=str, help='Path to a text file containing house directories to process', default=None)
    
    # Add outdir_prefix argument
    parser.add_argument('--outdir_prefix', type=str, help='Prefix for output directory names', default="")
    
    # Keep original arguments for single-file processing
    parser.add_argument('--house_index', type=int, help='Index of a specific house to load (0-based)', default=None)
    parser.add_argument('--episode_idx', type=str, help='Specific episode index to process', default="0")
    parser.add_argument('--hdf5_path', type=str, help='Path to a specific HDF5 file for testing', default=None)
    
    args = parser.parse_args()
    
    if args.process_all:
        # Process houses within the specified range
        process_all_houses_and_episodes(
            args.base_dir, 
            args.output_dir,
            start_house=args.start_house,
            end_house=args.end_house,
            houses_file=args.houses_file,
            outdir_prefix=args.outdir_prefix
        )
    else:
        # Process a single house/episode using the original function
        if args.house_index is None:
            print("ERROR: When not using --process_all, you must specify --house_index")
            exit(1)

        process_one_episode(house_index=args.house_index, episode_idx=args.episode_idx, hdf5_path=args.hdf5_path)