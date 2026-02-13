import numpy as np
from omegaconf import DictConfig, OmegaConf
from environment.stretch_controller import StretchController
from spoc_utils.constants.stretch_initialization_utils import STRETCH_ENV_ARGS
from embodied.object_searching_task_manager import ObjectSearchingTaskManager
from scipy.spatial.transform import Rotation as R
from spoc_utils.embodied_utils import square_image, distance_traveled_step


class EnvInterface:
    ROTATION_STEP = 5.0 # degrees
    MOVE_STEP = 0.25 # meters
    ACTION_MAP = {
        # Navigation actions
        'm': ('MoveAhead', {'moveMagnitude': MOVE_STEP}),
        'b': ('MoveBack', {'moveMagnitude': MOVE_STEP}),
        'l': ('RotateLeft', {'degrees': ROTATION_STEP}),
        'r': ('RotateRight', {'degrees': ROTATION_STEP}),
    }

    def __init__(self, config: DictConfig, stretch_controller: StretchController, task_manager: ObjectSearchingTaskManager):
        self.stretch_controller = stretch_controller
        self.task_manager = task_manager
        self.config = config
        self.save_trajectory = config.trajectory.save
        self.trajectory_save_path = config.trajectory.save_path
        self.trajectory_image_keys = config.trajectory.image_keys
        self.trajectory_string_keys = config.trajectory.string_keys
        self._image_buffer = {key: [] for key in self.trajectory_image_keys} if self.save_trajectory else None
        self._string_buffer = {key: [] for key in self.trajectory_string_keys} if self.save_trajectory else None

    @property
    def action_space(self):
        return list(self.ACTION_MAP.keys())
    
    @property
    def image_buffer(self):
        return self._image_buffer

    @property
    def string_buffer(self):
        return self._string_buffer

    def reset(self, idx=None):
        self.task_manager.reset(idx)
        self._image_buffer = {key: [] for key in self.trajectory_image_keys} if self.save_trajectory else None
        self._string_buffer = {key: [] for key in self.trajectory_string_keys} if self.save_trajectory else None

    def get_observation(self): # TODO check posse conversion
        # get RGB
        rgb_image = square_image(self.stretch_controller.navigation_camera)
        # get depth
        depth_image = square_image(self.stretch_controller.navigation_depth_frame)
        # get camera pose in world coordinates
        event = self.stretch_controller.controller.last_event
        pos = event.metadata["cameraPosition"]                      # dict x,y,z (world)
        agent = event.metadata["agent"]
        yaw = agent["rotation"]["y"]                 # degrees (around Y)
        pitch = agent["cameraHorizon"]

        R_aw = R.from_euler('xyz', [0.0, yaw, 0.0], degrees=True).as_matrix() # R_aw * R_ca = R_cw
        R_ca = R.from_euler('xyz', [pitch, 0.0, 0.0], degrees=True).as_matrix()
        R_cw = R_aw @ R_ca
        t_cw = np.array([pos["x"], pos["y"], pos["z"]], dtype=float)
        pose = np.eye(4, dtype=float)
        pose[:3,:3] = R_cw
        pose[:3, 3] = t_cw
        F = np.diag([1, 1, -1, 1])
        pose =  F @ pose @ F  # left-handed to right-handed
        observation = {
            "rgb": rgb_image,
            "depth": depth_image,
            "pose": pose,
            "position": t_cw,
            "rotation": R_cw,
        }
        return observation

    def navigate_to(self, target_pose: np.ndarray):
        ## DEBUG here, target_pose is the agent pose in world coordinates
        print("Navigating to:", target_pose)
        # == find the closest navigable point to target_pose ==
        target_position = target_pose[:3, 3]
        reachable_positions = self.stretch_controller.controller.step(
            action="GetReachablePositions"
        ).metadata["actionReturn"] # list[dict[str, float]] like [dict(x=(...), y=(...), z=(...)) ,...]
        reachable_positions = np.array([[p["x"], p["y"], p["z"]] for p in reachable_positions], dtype=float) # (N, 3)
        dists = np.linalg.norm(reachable_positions - target_position[None, :], axis=1) # (N,)
        closest_position = reachable_positions[np.argmin(dists), :]
        # face to the target position, first find angle between the current forward and the vector from current to target
        current_event = self.stretch_controller.controller.last_event
        current_agent = current_event.metadata["agent"]
        current_yaw = current_agent["rotation"]["y"] # degrees
        current_observation = self.get_observation()
        current_position = current_observation["position"]
        current_forward = current_observation["rotation"][:, 2] # (3,)
        to_target = closest_position - current_position
        to_target[1] = 0.0 # project to the ground plane
        to_target = to_target / np.linalg.norm(to_target)
        dot = np.clip(np.dot(current_forward, to_target), -1.0, 1.0)
        angle = np.arccos(dot) # radians, in [0, pi]
        cross = np.cross(current_forward, to_target)
        if cross[1] < 0: # right-handed coordinate system, Y is up
            angle = -angle
        angle = np.degrees(angle) # degrees, in [-180, 180]
        face_to_yaw = (current_yaw + angle) % 360 # degrees, in [0, 360]

        # == rotate to face to the target position ==
        yaw_diff = (face_to_yaw - current_yaw + 180) % 360 - 180 # degrees, in [-180, 180]
        while abs(yaw_diff) > self.ROTATION_STEP:
            if yaw_diff > 0:
                action_name = "RotateRight"
                action_args = self.ACTION_MAP['r'][1]
                self.stretch_controller.controller.step(action=action_name, **action_args)
            else:
                action_name = "RotateLeft"
                action_args = self.ACTION_MAP['l'][1]
                self.stretch_controller.controller.step(action=action_name, **action_args)
            current_event = self.stretch_controller.controller.last_event
            current_agent = current_event.metadata["agent"]
            current_yaw = current_agent["rotation"]["y"] # degrees
            yaw_diff = (face_to_yaw - current_yaw + 180) % 360 - 180 # degrees, in [-180, 180]
            self._step_trajectory_logger(action_name, action_args)

        # == move to the closest position ==
        current_position = np.array([
            current_agent["position"]["x"],
            current_agent["position"]["y"],
            current_agent["position"]["z"]
        ], dtype=float)
        while np.linalg.norm(current_position - closest_position) > self.MOVE_STEP:
            self.stretch_controller.controller.step(action="MoveAhead", **self.ACTION_MAP['m'][1])
            current_event = self.stretch_controller.controller.last_event
            current_agent = current_event.metadata["agent"]
            current_position = np.array([
                current_agent["position"]["x"],
                current_agent["position"]["y"],
                current_agent["position"]["z"]
            ], dtype=float)
            self._step_trajectory_logger("MoveAhead", self.ACTION_MAP['m'][1])

        # == move to the target rotation ==
        current_event = self.stretch_controller.controller.last_event
        current_agent = current_event.metadata["agent"]
        current_yaw = current_agent["rotation"]["y"] # degrees
        target_rotation = target_pose[:3, :3]
        # use scipy to convert rotation matrix to yaw, assume euler order is 'xyz' and x, z are zeros
        # target_yaw = R.from_matrix(target_rotation).as_euler('xyz', degrees=True)[1] % 360
        target_yaw = np.degrees(np.arctan2(target_rotation[0, 2], target_rotation[2, 2])) % 360
        yaw_diff = (target_yaw - current_yaw + 180) % 360 - 180 # degrees, in [-180, 180]
        while abs(yaw_diff) > self.ROTATION_STEP:
            if yaw_diff > 0:
                action_name = "RotateRight"
                action_args = self.ACTION_MAP['r'][1]
                self.stretch_controller.controller.step(action=action_name, **action_args)
            else:
                action_name = "RotateLeft"
                action_args = self.ACTION_MAP['l'][1]
                self.stretch_controller.controller.step(action=action_name, **action_args)
            current_event = self.stretch_controller.controller.last_event
            current_agent = current_event.metadata["agent"]
            current_yaw = current_agent["rotation"]["y"] # degrees
            yaw_diff = (target_yaw - current_yaw + 180) % 360 - 180 # degrees, in [-180, 180]
            self._step_trajectory_logger(action_name, action_args)

    def _step_trajectory_logger(self, action_name, action_args):
        distance_traveled, angle_turned = distance_traveled_step(action_name, action_args)
        self.task_manager._distance_traveled += distance_traveled
        self.task_manager._angle_turned += angle_turned
        self.task_manager._current_step += 1
        if self.save_trajectory:
            obs = self.get_observation()
            image_dict = {key: obs[key] for key in self.trajectory_image_keys if key in obs}
            string_dict = {key: obs[key] for key in self.trajectory_string_keys if key in obs}
            if 'action' in self.trajectory_string_keys:
                string_dict['action'] = (action_name, action_args)
            if 'distance_to_goal' in self.trajectory_string_keys:
                string_dict['distance_to_goal'] = self.task_manager.distance_to_goal
            if 'distance_traveled' in self.trajectory_string_keys:
                string_dict['distance_traveled'] = self.task_manager.distance_traveled
            if 'angle_turned' in self.trajectory_string_keys:
                string_dict['angle_turned'] = self.task_manager.angle_turned
            for key, value in image_dict.items():
                self._image_buffer[key].append(value)
            for key, value in string_dict.items():
                self._string_buffer[key].append(value.tolist() if isinstance(value, np.ndarray) else value)
    
    @staticmethod
    def pose_belief2spoc(pose_belief: np.ndarray):
        pass
        