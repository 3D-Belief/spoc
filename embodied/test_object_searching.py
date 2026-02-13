import sys
import os
import random
import pathlib
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import imageio
from copy import deepcopy
from pathlib import Path
import math
import json
import time
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from typing import Any, Dict
from omegaconf import DictConfig, OmegaConf
import jsonlines
from scipy.spatial.transform import Rotation as R
from hydra import initialize_config_dir, compose

ROOT_DIR = str(Path(__file__).parent.parent)
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

# Set environment variables
os.environ["OBJAVERSE_DATA_DIR"] = "/home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/spoc/data"
os.environ["OBJAVERSE_HOUSES_DIR"] = "/home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/spoc/data/houses_2023_07_28"

from environment.stretch_controller import StretchController
from spoc_utils.constants.stretch_initialization_utils import STRETCH_ENV_ARGS
from spoc_utils.constants.objaverse_data_dirs import OBJAVERSE_HOUSES_DIR
from embodied.env_interface import EnvInterface
from embodied.object_searching_task_manager import ObjectSearchingTaskManager

def run_object_searching():
    # initialize the Stretch controller
    stretch_controller = StretchController(**STRETCH_ENV_ARGS)
    # load task manager config using hydra
    with initialize_config_dir(config_dir=str(pathlib.Path(__file__).parent.parent / "configuration")):
        embodied_config = compose(config_name="object_searching")
    task_manager = ObjectSearchingTaskManager(config=embodied_config, stretch_controller=stretch_controller)
    env_interface = EnvInterface(config=embodied_config, stretch_controller=stretch_controller, task_manager=task_manager)
    # extract some info
    save_path = embodied_config.trajectory.save_path
    print(f"Loaded {len(task_manager.episodes)} episodes from {task_manager.episode_root}")
    print(f"Max steps per episode: {task_manager.max_steps}")

    env_interface.reset()
    # dummy target pose as 4x4 np array
    target_pose = np.array([[1, 0, 0, 1.20],
                            [0, 1, 0, 0.90],
                            [0, 0, 1, 10.50],
                            [0, 0, 0, 1]], dtype=np.float32)

    env_interface.navigate_to(target_pose)

    # save navigation video
    image_buffer = env_interface.image_buffer
    string_buffer = env_interface.string_buffer
    # save images as a video
    video_path = os.path.join(save_path, f"episode_{task_manager.current_episode_index}_navigation.mp4")
    imageio.mimwrite(video_path, image_buffer['rgb'], fps=10, quality=8)
    print(f"Saved navigation video to {video_path}")
    # save string buffer as a jsonl file
    jsonl_path = os.path.join(save_path, f"episode_{task_manager.current_episode_index}_navigation.jsonl")
    with jsonlines.open(jsonl_path, mode='w') as writer:
        for i in range(len(string_buffer['action'])):
            entry = {key: string_buffer[key][i] for key in string_buffer}
            writer.write(entry)
    print(f"Saved navigation log to {jsonl_path}")

    # setup final log
    final_log = task_manager.get_final_log()
    # dump final log
    with open(os.path.join(save_path, f"final_log.json"), "w") as f:
        json.dump(final_log, f, indent=4)

if __name__ == "__main__":
    run_object_searching()