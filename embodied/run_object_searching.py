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
from hydra import initialize_config_dir, compose
from scipy.spatial.transform import Rotation as R

from agents.vlm_agent import VLMAgent
from pixelbelief.belief_agent import BeliefAgent, prepare_video
from pixelbelief.occupancy import OccupancyMap
from pixelsplat.ply_export import export_gaussians_to_ply

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

def create_save_folders(run_dir: str, prefix: str):
    save_folder_sample = os.path.join(run_dir, prefix)
    os.makedirs(
        save_folder_sample, exist_ok=True,
    )
    save_folder_nav_video = os.path.join(save_folder_sample, f'nav_video')
    os.makedirs(
        save_folder_nav_video, exist_ok=True,
    )
    save_folder_observation = os.path.join(save_folder_sample, f'observation')
    os.makedirs(
        save_folder_observation, exist_ok=True,
    )
    save_folder_planning = os.path.join(save_folder_sample, f'planning')
    os.makedirs(
        save_folder_planning, exist_ok=True,
    )
    save_folder_obs = os.path.join(save_folder_observation, f'obs_frames')
    os.makedirs(
        save_folder_obs, exist_ok=True,
    )
    save_folder_obs_obs_map = os.path.join(save_folder_observation, f'obs_maps')
    os.makedirs(
        save_folder_obs_obs_map, exist_ok=True,
    )
    save_folder_obs_height_map = os.path.join(save_folder_observation, f'height_maps')
    os.makedirs(
        save_folder_obs_height_map, exist_ok=True,
    )
    save_folder_height_map = os.path.join(save_folder_planning, f'height_map')
    os.makedirs(
        save_folder_height_map, exist_ok=True,
    )
    save_folder_imagine = os.path.join(save_folder_planning, f'imagined_frames')
    os.makedirs(
        save_folder_imagine, exist_ok=True,
    )
    save_folder_obs_map = os.path.join(save_folder_planning, f'obs_map')
    os.makedirs(
        save_folder_obs_map, exist_ok=True,
    )

    all_folders = {
        "save_folder_sample": save_folder_sample,
        "save_folder_nav_video": save_folder_nav_video,
        "save_folder_observation": save_folder_observation,
        "save_folder_planning": save_folder_planning,
        "save_folder_obs": save_folder_obs,
        "save_folder_obs_obs_map": save_folder_obs_obs_map,
        "save_folder_obs_height_map": save_folder_obs_height_map,
        "save_folder_height_map": save_folder_height_map,
        "save_folder_imagine": save_folder_imagine,
        "save_folder_obs_map": save_folder_obs_map,
    }
    return all_folders

def create_step_visualization(visuals, step, save_path, prev_img=None, target_obj=None):
    """
    Create a visualization that combines multiple visual elements.
    
    Args:
        visuals: Dictionary with visual elements:
            - visual_0: observation image
            - visual_1: imagined path image
            - visual_2: list of imagined frames
            - visual_2_scores: list of scores for imagined frames
            - visual_3: path image
        step: Current step number
        save_path: Path to save the visualization
        prev_img: Previous visualization to append to (if not the first step)
        target_obj: Name of the target object being searched for
    """
    # Convert PIL images to numpy arrays if needed
    for key in ['visual_0', 'visual_1', 'visual_3']:
        if key in visuals and isinstance(visuals[key], Image.Image):
            visuals[key] = np.array(visuals[key])
    
    # Calculate how many frames in visual_2
    num_frames = len(visuals['visual_2']) if 'visual_2' in visuals else 0
    total_cols = 3 + num_frames  # visual_0, visual_1, visual_2 (multiple frames), visual_3
    
    # Create a new figure for this step
    fig = plt.figure(figsize=(20, 5))
    
    # Add step number as a title for the entire row (inside the bounding box)
    fig.suptitle(f"Step {step}", fontsize=20, y=0.95)
    
    # Define grid - we'll create a special layout to show visuals in order 0,1,2,3
    if num_frames > 0:
        # Create a grid with proper proportions for the visual_2 frames
        gs = GridSpec(1, total_cols)
        
        # Get reference height from visual_0 or first frame of visual_2
        ref_height = visuals['visual_0'].shape[0] if 'visual_0' in visuals else visuals['visual_2'][0].shape[0]
        
        # Resize visual_1 and visual_3 to match reference height
        if 'visual_1' in visuals and visuals['visual_1'] is not None:
            h, w = visuals['visual_1'].shape[:2]
            new_w = int(w * (ref_height / h))
            visuals['visual_1'] = np.array(Image.fromarray(visuals['visual_1']).resize((new_w, ref_height)))
            
        if 'visual_3' in visuals and visuals['visual_3'] is not None:
            h, w = visuals['visual_3'].shape[:2]
            new_w = int(w * (ref_height / h))
            visuals['visual_3'] = np.array(Image.fromarray(visuals['visual_3']).resize((new_w, ref_height)))
        
        # Add visual_0
        ax0 = fig.add_subplot(gs[0, 0])
        ax0.imshow(visuals['visual_0'])
        ax0.set_title('Obs', fontsize=18)
        ax0.axis('off')
        
        # Add visual_1
        ax1 = fig.add_subplot(gs[0, 1])
        ax1.imshow(visuals['visual_1'])
        ax1.set_title('Imagined Path', fontsize=18)
        ax1.axis('off')
        
        # Add visual_2 (multiple frames)
        for i, frame in enumerate(visuals['visual_2']):
            ax = fig.add_subplot(gs[0, 2 + i])
            ax.imshow(frame)
            
            # Add score as subtitle if available
            if 'visual_2_scores' in visuals and i < len(visuals['visual_2_scores']):
                score = visuals['visual_2_scores'][i]
                title = f'Imag. Score: {score}'
            else:
                title = 'Imag. Frames' if i == 0 else ''
                
            ax.set_title(title, fontsize=18)
            ax.axis('off')
        
        # Add visual_3
        ax3 = fig.add_subplot(gs[0, total_cols - 1])
        ax3.imshow(visuals['visual_3'])
        ax3.set_title('Path', fontsize=18)
        ax3.axis('off')
        
    else:
        # If no frames in visual_2, just show the other visuals
        gs = GridSpec(1, 3)
        
        # Get reference height from visual_0
        if 'visual_0' in visuals and visuals['visual_0'] is not None:
            ref_height = visuals['visual_0'].shape[0]
            
            # Resize visual_1 and visual_3 to match reference height
            if 'visual_1' in visuals and visuals['visual_1'] is not None:
                h, w = visuals['visual_1'].shape[:2]
                new_w = int(w * (ref_height / h))
                visuals['visual_1'] = np.array(Image.fromarray(visuals['visual_1']).resize((new_w, ref_height)))
                
            if 'visual_3' in visuals and visuals['visual_3'] is not None:
                h, w = visuals['visual_3'].shape[:2]
                new_w = int(w * (ref_height / h))
                visuals['visual_3'] = np.array(Image.fromarray(visuals['visual_3']).resize((new_w, ref_height)))
        
        # Add titles and images in order
        titles = {
            'visual_0': 'Obs',
            'visual_1': 'Imag. Path',
            'visual_3': 'Path'
        }
        
        # Add the individual images
        for i, (key, title) in enumerate(titles.items()):
            if key in visuals and visuals[key] is not None:
                ax = fig.add_subplot(gs[0, i])
                ax.imshow(visuals[key])
                ax.set_title(title, fontsize=18)
                ax.axis('off')
    
    # Add a bounding box around the entire row
    from matplotlib.patches import Rectangle
    fig.patches.extend([Rectangle((0, 0), 1, 1, fill=False, edgecolor='black', 
                                 linewidth=2, transform=fig.transFigure)])
    
    # Save this step's visualization temporarily
    plt.tight_layout(rect=[0.01, 0.01, 0.99, 0.95])  # Adjust layout to account for the bounding box
    temp_path = save_path.replace('.png', f'_temp_{step}.png')
    plt.savefig(temp_path)
    plt.close()
    
    # Now combine with previous visualization if it exists
    step_img = Image.open(temp_path)
    
    if prev_img is None:
        # First step - add target object title at the top
        if target_obj:
            # Create a separate figure just for the title with a large font
            title_height = 120  # Much larger height for the title section
            title_fig = plt.figure(figsize=(step_img.width/100, title_height/100))  # Convert pixels to inches
            title_fig.patch.set_facecolor('white')
            
            # Add a large, centered title
            plt.figtext(0.5, 0.5, f"Target Object: {target_obj}", 
                      fontsize=36, fontweight='bold', ha='center', va='center')
            
            # No axes for the title
            plt.axis('off')
            
            # Save the title image
            title_path = save_path.replace('.png', '_title.png')
            plt.savefig(title_path, bbox_inches='tight', pad_inches=0.3)
            plt.close(title_fig)
            
            # Open the title image
            title_img = Image.open(title_path)
            
            # Create a new combined image
            combined_height = title_img.height + step_img.height
            combined_img = Image.new('RGB', (max(title_img.width, step_img.width), combined_height), color=(255, 255, 255))
            
            # Paste the title and step images
            combined_img.paste(title_img, ((combined_img.width - title_img.width) // 2, 0))
            combined_img.paste(step_img, ((combined_img.width - step_img.width) // 2, title_img.height))
            
            # Save the combined image
            combined_img.save(save_path)
            
            # Clean up temporary title image
            import os
            if os.path.exists(title_path):
                os.remove(title_path)
                
            return combined_img
        else:
            # No target object specified
            step_img.save(save_path)
            return step_img
    else:
        # Append this step to previous visualization
        combined_height = prev_img.height + step_img.height
        combined_img = Image.new('RGB', (max(prev_img.width, step_img.width), combined_height))
        combined_img.paste(prev_img, (0, 0))
        combined_img.paste(step_img, (0, prev_img.height))
        combined_img.save(save_path)
        
        # Clean up temporary file
        import os
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
        return combined_img

def run_object_searching(cfg: DictConfig):
    # load config
    run_dir = cfg.results_folder
    save_scene = cfg.agent.save_scene
    num_imagined_trajectories = cfg.agent.num_imagined_trajectories
    semantic_thred = cfg.agent.semantic_thred
    adjacent_angle = cfg.adjacent_angle
    adjacent_distance = cfg.adjacent_distance
    # initialize the belief agent
    belief_agent = BeliefAgent(cfg)
    # initialize the vlm agent
    vlm = VLMAgent(vlm_model_name="gpt-4o")

    # initialize the Stretch controller
    stretch_controller = StretchController(**STRETCH_ENV_ARGS)
    # load task manager config using hydra
    with initialize_config_dir(config_dir=str(pathlib.Path(__file__).parent.parent / "configuration")):
        embodied_config = compose(config_name="object_searching")
    task_manager = ObjectSearchingTaskManager(config=embodied_config, stretch_controller=stretch_controller)
    env_interface = EnvInterface(config=embodied_config, stretch_controller=stretch_controller, task_manager=task_manager)
    # extract some info
    num_episodes = len(task_manager.episodes)
    max_steps = embodied_config.max_steps
    print(f"Loaded {len(task_manager.episodes)} episodes from {task_manager.episode_root}")
    print(f"Max steps per episode: {task_manager.max_steps}")

    # run the object searching task
    for idx in range(num_episodes):
        env_interface.reset()
        belief_agent.reset()
        target_object = task_manager.current_target_obj
        print(f"Episode {idx}: Searching for {target_object}")
        all_folders = create_save_folders(run_dir, f"episode_{idx}_{target_object}")

        first_pose_spoc = None
        step = 0
        done = False
        start_time = time.time()
        while step < max_steps and not done:
            # step start time
            step_start_time = time.time()
            # extract current obs
            spoc_obs = env_interface.get_observation()

            if step == 0:
                first_pose_spoc = spoc_obs["pose"]

            Image.fromarray(spoc_obs["rgb"]).save(
                os.path.join(all_folders["save_folder_obs"], f"observed_{step}.png")
            )
            
            visual_0 = spoc_obs["rgb"]

            # TODO implement conversion for spoc to belief
            belief_obs = BeliefAgent.convert_to_belief_obs(spoc_obs, first_pose_spoc)

            current_location = belief_obs["pose"][:3, 3].detach().cpu().numpy()
            
            # observe with the current observation
            belief_agent.observe([belief_obs["rgb"]], [belief_obs["pose"]])
            # save obs map
            belief_agent.obs_map.save_occupancy_map(
                os.path.join(save_folder_obs_obs_map, f"obs_map_{step}.png"),
            )
            # save height map
            belief_agent.obs_map.save_height_map(
                os.path.join(save_folder_obs_height_map, f"height_map_{step}.png"),
            )
            # render at the current pose
            rgb, depth, _ = belief_agent.render_image(extrinsics=belief_obs["pose"], query_label=target_obj)

            # use vlm TODO
            success = vlm.prompt_score_obj_image(
                image_file=os.path.join(save_folder_obs, f"observed_{step}.png"),
                object_name=target_obj,
            )

            # If new observation contains the target object, set success
            if success:
                step_log = {
                    "step": idx,
                    "is_direct": success,
                    "target_obj": target_obj,
                    "step_time": time.time() - step_start_time,
                }
                # dump the step log
                with open(os.path.join(save_folder_sample, f"step_log_{step}.json"), "w") as f:
                    json.dump(step_log, f, indent=4)
                
                # Create and save a final visualization with just the observation where object was found
                vis_path = os.path.join(save_folder_sample, "visualization.png")
                prev_vis = Image.open(vis_path) if os.path.exists(vis_path) and step > 0 else None
                
                # Create a simplified visualization with only the observation image
                final_visuals = {
                    'visual_0': visual_0,
                    'visual_1': None,
                    'visual_2': [],
                    'visual_3': None
                }
                
                # Create a special version of visualization for success state
                create_success_visualization(final_visuals, step, vis_path, prev_vis, target_obj)
                
                done = True
                print(f"Found target object {target_obj} in observation {step}.")
                continue
            else: # Otherwise, continue exploring and imagining
                goals = belief_agent.sample_next_exploration_goals(
                    belief_agent.obs_map, 
                    belief_agent.current_pose[:3, 3].detach().cpu().numpy(),
                    plot_path=os.path.join(save_folder_obs_map, f"map_{step}.png")
                )   
                print("# Goals", len(goals))

                backup_goal = goals[-1]

                # filter out goals that not in the room
                goals = [goal for goal in goals if point_in_room(
                    env_interface, 
                    BeliefAgent.points_belief2spoc([goal["pose"][-1][:3, 3]], first_pose_spoc)[0],
                    task_manager.room_name
                )]
                # filter out goals that are too close to a wall
                goals = [goal for goal in goals if not is_too_close_to_wall(
                    env_interface,
                    BeliefAgent.points_belief2spoc([goal["pose"][-1][:3, 3]], first_pose_spoc)[0],
                    forward=BeliefAgent.pose_belief2spoc(goal["pose"][-1], first_pose_spoc)[:3, 2],
                    buffer=0.2
                )]

                goals.append(backup_goal)

                # keep at most num_imagined_trajectories goals
                if len(goals) > num_imagined_trajectories-1:
                    goals = random.sample(goals, num_imagined_trajectories-1)
                # append the backup goal

                save_folder_imagine_step = os.path.join(save_folder_imagine, f'step_{step}')
                os.makedirs(
                    save_folder_imagine_step, exist_ok=True,
                )
                
                optimal_goal = None
                optimal_belief_scene = None
                optimal_key_poses = None
                optimal_frames = None
                optimal_scores = None
                best_semantic_score = -1
                for gidx, goal_dict in enumerate(goals):
                    path = goal_dict["path"]
                    poses = goal_dict["pose"]

                    imagined_frames = []

                    belief_agent.obs_map.save_height_map(
                        os.path.join(save_folder_height_map, f"height_map_with_goals_{step}_{gidx}.png"), path=path
                    )

                    save_folder_imagine_step_goal = os.path.join(save_folder_imagine_step, f'goal_{gidx}')
                    os.makedirs(
                        save_folder_imagine_step_goal, exist_ok=True,
                    )

                    # 3D imagination
                    key_output, _, belief_scene = belief_agent.imagine_in_place(
                                                        poses[-1], 
                                                        query_label=target_obj,
                                                        return_belief_scene=True
                                                    )
                    frames = key_output.rgb
                    depths = key_output.depth
                    key_poses = key_output.pose

                    for p, frame in enumerate(frames):
                        Image.fromarray(frame).save(
                            os.path.join(save_folder_imagine_step_goal, f"rendered_{p}.png")
                        )
                        imagined_frames.append(frame)

                    # object detection on the imagined frames
                    results = vlm.prompt_score_obj_folder(
                        image_folder=save_folder_imagine_step_goal,
                        object_name=target_obj,
                    )
                    presences = [ele[0] for ele in results]
                    scores = [ele[1] for ele in results]
                    max_idx = np.argmax(scores)
                    semantic_score = scores[max_idx]
                    
                    # save the belief scene
                    if save_scene:
                        ply_path = Path(f"{save_folder_imagine_step_goal}/scene_goal_{gidx}.ply")
                        export_gaussians_to_ply(
                            belief_scene[-1].float(),
                            key_poses[-1].detach().unsqueeze(0).to("cuda"),
                            ply_path
                        )
                    
                    if semantic_score > best_semantic_score:
                        best_semantic_score = semantic_score
                        optimal_belief_scene = belief_scene
                        optimal_key_poses = key_poses
                        optimal_frames = imagined_frames
                        optimal_scores = scores
                        optimal_goal = optimal_key_poses[max_idx].detach().cpu().numpy()[:3, 3]

                # Set imagined occupancy map
                obs_map = deepcopy(belief_agent.obs_map)
                assert len(optimal_belief_scene) == len(optimal_key_poses)
                for i in range(len(optimal_key_poses)):
                    inc_map = OccupancyMap(resolution=belief_agent.step_size, obstacle_height_thresh=belief_agent.obstacle_height_thresh)
                    obs_pose_np = optimal_key_poses[i].detach().cpu().numpy()
                    Rcw = obs_pose_np[:3, :3]
                    forward = Rcw @ np.array([0.0, 0.0, 1.0])
                    yaw = -math.atan2(forward[0], forward[2])
                    yaw = yaw % (2*math.pi)
                    inc_map.set_point_cloud(
                        pcd=optimal_belief_scene[i].float().means.squeeze(0).detach().cpu().numpy(), 
                        sensor_origin=tuple(obs_pose_np[:3, 3]), 
                        yaw=yaw, 
                        intrinsics=belief_agent.camera.intrinsics
                    )
                    obs_map.merge(inc_map)
                
                # DEBUG save occupancy map
                obs_map.save_occupancy_map(
                    os.path.join(save_folder_obs_map, f"imagined_plan_obs_map_{step}.png"),
                    goals=[optimal_goal],
                )

                imagined_key_points = [key.detach().cpu().numpy()[:3, 3] for key in optimal_key_poses]
                # add current_location as the first point
                imagined_key_points.insert(0, current_location)

                visual_1 = belief_agent.obs_map.save_occupancy_map(
                    os.path.join(save_folder_obs_map, f"imagined_path_obs_map_{step}.png"),
                    goals=[current_location],
                    path=imagined_key_points,
                    return_image=True
                )

                visual_2 = optimal_frames
                visual_2_scores = optimal_scores

                # plan a path to the goal
                path = obs_map.plan(tuple(belief_obs["pose"][:3, 3].detach().cpu().numpy()), tuple(optimal_goal))
                if path is None:
                    print("Failed to plan a path to the goal, using interpolation.")
                    path = belief_agent.interpolate_path(
                        belief_obs["pose"][:3, 3].detach().cpu().numpy(),
                        optimal_goal,
                        step_size=0.05, # TODO set a step size
                    )
            
            # Calculate the path distance between start and goal
            if path is not None:
                path_distance = np.linalg.norm(
                    np.array(path[-1]) - np.array(path[0])
                )
                if path_distance < 1.5:
                    face_to_object = True
                else:
                    face_to_object = False

            visual_3 = belief_agent.obs_map.save_occupancy_map(
                os.path.join(save_folder_planning, f"path_obs_map_{step}.png"),
                goals=[current_location],
                path=path[:len(path)+1],
                return_image=True
            )
            
            # Create and save visualization with all visuals
            vis_path = os.path.join(save_folder_sample, "visualization.png")
            prev_vis = Image.open(vis_path) if os.path.exists(vis_path) and step > 0 else None
            
            # Create a dictionary of visuals
            visuals = {
                'visual_0': visual_0,
                'visual_1': visual_1,
                'visual_2': visual_2,
                'visual_2_scores': visual_2_scores,
                'visual_3': visual_3
            }
            
            # Update the visualization - pass target_obj to create_step_visualization
            _ = create_step_visualization(visuals, step, vis_path, prev_vis, target_obj)

            path_spoc = BeliefAgent.points_belief2spoc(path, first_pose_spoc)
            path_spoc_exe = path_spoc[:len(path_spoc)+1] # TODO find a subset of the path with step size

            # record current forward direction
            current_position, current_forward = get_current_position_and_forward(env_interface)
            z_previous = current_forward
            t_previous = current_position

        # save navigation video
        image_buffer = env_interface.image_buffer
        string_buffer = env_interface.string_buffer
        # save images as a video
        video_path = os.path.join(all_folders['save_folder_sample'], f"episode_{task_manager.current_episode_index}_navigation.mp4")
        imageio.mimwrite(video_path, image_buffer['rgb'], fps=5, quality=8)
        print(f"Saved navigation video to {video_path}")
        # save string buffer as a jsonl file
        jsonl_path = os.path.join(all_folders['save_folder_sample'], f"episode_{task_manager.current_episode_index}_navigation.jsonl")
        with jsonlines.open(jsonl_path, mode='w') as writer:
            for i in range(len(string_buffer['action'])):
                entry = {key: string_buffer[key][i] for key in string_buffer}
                writer.write(entry)
        print(f"Saved navigation log to {jsonl_path}")

        # setup final log
        final_log = {
            "scene": task_manager.current_house_index,
            "target_obj": target_object,
            "num_steps": step,
            "success": done,
            "time_taken": time.time() - start_time,
        }
        # dump final log
        with open(os.path.join(all_folders['save_folder_sample'], f"final_log.json"), "w") as f:
            json.dump(final_log, f, indent=4)

if __name__ == "__main__":
    cfg_path = "/home/ubuntu/VLMP/tianmin-project/yyin34/codebase/embodied_tasks/DFM/configurations"
    with initialize_config_dir(config_dir=cfg_path, version_base="1.2"):
        cfg = compose(
            config_name="sp_reason.yaml",
            overrides=[
                "sampling_steps=10",
                "semantic_mode=embed",
                "semantic_viz=query",
                "adjacent_angle=0.785",
                "adjacent_distance=1.0",
                "clean_target=False",
                "use_history=False",
                "model.encoder.use_epipolar_transformer=False",
                "model.encoder.use_image_condition=True",
                "model.encoder.depth_predictor_time_embed=True",
                "model.encoder.evolve_ctxt=True",
                "model.encoder.use_camera_pose=True",
                "model.encoder.use_semantic=False",
                "model.encoder.use_reg_model=False",
                "model.encoder.d_semantic=512",
                "model.encoder.d_semantic_reg=384",
                "model.encoder.gaussians_per_pixel=3",
                "model.encoder.inference_mode=False",
                "model.encoder.backbone.use_diff_pos_embed=True",
                "model.encoder.backbone.pose_condition_type=prope",
                "agent.save_scene=False",
            ]
        )
    cfg.checkpoint_path = "/home/ubuntu/VLMP/tianmin-project/yyin34/codebase/DFM/outputs/weights/habelief/dfm_prope_evolve_ctxt_semantic_room_ft/model-2.pt"
    cfg.results_folder = "/home/ubuntu/VLMP/tianmin-project/yyin34/codebase/embodied_tasks/DFM/outputs/belief_agent_prope_evolve_ctxt_semantic_room_ft"
    cfg.semantic_config = "/home/ubuntu/VLMP/tianmin-project/yyin34/codebase/embodied_tasks/DFM/configurations/semantic/onehot.yaml"

    run_object_searching(cfg)