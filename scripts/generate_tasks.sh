source /home/ubuntu/jianwen-us-midwest-1/shulab-jhu/miniconda3/etc/profile.d/conda.sh
conda activate spoc

export CUDA_VISIBLE_DEVICES=7
export XFORMERS_DISABLED=1

python /home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/spoc/rerender/reasoning_task_gen_main.py \
    --task_name visibility_trajectory \
    --output_dir /home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/spoc/data/spoc_object_visibility_more \
    --num_houses 200 \
    --start_house 50 \
    --max_objects_per_house 30 \
    --rotate_dir random \
    --deg_step 6 \
    --resume

python /home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/spoc/rerender/reasoning_task_gen_main.py \
    --task_name full_rotation \
    --output_dir /home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/spoc/data/spoc_full_rotation_unit \
    --num_houses 50 \
    --max_objects_per_house 10 \
    --rotate_dir random \
    --deg_step 6 \

python /home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/spoc/rerender/reasoning_task_gen_main.py \
    --task_name door_traversal \
    --output_dir /home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/spoc/data/spoc_door_traversal_unit \
    --num_houses 50 \
    --max_objects_per_house 20 \
    --rotate_dir random \
    --deg_step 6 \
    --resume

python /home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/spoc/rerender/door_passing.py \
  --process_all \
  --output_dir /home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/spoc/data/spoc_door_passing \
  --door_pre_steps 15 \
  --door_post_steps 5 \
  --extra_rot_steps 5 \

python /home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/spoc/rerender/camera_shaking.py \
  --process_all \
  --output_dir /home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/spoc/data/spoc_camera_shaking \
  --min_seq_len 20 \
  --max_seq_len 50 \
  --shake_deg 30 \
  --shake_dir random \
