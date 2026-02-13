source /home/ubuntu/jianwen-us-midwest-1/shulab-jhu/miniconda3/etc/profile.d/conda.sh
conda activate spoc

export CUDA_VISIBLE_DEVICES=2
export XFORMERS_DISABLED=1

## installation
conda create -n spoc python=3.9 -y
conda activate spoc
conda install -c conda-forge swig
conda install -c conda-forge compilers
# pip install --no-cache-dir box2d-py
pip install -r requirements.txt
pip install --extra-index-url https://ai2thor-pypi.allenai.org ai2thor==0+5d0ab8ab8760eb584c5ae659c2b2b951cab23246

sudo apt-get update
sudo apt-get -y install libvulkan1
sudo apt install vulkan-tools
sudo apt-get install -y \
  libnvidia-gl-570-server=570.148.08-0lambda0.22.04.1 \
  nvidia-driver-570-server=570.148.08-0lambda0.22.04.1
ls -l /usr/share/vulkan/icd.d/ /etc/vulkan/icd.d/ 2>/dev/null | grep -i nvidia
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json  
vulkaninfo --summary | egrep 'deviceName|driverID|apiVersion'

## download data
python -m scripts.download_training_data --save_dir /home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/spoc/data --types all --task_types ObjectNavType ObjectNavRoom ObjectNavDescription ObjectNavLocalRef
python -m objathor.dataset.download_assets --version 2023_07_28 --path /home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/spoc/data
python -m objathor.dataset.download_annotations --version 2023_07_28 --path /home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/spoc/data
python -m scripts.download_objaverse_houses --save_dir /home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/spoc/data --subset val
python -m scripts.download_objaverse_houses --save_dir /home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/spoc/data --subset train

## rerender
# when switch to val generation, remember to change reconstruct_scene.py l601 from ["train", "val"] to ["val"]
python rerender/reconstruct_scene.py --process_all

python /home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/spoc/scripts/split.py \
  --base_dir /home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/spoc/data/all/ObjectNavRoom/train \
  --house_groups_dir /home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/spoc/data/all/ObjectNavRoom/house_groups \
  --groups_per_gpu 20

## visualize camera poses
python /home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/spoc/utils/visualize_camera_pose.py \
    --pose_dir /home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/spoc/data/spoc_rerendered_trajectories_unit/house_000115_episode_0/pose \
    --output_dir /home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/spoc/data/visualize_camera \
    --frustum