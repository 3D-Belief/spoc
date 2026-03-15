source /home/ubuntu/yifan/miniconda3/etc/profile.d/conda.sh
conda activate spoc

export CUDA_VISIBLE_DEVICES=7

export OBJAVERSE_DATA_DIR="/home/ubuntu/yifan/dataset/spoc"
export OBJAVERSE_HOUSES_DIR="/home/ubuntu/yifan/dataset/spoc/houses_2023_07_28"

# sudo apt-get update
# sudo apt-get -y install libvulkan1
# sudo apt install vulkan-tools
# ls -l /usr/share/vulkan/icd.d/ /etc/vulkan/icd.d/ 2>/dev/null | grep -i nvidia
# export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json  
# vulkaninfo --summary | egrep 'deviceName|driverID|apiVersion'

python /home/ubuntu/yifan/codebase/3d-belief/third_party/spoc/scripts/split.py \
    --base_dir /home/ubuntu/yifan/dataset/spoc/all/ObjectNavType/train \
    --house_groups_dir /home/ubuntu/yifan/dataset/spoc/all/ObjectNavType/house_groups \
    --gpus "0-7" \
    --groups_per_gpu 5

python /home/ubuntu/yifan/codebase/3d-belief/third_party/spoc/scripts/rerender.py --gpu_id 7 \
    --base_dir /home/ubuntu/yifan/dataset/spoc/all/ObjectNavType/train \
    --house_groups_dir /home/ubuntu/yifan/dataset/spoc/all/ObjectNavType/house_groups \
    --output_base_dir /home/ubuntu/yifan/dataset/spoc/training_data \
    --outdir_prefix obj_nav_type_