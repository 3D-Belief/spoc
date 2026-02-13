source /home/ubuntu/jianwen-us-midwest-1/shulab-jhu/miniconda3/etc/profile.d/conda.sh
conda activate spoc

export CUDA_VISIBLE_DEVICES=3

python /home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/spoc/scripts/rerender.py --gpu_id 3 \
    --base_dir /home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/spoc/data/all/ObjectNavType/train \
    --house_groups_dir /home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/spoc/data/all/ObjectNavType/house_groups \
    --output_base_dir /home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/spoc/data/spoc_sat_3 \
    --outdir_prefix obj_nav_type_