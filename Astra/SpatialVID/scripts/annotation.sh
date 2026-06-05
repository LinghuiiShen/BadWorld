#!/bin/bash
CSV=[Replace with the path to the CSV file generated in the scoring step]
OUTPUT_DIR=[Replace with the path to your output directory]
mkdir -p ${OUTPUT_DIR}

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
GPU_NUM=8
ENHANCED=true  # Set to true to enable enhanced instruction generation

measure_time() {
    local step_number=$1
    shift
    local green="\e[32m"
    local red="\e[31m"
    local no_color="\e[0m"
    local yellow="\e[33m"
    
    start_time=$(date +%s)
    echo -e "${green}Step ${step_number} started at: $(date)${no_color}"

    "$@"

    end_time=$(date +%s)
    echo -e "${red}Step ${step_number} finished at: $(date)${no_color}"
    echo -e "${yellow}Duration: $((end_time - start_time)) seconds${no_color}"
    echo "---------------------------------------"
}

# 1. Extract frames
measure_time 1 python utils/extract_frames.py \
  --csv_path ${CSV} \
  --output_dir ${OUTPUT_DIR} \
  --num_workers $((GPU_NUM * 2)) \
  --target_size "1280*720" \
  --backend "opencv" \
  --interval 0.2

# 2.1 Depth Estimation with Depth-Anything
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} measure_time 2.1 torchrun --standalone --nproc_per_node ${GPU_NUM} camera_pose_annotation/depth_estimation/Depth-Anything/inference_batch.py \
  --csv_path ${CSV} \
  --encoder vitl \
  --checkpoints_path checkpoints \
  --output_dir ${OUTPUT_DIR} \
  --bs 16 \
  --num_workers ${GPU_NUM}

# 2.2 Depth Estimation with UniDepth
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} measure_time 2.2 torchrun --standalone --nproc_per_node ${GPU_NUM} camera_pose_annotation/depth_estimation/UniDepth/inference_batch.py \
  --csv_path ${CSV} \
  --output_dir ${OUTPUT_DIR} \
  --checkpoints_path checkpoints \
  --bs 32 \
  --num_workers ${GPU_NUM}

# 3. Camera Tracking
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} measure_time 3 python camera_pose_annotation/camera_tracking/inference_batch.py \
  --csv_path ${CSV} \
  --dir_path ${OUTPUT_DIR} \
  --checkpoints_path checkpoints \
  --gpu_id ${CUDA_VISIBLE_DEVICES} \
  --num_workers $((GPU_NUM * 2))

# 4.1 CVD Optimization Preprocess
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} measure_time 4.1 python camera_pose_annotation/cvd_opt/preprocess/inference_batch.py \
  --csv_path ${CSV} \
  --dir_path ${OUTPUT_DIR} \
  --checkpoints_path checkpoints \
  --gpu_id ${CUDA_VISIBLE_DEVICES} \
  --num_workers $((GPU_NUM * 2))

# 4.2 CVD Optimization
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} measure_time 4.2 python camera_pose_annotation/cvd_opt/inference_batch.py \
  --csv_path ${CSV} \
  --dir_path ${OUTPUT_DIR} \
  --gpu_id ${CUDA_VISIBLE_DEVICES} \
  --num_workers $((GPU_NUM * 2))
  # --only_depth

# 5. Dynamic Mask Prediction
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} measure_time 5 python camera_pose_annotation/dynamic_mask/inference_batch.py \
  --csv_path ${CSV} \
  --dir_path ${OUTPUT_DIR} \
  --checkpoints_path checkpoints \
  --gpu_num ${GPU_NUM} \
  --num_workers $((GPU_NUM * 2))

# 6. Evaluation of the results
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} measure_time 6 python utils/evaluation.py \
  --csv_path ${CSV} \
  --dir_path ${OUTPUT_DIR} \
  --gpu_num ${GPU_NUM} \
  --num_workers $((GPU_NUM * 2)) \
  --output_path ${OUTPUT_DIR}/final_results.csv

# 7. Get motion instructions
if [ "$ENHANCED" = false ] ; then
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} measure_time 7 python utils/get_instructions.py \
    --csv_path ${CSV} \
    --dir_path ${OUTPUT_DIR} \
    --interval 2 \
    --num_workers $((GPU_NUM * 2))
else
  echo "Standard instruction generation is enabled."
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} measure_time 7 python utils/get_instructions_enhanced.py \
    --csv_path ${OUTPUT_DIR}/final_results.csv \
    --dir_path ${OUTPUT_DIR} \
    --num_workers $((GPU_NUM * 2))
fi

# 8. Normalize the intrinsics
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} measure_time 8 python utils/normalize_intrinsics.py \
  --csv_path ${CSV} \
  --dir_path ${OUTPUT_DIR} \
  --num_workers $((GPU_NUM * 2))

# [Optional] Convert the output poses.npy into a c2w/w2c matrix
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} measure_time 9 python utils/quat_to_mat.py \
  --csv_path ${CSV} \
  --format c2w \
  --dir_path ${OUTPUT_DIR} \
  --num_workers $((GPU_NUM * 2))
