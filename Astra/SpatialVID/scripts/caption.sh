#!/bin/bash
CSV=[Replace with the path to the result CSV file generated in the annotation step]
SRC_DIR=[Replace with the path to the annotation output directory]
OUTPUT_DIR=[Replace with the path to your output directory]
mkdir -p ${OUTPUT_DIR}

num_workers=8
wait_time=1

# VQA
vqa_prompt_file=caption/VQA/prompt.txt
vqa_model=gemini-2.0-flash
vqa_api_key=[Replace with your api key]
vqa_base_domain=https://generativelanguage.googleapis.com/

# LLM
llm_prompt_dir=caption/LLM
llm_model=qwen3-30b-a3b
llm_api_key=[Replace with your api key]
llm_base_domain=https://dashscope.aliyuncs.com/compatible-mode/

# Tagging
tag_prompt_file=caption/tagging/prompt.txt
tag_model=qwen3-30b-a3b
tag_api_key=[Replace with your api key]
tag_base_domain=https://dashscope.aliyuncs.com/compatible-mode/

measure_time() {
    local step_number=$1
    shift
    local green="\e[32m"
    local red="\e[31m"
    local no_color="\e[0m"
    local yellow="\e[33m"
    
    start_time=$(date +%s)
    echo -e "${green}Step $step_number started at: $(date)${no_color}"

    "$@"

    end_time=$(date +%s)
    echo -e "${red}Step $step_number finished at: $(date)${no_color}"
    echo -e "${yellow}Duration: $((end_time - start_time)) seconds${no_color}"
    echo "---------------------------------------"
}

# 1. VQA caption
measure_time 1 python caption/VQA/inference.py \
  --csv_path ${CSV} \
  --fig_load_dir ${SRC_DIR} \
  --output_dir ${OUTPUT_DIR} \
  --prompt_file ${vqa_prompt_file} \
  --model ${vqa_model} \
  --api_key ${vqa_api_key} \
  --base_domain ${vqa_base_domain} \
  --num_workers ${num_workers} \
  --wait_time ${wait_time}

# 2. LLM caption
measure_time 2 python caption/LLM/inference.py \
  --csv_path $CSV \
  --pose_load_dir $SRC_DIR \
  --output_dir $OUTPUT_DIR \
  --prompt_dir $llm_prompt_dir \
  --model $llm_model \
  --api_key $llm_api_key \
  --num_workers $num_workers \
  --base_domain $llm_base_domain \
  --wait_time $wait_time

# 3. Combine results
measure_time 3 python caption/utils/combine.py \
  --csv_path $CSV \
  --load_dir $OUTPUT_DIR \
  --output_dir $OUTPUT_DIR/results \
  --num_workers $num_workers

# 4. Add tags
python caption/tagging/inference.py \
    --csv_path $CSV \
    --json_load_dir $OUTPUT_DIR/results \
    --prompt_file $tag_prompt_file \
    --model $tag_model \
    --api_key $tag_api_key \
    --num_workers $num_workers \
    --base_domain $tag_base_domain \
    --wait_time $wait_time