python atk_driftmin.py \
    --input_image ./demo_images/gta_drive/1.png \
    --output_dir ./attacked/driftmin/1 \
    --eps 0.05 \
    --alpha 0.004 \
    --num_steps 300

python inference.py \
    --config_path ./configs/inference_yaml/inference_gta_drive.yaml\
    --img_path ./attacked/driftmin/1/adv_step_0300.png \
    --checkpoint_path Matrix-Game-2.0/gta_distilled_model/gta_keyboard2dim.safetensors \
    --output_folder ./outputs/driftmin/1 \
    --num_output_frames 60 \
    --seed 1234 \
    --pretrained_model_path Matrix-Game-2.0
