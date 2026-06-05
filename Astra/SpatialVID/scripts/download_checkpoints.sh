mkdir -p ./checkpoints/
cd ./checkpoints/

# aesthetic
wget https://github.com/christophschuhmann/improved-aesthetic-predictor/raw/main/sac+logos+ava1-l14-linearMSE.pth -O aesthetic.pth

# megasam
wget https://github.com/mega-sam/mega-sam/blob/main/checkpoints/megasam_final.pth -O megasam_final.pth

# raft
gdown -c https://drive.google.com/uc?id=1MqDajR89k-xLV0HIrmJ0k-n8ZpG6_suM -O raft-things.pth

# depth anything
huggingface-cli download --resume-download depth-anything/Depth-Anything-V2-Large --local-dir Depth-Anything

# unidepth
huggingface-cli download --resume-download lpiccinelli/unidepth-v2-vitl14 --local-dir UniDepth

# sam
huggingface-cli download --resume-download facebook/sam2.1-hiera-large --local-dir SAM2
