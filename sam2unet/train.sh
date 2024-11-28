CUDA_VISIBLE_DEVICES="0" \
python train.py \
--hiera_path "/data/ephemeral/home/whth/sam2unet/sam2_hiera_large.pt" \
--train_image_path "/data/ephemeral/home/ade20k_format/images/training" \
--train_mask_path "/data/ephemeral/home/ade20k_format/annotations/training" \
--save_path "/data/ephemeral/home/whth/checkpoints" \
--epoch 100 \
--lr 0.001 \
--batch_size 12 \
--api_key "" \
--team_name "" \
--project_name "" \
--experiment_detail "Sam2-unet"