CUDA_VISIBLE_DEVICES="0" \
python inference.py \
--checkpoint "/data/ephemeral/home/whth/checkpoints/SAM2-UNet-100.pth" \
--image_path "/data/ephemeral/home/ade20k_format/images/validation/00000001.jpg" \
--save_path "/data/ephemeral/home/whth/checkpointss"