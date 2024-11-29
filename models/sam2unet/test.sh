CUDA_VISIBLE_DEVICES="0" \
python test.py \
--checkpoint "/data/ephemeral/home/whth/checkpoints/SAM2-UNet-100.pth" \
--test_image_path "/data/ephemeral/home/ade20k_format/images/validation" \
--save_path "/data/ephemeral/home/whth/checkpointss"