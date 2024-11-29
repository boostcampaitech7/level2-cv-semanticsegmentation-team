CUDA_VISIBLE_DEVICES="0" \
python realtest.py \
--checkpoint "/data/ephemeral/home/whth/checkpoints/SAM2-UNet-20.pth" \
--test_image_path "/data/ephemeral/home/ade20k_format/images/validation" \
--test_gt_path "/data/ephemeral/home/ade20k_format/annotations/validation" \
--save_path "/data/ephemeral/home/whth/checkpointss"