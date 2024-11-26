import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from collections import defaultdict
from tqdm import tqdm
from mmengine.runner import load_checkpoint
from mmseg.registry import MODELS
from mmengine.dataset import Compose
from mmengine.config import Config
from mmseg.utils import register_all_modules

# MMSegmentation 모듈 등록
register_all_modules()

# Constants
IMAGE_ROOT = "/data/ephemeral/home/data/test/DCM"
CLASSES = [
    'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
    'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
    'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
    'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
    'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
    'Triquetrum', 'Pisiform', 'Radius', 'Ulna',
]  # Add your class names here
IND2CLASS = {i: cls for i, cls in enumerate(CLASSES)}  # Map index to class name

# Model setup
cfg = Config.fromfile('configs/config.py')  # Load your configuration file here
model = MODELS.build(cfg.model)
checkpoint = load_checkpoint(
    model,
    "/data/ephemeral/home/github/mmsegmentation/work_dir/segformer/7lg1mghx/best_mDice_epoch_33.pth",
    map_location="cpu"
)

# Get PNG image paths
pngs = {
    os.path.relpath(os.path.join(root, fname), start=IMAGE_ROOT)
    for root, _dirs, files in os.walk(IMAGE_ROOT)
    for fname in files
    if os.path.splitext(fname)[1].lower() == ".png"
}

# Utility function to encode mask to RLE
def encode_mask_to_rle(mask):
    '''
    mask: numpy array binary mask
    1 - mask
    0 - background
    Returns encoded run length
    '''
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

# Prepare data for inference
def _preprare_data(imgs, model):
    '''
    Prepare input data for inference based on the model's configuration.
    '''
    # Remove unnecessary pipeline steps
    for t in cfg.test_pipeline:
        if t.get('type') in ['LoadXRayAnnotations', 'TransposeAnnotations']:
            cfg.test_pipeline.remove(t)

    # Handle single image input
    is_batch = True
    if not isinstance(imgs, (list, tuple)):
        imgs = [imgs]
        is_batch = False

    # Handle numpy array inputs
    if isinstance(imgs[0], np.ndarray):
        cfg.test_pipeline[0]['type'] = 'LoadImageFromNDArray'

    # Build the pipeline
    pipeline = Compose(cfg.test_pipeline)

    # Process each image
    data = defaultdict(list)
    for img in imgs:
        if isinstance(img, np.ndarray):
            data_ = dict(img=img)
        else:
            data_ = dict(img_path=img)
        data_ = pipeline(data_)
        data['inputs'].append(data_['inputs'])
        data['data_samples'].append(data_['data_samples'])

    return data, is_batch

# Test function
def test(model, image_paths, thr=0.5):
    rles = []
    filename_and_class = []
    with torch.no_grad():
        n_class = len(CLASSES)

        for step, image_path in tqdm(enumerate(image_paths), total=len(image_paths)):
            img = cv2.imread(os.path.join(IMAGE_ROOT, image_path))

            # Prepare data
            data, is_batch = _preprare_data(img, model)

            # Forward the model
            with torch.no_grad():
                outputs = model.test_step(data)

            outputs = outputs[0].pred_sem_seg.data
            outputs = outputs[None]

            # Restore original size
            outputs = F.interpolate(outputs, size=(2048, 2048), mode="bilinear")
            outputs = torch.sigmoid(outputs)
            outputs = (outputs > thr).detach().cpu().numpy()

            output = outputs[0]
            image_name = os.path.basename(image_path)
            for c, segm in enumerate(output):
                rle = encode_mask_to_rle(segm)
                rles.append(rle)
                filename_and_class.append(f"{IND2CLASS[c]}_{image_name}")

    return rles, filename_and_class

# Main entry point
if __name__ == "__main__":
    # Run the test and save results
    rles, filename_and_class = test(model, pngs)
    classes, filename = zip(*[x.split("_") for x in filename_and_class])
    df = pd.DataFrame({
        "image_name": filename,
        "class": classes,
        "rle": rles,
    })

    df.to_csv("segformer_submission_2048.csv", index=False)
