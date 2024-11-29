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

def initialize_model(config_path, checkpoint_path, device):
    """
    Initialize the MMSegmentation model with configuration and checkpoint.

    Args:
        config_path (str): Path to the model configuration file.
        checkpoint_path (str): Path to the model checkpoint file.
        device (torch.device): Device to load the model onto.

    Returns:
        model (nn.Module): Initialized MMSegmentation model.
    """
    cfg = Config.fromfile(config_path)
    model = MODELS.build(cfg.model)
    load_checkpoint(model, checkpoint_path, map_location=device)
    model = model.to(device)
    model.eval()
    return model, cfg

def test(model, image_paths, cfg, device, thr=0.5):
    """
    Perform inference on a list of images.

    Args:
        model (nn.Module): MMSegmentation model.
        image_paths (list): List of image file paths.
        cfg (Config): Configuration for the model and pipeline.
        device (torch.device): Device to perform inference on.
        thr (float): Threshold for binary segmentation.

    Returns:
        rles (list): List of encoded RLEs for each segmentation mask.
        filename_and_class (list): List of filename and class name tuples.
    """
    rles = []
    filename_and_class = []
    with torch.no_grad():
        n_class = len(CLASSES)

        for step, image_path in tqdm(enumerate(image_paths), total=len(image_paths)):
            img = cv2.imread(os.path.join(IMAGE_ROOT, image_path))

            # Prepare data
            data, is_batch = _preprare_data(img, model)

            # Move inputs and data_samples to GPU
            data['inputs'] = [input_.to(device) for input_ in data['inputs']]
            data['data_samples'] = [sample.to(device) for sample in data['data_samples']]

            # Forward the model
            outputs = model.test_step(data)

            outputs = outputs[0].pred_sem_seg.data
            outputs = outputs[None]

            # Restore original size
            outputs = torch.sigmoid(outputs)
            outputs = (outputs > thr).detach().cpu().numpy()

            output = outputs[0]
            image_name = os.path.basename(image_path)
            for c, segm in enumerate(output):
                rle = encode_mask_to_rle(segm)
                rles.append(rle)
                filename_and_class.append(f"{IND2CLASS[c]}_{image_name}")

    return rles, filename_and_class

if __name__ == "__main__":
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model and configuration setup
    config_path = 'configs/config.py'
    checkpoint_path = '/data/ephemeral/home/github/mmsegmentation/work_dir/segformer/fold4/epoch_50.pth'
    model, cfg = initialize_model(config_path, checkpoint_path, device)

    # Run the test and save results
    rles, filename_and_class = test(model, pngs, cfg, device)
    classes, filename = zip(*[x.split("_") for x in filename_and_class])
    df = pd.DataFrame({
        "image_name": filename,
        "class": classes,
        "rle": rles,
    })

    df.to_csv("segformer_submission_2048_fold0123.csv", index=False)
