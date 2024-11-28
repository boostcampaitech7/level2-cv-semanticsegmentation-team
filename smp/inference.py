import os
import yaml
import torch
import argparse
import numpy as np
import pandas as pd
import os.path as osp
import albumentations as A
import torch.nn.functional as F
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import cv2


class Config:
    def __init__(self, config_path=None):
        self.MODEL_NAME = "FPN"  # 기본값
        self.ENCODER = "efficientnet-b4"
        self.ENCODER_WEIGHTS = "imagenet"
        self.IN_CHANNELS = 3
        self.CLASSES = ['finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
                       'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
                       'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
                       'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
                       'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
                       'Triquetrum', 'Pisiform', 'Radius', 'Ulna']
        self.ind2class = {i: c for i, c in enumerate(self.CLASSES)}
        
        if config_path:
            self.update_from_yaml(config_path)
    
    def update_from_yaml(self, config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        if 'model' in config:
            if 'name' in config['model']:
                self.MODEL_NAME = config['model']['name']
            if 'encoder' in config['model']:
                self.ENCODER = config['model']['encoder']
            if 'encoder_weights' in config['model']:
                self.ENCODER_WEIGHTS = config['model']['encoder_weights']
        
        if 'training' in config:
            self.BATCH_SIZE = config['training'].get('batch_size', 2)
            self.NUM_WORKERS = config['training'].get('num_workers', 2)

class XRayInferenceDataset(Dataset):
    def __init__(self, fnames, image_root, transforms=None):
        self.fnames = np.array(sorted(fnames))
        self.image_root = image_root
        self.transforms = transforms
        self.ind2class = {i: v for i, v in enumerate(['finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
                       'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
                       'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
                       'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
                       'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
                       'Triquetrum', 'Pisiform', 'Radius', 'Ulna'])}
    
    def __len__(self):
        return len(self.fnames)
    
    def __getitem__(self, item):
        image_name = self.fnames[item]
        image_path = osp.join(self.image_root, image_name)
        
        image = cv2.imread(image_path)
        image = image / 255.
        
        if self.transforms is not None:
            inputs = {"image": image}
            result = self.transforms(**inputs)
            image = result["image"]

        image = image.transpose(2, 0, 1)  
        image = torch.from_numpy(image).float()
            
        return image, image_name

def get_model(config):
    try:
        model = getattr(smp, config.MODEL_NAME)(
            encoder_name=config.ENCODER,
            encoder_weights=config.ENCODER_WEIGHTS,
            in_channels=config.IN_CHANNELS,
            classes=len(config.CLASSES)
        )
        return model
    except AttributeError:
        raise ValueError(
            f"Unsupported model architecture: {config.MODEL_NAME}. "
            f"Available architectures: {', '.join(smp.architectures)}"
        )

def _load_model(model_path, config):
    checkpoint = torch.load(model_path)
    model = get_model(config)
    model_state_dict = {k.replace('model.', ''): v for k, v in checkpoint.items()}
    model.load_state_dict(model_state_dict)
    return model

def encode_mask_to_rle(mask):
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def visualize_and_save(output_dir, image_path, image_name, outputs, config):
    # 이미지 파일이 있는 디렉토리 구조 그대로 visualization 폴더에 생성
    output_subdir = os.path.join(output_dir, os.path.dirname(image_name))
    if not os.path.exists(output_subdir):
        os.makedirs(output_subdir, exist_ok=True)

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    PALETTE = [
        (220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
        (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30),
        (100, 170, 30), (220, 220, 0), (175, 116, 175), (250, 0, 30), (165, 42, 42),
        (255, 77, 255), (0, 226, 252), (182, 182, 255), (0, 82, 0), (120, 166, 157),
        (110, 76, 0), (174, 57, 255), (199, 100, 0), (72, 0, 118), (255, 179, 240),
        (0, 125, 92), (209, 0, 151), (188, 208, 182), (0, 220, 176),
    ]

    preds = []
    for c in range(len(config.CLASSES)):
        pred = outputs[c]
        preds.append(pred)
    
    preds = np.stack(preds, 0)

    def label2rgb(label):
        image_size = label.shape[1:] + (3,)
        image = np.zeros(image_size, dtype=np.uint8)
        
        for i, class_label in enumerate(label):
            image[class_label == 1] = PALETTE[i]
        
        return image

    fig, ax = plt.subplots(1, 2, figsize=(24, 12))
    
    ax[0].imshow(image)
    ax[0].set_title('Original Image', fontsize=16)
    ax[0].axis('off')
    
    ax[1].imshow(label2rgb(preds))
    ax[1].set_title('Predicted Masks', fontsize=16)
    ax[1].axis('off')

    output_path = os.path.join(output_subdir, f"visualization_{os.path.basename(image_name)}")
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def main(args):
    # 설정 로드
    config = Config(args.config)
    
    # 테스트 이미지 파일 목록 생성
    fnames = {
        osp.relpath(osp.join(root, fname), start=args.image_root)
        for root, _, files in os.walk(args.image_root)
        for fname in files
        if osp.splitext(fname)[1].lower() == ".png"
    }

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = _load_model(args.model, config).to(device)
    model.eval()

    # 데이터 로더 설정
    tf = A.Resize(height=args.resize, width=args.resize)
    dataset = XRayInferenceDataset(fnames, args.image_root, transforms=tf)
    data_loader = DataLoader(
        dataset, 
        batch_size=getattr(config, 'BATCH_SIZE', 2),
        shuffle=False, 
        num_workers=getattr(config, 'NUM_WORKERS', 2),
        drop_last=False
    )

    rles = []
    filename_and_class = []
    
    with torch.no_grad():
        with tqdm(total=len(data_loader), desc="[Inference...]") as pbar:
            for images, image_names in data_loader:
                images = images.to(device)
                outputs = model(images)

                outputs = F.interpolate(outputs, size=(2048, 2048), mode="bilinear")
                outputs = torch.sigmoid(outputs)
                outputs = (outputs > args.thr).detach().cpu().numpy()

                for output, image_name in zip(outputs, image_names):
                    if args.save_vis:
                        image_path = os.path.join(args.image_root, image_name)
                        visualize_and_save(args.vis_dir, image_path, image_name, output, config)

                    for c, segm in enumerate(output):
                        rle = encode_mask_to_rle(segm)
                        rles.append(rle)
                        filename_and_class.append(f"{config.ind2class[c]}_{image_name}")
                
                pbar.update(1)

    # CSV 파일 생성
    classes, filename = zip(*[x.split("_") for x in filename_and_class])
    image_names = [os.path.basename(f) for f in filename]

    df = pd.DataFrame({
        "image_name": image_names,
        "class": classes,
        "rle": rles,
    })

    df.to_csv(args.output, index=False)
    print(f"Results saved to {args.output}")
    if args.save_vis:
        print(f"Visualizations saved to {args.vis_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str, help="Path to the model to use")
    parser.add_argument("--config", type=str, help="Path to config.yaml file")
    parser.add_argument("--image_root", type=str, default="/data/ephemeral/home/data/test/DCM")
    parser.add_argument("--thr", type=float, default=0.5)
    parser.add_argument("--output", type=str, default="./output.csv")
    parser.add_argument("--resize", type=int, default=512, help="Size to resize images (both width and height)")
    parser.add_argument("--save_vis", action="store_true", help="Save visualization results")
    parser.add_argument("--vis_dir", type=str, default="./visualization", help="Directory to save visualizations")
    args = parser.parse_args()

    main(args)