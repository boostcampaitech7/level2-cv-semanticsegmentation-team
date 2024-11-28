import argparse
import os
import torch
import imageio
import numpy as np
import torch.nn.functional as F
from SAM2UNet import SAM2UNet
from dataset import TestDataset, NoneGTTestDataset


parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True,
                help="path to the checkpoint of sam2-unet")
parser.add_argument("--test_image_path", type=str, required=True, 
                    help="path to the image files for testing")
# parser.add_argument("--test_gt_path", type=str, required=True,
#                     help="path to the mask files for testing")
parser.add_argument("--save_path", type=str, required=True,
                    help="path to save the predicted masks")
args = parser.parse_args()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
test_loader = NoneGTTestDataset(args.test_image_path, 352)
model = SAM2UNet().to(device)
model.load_state_dict(torch.load(args.checkpoint), strict=True)
model.eval()
model.cuda()
os.makedirs(args.save_path, exist_ok=True)

for i in range(test_loader.size):
    with torch.no_grad():
        image, name = test_loader.load_data()
        image = image.to(device)
        res, _, _ = model(image)

        res = res.sigmoid().data.cpu().numpy()  
        res = res.squeeze(0)
        # res = F.interpolate(res, size=(2048, 2048), mode='bilinear', align_corners=False)
        # pred = torch.argmax(res, dim=1).cpu().numpy()  # [batch_size, H, W]
        # imageio.imsave('prediction.png', pred[0].astype(np.uint8))

        save_path = os.path.join(args.save_path, name[:-4] + ".tiff") # tiff로 저장
        imageio.mimwrite(save_path, res.astype(np.float32))  
        print(f"Saved multi-class output to {save_path}")
