import argparse
import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from SAM2UNet import SAM2UNet

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SAM2UNet()
    model.load_state_dict(torch.load(args.checkpoint))
    model.eval().to(device)

    transform = transforms.Compose([
        transforms.Resize((352, 352)), 
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    image = Image.open(args.image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output, _, _ = model(input_tensor)
        output = torch.softmax(output, dim=1)  
        mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy() 

    print("Output mask shape:", mask.shape)
    print("Mask unique values:", np.unique(mask))

    mask_image = Image.fromarray(mask.astype(np.uint8))
    mask_image = mask_image.resize((2048, 2048), resample=Image.NEAREST)

    save_path = args.save_path
    if not save_path.endswith(('.png', '.jpg', '.jpeg')):
        save_path += '.png'  
        
    mask_image.save(save_path)
    print(f"Predicted mask saved at {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--image_path", type=str, required=True, help="Path to input image")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save the predicted mask")
    args = parser.parse_args()

    main(args)
