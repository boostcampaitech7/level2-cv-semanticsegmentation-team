import os
import argparse
import random
import numpy as np
import torch
import torch.optim as opt
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from dataset import FullDataset
from SAM2UNet import SAM2UNet
import wandb
from wwandb import set_wandb

# wandb 세팅 + 다중 클래스로 loss 변경 

parser = argparse.ArgumentParser("SAM2-UNet")
parser.add_argument("--hiera_path", type=str, required=True, 
                    help="path to the sam2 pretrained hiera")
parser.add_argument("--train_image_path", type=str, required=True, 
                    help="path to the image that used to train the model")
parser.add_argument("--train_mask_path", type=str, required=True,
                    help="path to the mask file for training")
parser.add_argument('--save_path', type=str, required=True,
                    help="path to store the checkpoint")
parser.add_argument("--epoch", type=int, default=20, 
                    help="training epochs")
parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
parser.add_argument("--batch_size", default=12, type=int)
parser.add_argument("--weight_decay", default=5e-4, type=float)
parser.add_argument("--api_key", type=str, required=True)
parser.add_argument("--team_name", type=str, required=True)
parser.add_argument("--project_name", type=str, required=True)
parser.add_argument("--experiment_detail", type=str, required=True)
args = parser.parse_args()


def structure_loss(pred, mask):
    ce_loss = F.cross_entropy(pred, mask, reduction='mean')  
    pred_prob = F.softmax(pred, dim=1) 
    one_hot_mask = F.one_hot(mask, num_classes=pred.size(1)).permute(0, 3, 1, 2).float() 
    inter = (pred_prob * one_hot_mask).sum(dim=(2, 3)) 
    union = (pred_prob + one_hot_mask).sum(dim=(2, 3)) 
    wiou = 1 - (inter + 1) / (union - inter + 1) 
    wiou = wiou.mean(dim=1)  
    return ce_loss + wiou.mean()  


def main(args):    
    dataset = FullDataset(args.train_image_path, args.train_mask_path, 352, mode='train')
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    device = torch.device("cuda")
    model = SAM2UNet(args.hiera_path)
    model.to(device)
    optim = opt.AdamW([{"params":model.parameters(), "initia_lr": args.lr}], lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optim, args.epoch, eta_min=1.0e-7)
    os.makedirs(args.save_path, exist_ok=True)

    for epoch in range(args.epoch):
        model.train()
        for i, batch in enumerate(dataloader):
            x = batch['image']
            target = batch['label'].squeeze(1).to(torch.long)
            x = x.to(device)
            target = target.to(device)
            optim.zero_grad()
            
            pred0, pred1, pred2 = model(x)
            if i == 0 and epoch == 0:
                print("Pred0 shape:", pred0.shape)
                print("Target shape:", target.shape)
                print("First value in Pred0 (before sigmoid):", pred0.view(-1)[0].item())

            loss0 = structure_loss(pred0, target)
            loss1 = structure_loss(pred1, target)
            loss2 = structure_loss(pred2, target)
            loss = loss0 + loss1 + loss2
            
            if i % 50 == 0:
                print("epoch:{}-{}: loss:{}".format(epoch + 1, i + 1, loss.item()))
                wandb.log({"step_loss": loss.item(), "epoch": epoch + 1, "step": i + 1})

            loss.backward()
            optim.step()
                
        scheduler.step()
        if (epoch+1) % 5 == 0 or (epoch+1) == args.epoch:
            torch.save(model.state_dict(), os.path.join(args.save_path, 'SAM2-UNet-%d.pth' % (epoch + 1)))
            print('[Saving Snapshot:]', os.path.join(args.save_path, 'SAM2-UNet-%d.pth'% (epoch + 1)))

def seed_torch(seed=1024):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    seed_torch(1024)
    set_wandb(args)
    main(args)