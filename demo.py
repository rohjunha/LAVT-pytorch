"""
This is a demo to check the correctness of the model
"""
import torch
import transformers
from dataset.ReferDataset import ReferDataset
from dataset.transform import get_transform
from args import get_parser
import config
from model.LAVT import LVAT,criterion
import random
import torch.nn.functional as F
from torch.utils.data import DataLoader

torch.cuda.set_device(7)
parse=get_parser()
args=parse.parse_args()
cfg=config.get_config(args)
args.size=cfg.DATA.IMG_SIZE

print(cfg.MODEL.NAME)
model=LVAT(cfg)
model.cuda()
num_params=sum(p.numel() for p in model.parameters() if p.requires_grad)
print(num_params)

transform=get_transform(args)

dataset=ReferDataset(args,split='testB',image_transforms=transform,eval_mode=False)

print(f"{dataset.split}:{len(dataset)}")
# dataloader
data=DataLoader(dataset,batch_size=1)
for d in data:
    model.zero_grad()
    img,targt,emb,att_mask=d
    emb=emb.squeeze(1)
    att_mask=att_mask.squeeze(1)
    img,targt,emb,att_mask=img.cuda(),targt.cuda(),emb.cuda(),att_mask.cuda()
    print(img.size(),targt.size(),emb.size(),att_mask.size())
    print("\nForward PATH")
    pred=model(img,emb,att_mask)
    print(f"pred:{pred.size()}")
    loss=criterion(pred,targt)
    print(loss)
    print("\nBackward PATH")
    loss.backward()






