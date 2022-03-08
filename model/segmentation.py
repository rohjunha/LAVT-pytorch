import torch
from torch.functional import Tensor 
import torch.nn as nn 
import torch.nn.functional as F
from typing import List, Tuple


class TinyConv(nn.Module):
    def __init__(self, C_in, s1, s2):
        super().__init__()
        self.conv1=nn.Conv2d(C_in,C_in,3,1,1)
        self.conv2=nn.Conv2d(C_in,C_in,3,1,1)
        #self.norm1 = None
        #self.norm2 = None
        self.norm1=nn.LayerNorm((C_in, s1, s1))
        self.norm2=nn.LayerNorm((C_in, s2, s2))
        
    def forward(self,x):
        x = F.relu(self.norm1(self.conv1(x)))
        x = F.relu(self.norm2(self.conv2(x)))
        return x


class Segmentation(nn.Module):
    def __init__(
        self,
        channel_list:List[int]=[2048,2560,2816],
        size_list: List[Tuple[int, int]] = [(14, 14), (28, 28), (56, 56)]
    ):
        super().__init__()
        # build tiny conv nets
        self.tinyConvs=nn.ModuleList()
        for channel, sizes in zip(channel_list, size_list):
            self.tinyConvs.append(
                TinyConv(channel, *sizes)
            )
        # last projection map
        self.conv=nn.Conv2d(channel_list[-1],2,1)
    
    def forward(self,fuse_tensors:List[Tensor]):
        out=fuse_tensors[0]
        for i in range(1,len(fuse_tensors)):
            out=torch.cat([out,fuse_tensors[i]],dim=1)
            out=self.tinyConvs[i-1](out)
            out=F.interpolate(out,scale_factor=2,mode='bilinear',align_corners=True)
        
        return self.conv(out)
    



        

