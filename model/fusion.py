from model.LanguagePathway import LanguagePath
from model.PWAN import PixelWordAttention
import torch 
import torch.nn as nn 
from torch import Tensor
from torch.nn import functional as F 

"""
combine PWAM and LanguagePath 

in: visual_feature, language_feature
out: fusion_feature (will be trans into next stage of Image Encoder)
"""
class FusionModel(nn.Module):
    def __init__(
        self,
        vis_channel,
        lan_channel,
    ):
        super().__init__()
        # pixel Word Attention Module
        self.pixelWordAttn=PixelWordAttention(vis_channel,lan_channel)
        # Language path
        self.languagePathway=LanguagePath(vis_channel)
    
    def forward(self,vis_feat,lan_feat):
        fusion_feat=self.pixelWordAttn(vis_feat,lan_feat)
        return self.languagePathway(vis_feat,fusion_feat)


def get_list_fusionModel():
    pass
