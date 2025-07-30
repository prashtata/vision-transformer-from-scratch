import torch
import torch.nn as nn


class PatchEmbeddings(nn.Module):
    def __init__(self, image_size, in_channels, patch_size = 4, embed_size = 32):
        super().__init__()

        self.proj = nn.Conv2d(in_channels=in_channels, out_channels=embed_size, kernel_size=patch_size, stride=patch_size) #This patchifies the image (downsizing from kernel and stride), while projecting onto embed_size dims

        num_patches = (image_size//patch_size)**2
        self.cls_token = nn.Parameter(torch.randn(1,1,embed_size))
        self.pos_embed = nn.Parameter(torch.randn(1, 1+num_patches, embed_size))

    def forward(self, x):
        bs = x.shape[0]
        x = self.proj(x)
        x = x.flatten(2).transpose(1,2) # (bs, embed_size, h/patch_size, w/patch_size) -> (bs, num_patches, embed_size)
        cls_token = self.cls_token.expand(bs,-1,-1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_embed

        return x
