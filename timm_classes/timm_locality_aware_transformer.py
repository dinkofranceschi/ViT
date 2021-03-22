import torch
import torch.nn as nn
from timm.models.layers.drop import DropPath
from timm.models.vision_transformer import Mlp,VisionTransformer
from functools import partial

'''Code adapted from 
https://github.com/VITA-Group/TransGAN/blob/7e5fa2d0c4d45ed2bf89068f0a9edb61a2a6db33/models/TransGAN_8_8_1.py#L61
thanks
'''
def get_attn_mask(N, w):
    mask = torch.zeros(1, 1, N, N).cuda()
    for i in range(N):
        if i <= w:
            mask[:, :, i, 0:i+w+1] = 1
        elif N - i <= w:
            mask[:, :, i, i-w:N] = 1
        else:
            mask[:, :, i, i:i+w+1] = 1
            mask[:, :, i, i-w:i] = 1
    return mask

# plt.imshow(get_attn_mask(20,15).cpu().squeeze().squeeze().detach().numpy())

class AttentionLAI(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0, mask_epochs=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.mask_epochs = mask_epochs #List with 4 elements that define the epochs of locality aware mask
        #We can't define the mask because we don't know the # of tokens
        #If # of tokens were to vary with each image (image with =/= sizes) we need to compute it on the fly
        #Otherwise we define in the first forward
        self.mask_4 = None
        self.mask_6 = None
        self.mask_8 = None
        self.mask_10 = None 
    def forward(self, x, epoch):
        B, N, C = x.shape
        #print(N)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = ((q @ k.transpose(-2, -1))) * self.scale
        if self.mask_epochs is not None:
            if epoch < self.mask_epochs[-1]:
                if epoch < self.mask_epochs[-4]:
                    if self.mask_4 is None:
                        self.mask_4 = get_attn_mask(N,8)
                    mask = self.mask_4
                elif epoch < self.mask_epochs[-3]:
                    if self.mask_6 is None:
                        self.mask_6= get_attn_mask(N,12)
                    mask = self.mask_6
                elif epoch < self.mask_epochs[-2]:
                    if self.mask_8 is None:
                        self.mask_8 = get_attn_mask(N,18)
                    mask = self.mask_8
                else:
                    if self.mask_10 is None:
                        self.mask_10 = get_attn_mask(N,20)
                    mask = self.mask_10
                attn = attn.masked_fill(mask.to(attn.get_device()) == 0, -1e9)
            else:
                pass
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn@v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
class BlockLAI(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, mask_epochs=None):
        
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = AttentionLAI(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, mask_epochs=mask_epochs)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, epoch):
        x = x + self.drop_path(self.attn(self.norm1(x), epoch))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
    

class VisionTransformerLAI(VisionTransformer):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None, representation_size=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., hybrid_backbone=None, norm_layer=None,mask_epochs=None):
        super().__init__(img_size=img_size,
                         patch_size=patch_size,
                         in_chans=in_chans,
                         num_classes=num_classes,
                         embed_dim=embed_dim,
                         depth=depth,
                         num_heads=num_heads,
                         mlp_ratio=mlp_ratio,
                         qkv_bias=qkv_bias,
                         qk_scale=qk_scale,
                         representation_size=representation_size,
                         drop_rate=drop_rate,
                         attn_drop_rate=attn_drop_rate,
                         drop_path_rate=drop_path_rate,
                         hybrid_backbone=hybrid_backbone,
                         norm_layer=norm_layer,
                         )
        norm_layer=norm_layer or partial(nn.LayerNorm, eps=1e-6)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            BlockLAI(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,mask_epochs=mask_epochs)
            for i in range(depth)])
        
    def forward_features(self, x, epoch):
        
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x,epoch)

        x = self.norm(x)[:, 0]
        x = self.pre_logits(x)
        return x

    def forward(self, x,epoch):
        x = self.forward_features(x,epoch)
        x = self.head(x)
        return x
    
def plot_attn_mask(L=65,w=20):
    import matplotlib.pyplot as plt
    import numpy as np
    plt.imshow(get_attn_mask(L,w).cpu().squeeze().squeeze().detach().numpy(),cmap='summer')
    plt.axis('off')
    plt.title('Locality-aware intialization mask')
    
    plt.plot(np.linspace(0,L),np.linspace(0,L),'k--',linewidth=0.7)
    plt.annotate(text='', xy=(w,0), xytext=(0,0), arrowprops=dict(arrowstyle='<->'))
    plt.text(w//2-1, 3, 'w')
    
    plt.annotate(text='', xy=(L-1,L-4), xytext=(0,L-4), arrowprops=dict(arrowstyle='<->'))
    plt.text((L-1)//2-1 , L-6,'L')
    
    plt.annotate(text='', xy=(w+(L-w)//2+L//10,L-w-(L-w)//2-L//10), xytext=(w+(L-w)//2,L-w-(L-w)//2), arrowprops=dict(arrowstyle='->'))
    plt.text(w+(L-w)//2+L//10,L-w-(L-w)//2-L//10,'$\Delta$ Epoch')
    plt.show()
        