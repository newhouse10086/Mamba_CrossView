"""
Vision Mamba (VMamba) implementation for PyTorch
基础版本的Vision Mamba实现，兼容FSRA框架

注意：推荐使用vim_official.py中的官方实现(VIM-TINY/VIM-SMALL)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import os


def to_2tuple(x):
    if isinstance(x, (list, tuple)):
        return x
    return (x, x)


class PatchEmbed(nn.Module):
    """Image to Patch Embedding"""
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        return x


class PatchEmbed_overlap(nn.Module):
    """Image to Patch Embedding with overlapping patches"""
    def __init__(self, img_size=224, patch_size=16, stride_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        stride_size_tuple = to_2tuple(stride_size)
        
        self.num_x = (img_size[1] - patch_size[1]) // stride_size_tuple[1] + 1
        self.num_y = (img_size[0] - patch_size[0]) // stride_size_tuple[0] + 1
        num_patches = self.num_x * self.num_y
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride_size)
        
        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class SimpleAttentionBlock(nn.Module):
    """
    简化的注意力块，作为Mamba的fallback实现
    确保代码可以运行，但建议使用官方VIM实现
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class VisionMambaBlock(nn.Module):
    """Vision Mamba Block using fallback attention mechanism"""
    def __init__(self, dim, num_heads=8, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = SimpleAttentionBlock(dim, num_heads=num_heads, qkv_bias=qkv_bias, 
                                       attn_drop=attn_drop, proj_drop=drop)
        
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionMamba(nn.Module):
    """基础Vision Mamba实现，使用注意力机制作为fallback"""
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        stride_size=16,
        in_chans=3,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.,
        qkv_bias=True,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.,
        norm_layer=nn.LayerNorm,
        local_feature=False,
        **kwargs
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.local_feature = local_feature
        
        # Patch embedding
        self.patch_embed = PatchEmbed_overlap(
            img_size=img_size, patch_size=patch_size, stride_size=stride_size,
            in_chans=in_chans, embed_dim=embed_dim
        )
        num_patches = self.patch_embed.num_patches
        
        # Class token and position embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        
        # Vision Mamba blocks (using attention fallback)
        self.blocks = nn.ModuleList([
            VisionMambaBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer
            )
            for i in range(depth)
        ])
        
        self.norm = norm_layer(embed_dim)
        
        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        
        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights_module)

    def _init_weights_module(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        
        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add position embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Apply blocks
        if self.local_feature:
            for blk in self.blocks[:-1]:
                x = blk(x)
            return x
        else:
            for blk in self.blocks:
                x = blk(x)
            x = self.norm(x)
            return x

    def forward(self, x):
        x = self.forward_features(x)
        return x

    def load_param(self, model_path):
        """加载预训练参数"""
        if not model_path or model_path == '':
            print('No pretrained model provided for Vision Mamba, using random initialization')
            return
            
        if not os.path.exists(model_path):
            print(f'Pretrained model path {model_path} does not exist, using random initialization')
            return
            
        try:
            print(f'Loading pretrained Vision Mamba model from {model_path}')
            param_dict = torch.load(model_path, map_location='cpu')
            
            # Handle different checkpoint formats
            if 'model' in param_dict:
                param_dict = param_dict['model']
            elif 'state_dict' in param_dict:
                param_dict = param_dict['state_dict']
            elif 'model_state_dict' in param_dict:
                param_dict = param_dict['model_state_dict']
            
            # Load compatible parameters
            model_dict = self.state_dict()
            pretrained_dict = {}
            
            for k, v in param_dict.items():
                # Skip head/classifier layers
                if 'head' in k or 'dist' in k or 'classifier' in k or 'fc' in k:
                    continue
                
                # Remove 'module.' prefix if present
                key = k.replace('module.', '')
                
                # Load parameters with matching shapes
                if key in model_dict and v.shape == model_dict[key].shape:
                    pretrained_dict[key] = v
            
            if pretrained_dict:
                model_dict.update(pretrained_dict)
                self.load_state_dict(model_dict, strict=False)
                print(f'Successfully loaded {len(pretrained_dict)} parameters')
            else:
                print('No compatible parameters found in the pretrained model')
                
        except Exception as e:
            print(f'Failed to load pretrained model from {model_path}: {e}')
            print('Using random initialization instead...')


def vision_mamba_small(**kwargs):
    """Vision Mamba Small model (fallback implementation)"""
    print("⚠️  警告: 正在使用fallback Vision Mamba实现")
    print("💡 建议: 使用 --backbone VIM-TINY 获得更好的性能")
    
    model = VisionMamba(
        embed_dim=768,
        depth=8,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        **kwargs
    )
    return model


def vision_mamba_small_patch16_224_FSRA(img_size=(256, 256), stride_size=16, drop_rate=0., local_feature=False, **kwargs):
    """Vision Mamba Small for FSRA framework (fallback implementation)"""
    print("⚠️  警告: 正在使用fallback Vision Mamba实现")
    print("💡 建议: 使用 --backbone VIM-TINY 获得官方实现和更好的性能")
    
    model = VisionMamba(
        img_size=img_size,
        patch_size=16,
        stride_size=stride_size,
        embed_dim=768,
        depth=8,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        drop_rate=drop_rate,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        local_feature=local_feature,
        **kwargs
    )
    return model 