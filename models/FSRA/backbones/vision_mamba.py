"""
Vision Mamba (VMamba) implementation for PyTorch

A PyTorch implementation of Vision Mamba as described in various Mamba vision papers.
This implementation is designed to be compatible with the FSRA framework.

Based on the Mamba architecture with adaptations for computer vision tasks.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from typing import Optional


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


class MambaBlock(nn.Module):
    """
    Simplified Mamba block for vision tasks
    This is a simplified implementation focusing on selective state space modeling
    """
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.dim = dim
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.dim)

        self.in_proj = nn.Linear(dim, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=True,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )
        
        # Linear layers for selective mechanism
        self.x_proj = nn.Linear(self.d_inner, d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)
        
        # State space parameters
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        self.out_proj = nn.Linear(self.d_inner, dim, bias=False)

    def forward(self, x):
        """
        x: (B, L, D)
        """
        B, L, D = x.shape
        
        # Input projection
        xz = self.in_proj(x)  # (B, L, 2*d_inner)
        x, z = xz.chunk(2, dim=-1)  # (B, L, d_inner)
        
        # 1D Convolution
        x = x.transpose(1, 2)  # (B, d_inner, L)
        x = self.conv1d(x)[:, :, :L]  # (B, d_inner, L)
        x = x.transpose(1, 2)  # (B, L, d_inner)
        
        # Activation
        x = F.silu(x)
        
        # Selective mechanism (simplified)
        x_dbl = self.x_proj(x)  # (B, L, 2*d_state)
        delta, B_ssm = x_dbl.chunk(2, dim=-1)  # (B, L, d_state)
        
        delta = F.softplus(self.dt_proj(x))  # (B, L, d_inner)
        
        # Simplified state space computation
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        
        # Discretization (simplified)
        deltaA = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0))  # (B, L, d_inner, d_state)
        
        # State computation (simplified selective scan)
        states = torch.zeros(B, self.d_inner, self.d_state, device=x.device, dtype=x.dtype)
        outputs = []
        
        for i in range(L):
            states = deltaA[:, i] * states + delta[:, i:i+1].unsqueeze(-1) * B_ssm[:, i:i+1].unsqueeze(1) * x[:, i:i+1].unsqueeze(-1)
            y = torch.sum(states * B_ssm[:, i:i+1].unsqueeze(1), dim=-1) + self.D * x[:, i]
            outputs.append(y)
        
        y = torch.stack(outputs, dim=1)  # (B, L, d_inner)
        
        # Gating with z
        y = y * F.silu(z)
        
        # Output projection
        output = self.out_proj(y)
        return output


class VisionMambaBlock(nn.Module):
    """Vision Mamba Block with normalization and residual connection"""
    def __init__(self, dim, d_state=16, d_conv=4, expand=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm = norm_layer(dim)
        self.mamba = MambaBlock(dim, d_state, d_conv, expand)

    def forward(self, x):
        return x + self.mamba(self.norm(x))


class VisionMamba(nn.Module):
    """Vision Mamba for FSRA framework"""
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        stride_size=16,
        in_chans=3,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        d_state=16,
        d_conv=4,
        expand=2,
        drop_rate=0.,
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
        
        # Mamba blocks
        self.blocks = nn.ModuleList([
            VisionMambaBlock(
                dim=embed_dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                norm_layer=norm_layer
            )
            for _ in range(depth)
        ])
        
        self.norm = norm_layer(embed_dim)
        
        # Classifier head
        self.fc = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        
        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        # Initialize class token and position embedding
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        # Initialize other weights
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
        
        # Apply mamba blocks
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
        """Load pretrained parameters (adapted from ViT loading)"""
        try:
            param_dict = torch.load(model_path, map_location='cpu')
            if 'model' in param_dict:
                param_dict = param_dict['model']
            if 'state_dict' in param_dict:
                param_dict = param_dict['state_dict']
            
            # Load compatible parameters
            model_dict = self.state_dict()
            pretrained_dict = {}
            
            for k, v in param_dict.items():
                if 'head' in k or 'dist' in k:
                    continue
                if k in model_dict and model_dict[k].shape == v.shape:
                    pretrained_dict[k] = v
                elif 'patch_embed.proj.weight' in k and len(v.shape) < 4:
                    # Handle patch embedding weight shape mismatch
                    O, I, H, W = self.patch_embed.proj.weight.shape
                    v = v.reshape(O, -1, H, W)
                    if v.shape == model_dict[k].shape:
                        pretrained_dict[k] = v
                elif k == 'pos_embed' and v.shape != self.pos_embed.shape:
                    # Handle position embedding size mismatch
                    pretrained_dict[k] = self._resize_pos_embed(v)
            
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)
            print(f'Loaded {len(pretrained_dict)} parameters from {model_path}')
            
        except Exception as e:
            print(f'Failed to load pretrained model from {model_path}: {e}')
            print('Initializing with random weights...')

    def _resize_pos_embed(self, posemb):
        """Resize position embedding to match current model"""
        # This is a simplified version - in practice you might want more sophisticated resizing
        return F.interpolate(
            posemb.permute(0, 2, 1), 
            size=self.pos_embed.shape[1], 
            mode='linear'
        ).permute(0, 2, 1)


def vision_mamba_small(**kwargs):
    """Vision Mamba Small model"""
    model = VisionMamba(
        embed_dim=768,
        depth=8,
        d_state=16,
        d_conv=4,
        expand=2,
        **kwargs
    )
    return model


def vision_mamba_base(**kwargs):
    """Vision Mamba Base model"""
    model = VisionMamba(
        embed_dim=768,
        depth=12,
        d_state=16,
        d_conv=4,
        expand=2,
        **kwargs
    )
    return model


def vision_mamba_small_patch16_224_FSRA(img_size=(256, 128), stride_size=16, drop_rate=0., local_feature=False, **kwargs):
    """Vision Mamba Small for FSRA with specific configuration"""
    model = VisionMamba(
        img_size=img_size,
        patch_size=16,
        stride_size=stride_size,
        embed_dim=768,
        depth=8,
        d_state=16,
        d_conv=4,
        expand=2,
        drop_rate=drop_rate,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        local_feature=local_feature,
        **kwargs
    )
    return model 