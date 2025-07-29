"""
Vision Mamba Lite - 轻量级实现

优化点：
1. 只使用2个方向扫描（水平+垂直）
2. 简化的状态空间建模
3. 更少的参数和计算量
4. 更快的训练速度
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
        
        # HE初始化 (适合ReLU族激活函数)
        nn.init.kaiming_normal_(self.proj.weight, mode='fan_out', nonlinearity='relu')
        if self.proj.bias is not None:
            nn.init.constant_(self.proj.bias, 0)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class LightweightMambaBlock(nn.Module):
    """
    轻量级Mamba块 - 简化版本
    只使用简单的状态更新，计算量大幅减少
    """
    def __init__(self, dim, expand=1.5):
        super().__init__()
        self.dim = dim
        self.expand = expand
        self.d_inner = int(self.expand * self.dim)
        
        # 简化的投影层
        self.in_proj = nn.Linear(dim, self.d_inner * 2, bias=False)
        self.out_proj = nn.Linear(self.d_inner, dim, bias=False)
        
        # 简化的卷积
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner, 
            kernel_size=3, padding=1, groups=self.d_inner
        )
        
        # 简化的状态参数 - 使用更合理的初始化
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)
        # A矩阵：状态衰减参数，初始化为小的负值
        self.A = nn.Parameter(-torch.rand(self.d_inner) * 0.1)
        # D矩阵：直接连接权重，初始化为1
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        # 对dt_proj使用HE初始化 (适合SiLU激活)
        nn.init.kaiming_uniform_(self.dt_proj.weight, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(self.dt_proj.bias, 0)

    def forward(self, x):
        """
        x: (B, L, D)
        """
        B, L, D = x.shape
        
        # 输入投影
        xz = self.in_proj(x)  # (B, L, 2*d_inner)
        x_inner, z = xz.chunk(2, dim=-1)  # (B, L, d_inner)
        
        # 1D卷积
        x_conv = x_inner.transpose(1, 2)  # (B, d_inner, L)
        x_conv = self.conv1d(x_conv)  # (B, d_inner, L)
        x_conv = x_conv.transpose(1, 2)  # (B, L, d_inner)
        x_conv = F.silu(x_conv)
        
        # 简化的状态更新 - 使用累积求和代替复杂的selective scan
        dt = F.softplus(self.dt_proj(x_conv))  # (B, L, d_inner)
        
        # 简化的状态计算：使用指数移动平均
        A_exp = torch.exp(self.A.unsqueeze(0).unsqueeze(0) * dt)  # (B, L, d_inner)
        
        # 使用cumsum进行高效计算
        x_weighted = x_conv * dt
        # 简化的状态累积
        y = torch.cumsum(x_weighted * A_exp, dim=1) * A_exp
        
        # 添加直接连接
        y = y + x_conv * self.D.unsqueeze(0).unsqueeze(0)
        
        # 门控
        y = y * F.silu(z)
        
        # 输出投影
        output = self.out_proj(y)
        
        return output


class SS2D_Lite(nn.Module):
    """
    轻量级2D选择性扫描 - 只使用2个方向（水平+垂直）
    """
    def __init__(self, d_model, expand=1.5):
        super().__init__()
        self.d_model = d_model
        
        # 两个方向的Mamba块
        self.mamba_h = LightweightMambaBlock(d_model, expand)  # 水平扫描
        self.mamba_v = LightweightMambaBlock(d_model, expand)  # 垂直扫描
        
        # 融合层
        self.fusion = nn.Linear(d_model * 2, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        x: (B, H, W, C)
        """
        B, H, W, C = x.shape
        
        # 水平扫描：按行扫描
        x_h = x.view(B * H, W, C)  # (B*H, W, C)
        y_h = self.mamba_h(x_h)    # (B*H, W, C)
        y_h = y_h.view(B, H, W, C) # (B, H, W, C)
        
        # 垂直扫描：按列扫描
        x_v = x.permute(0, 2, 1, 3).contiguous().view(B * W, H, C)  # (B*W, H, C)
        y_v = self.mamba_v(x_v)    # (B*W, H, C)
        y_v = y_v.view(B, W, H, C).permute(0, 2, 1, 3).contiguous()  # (B, H, W, C)
        
        # 融合两个方向的结果
        y_combined = torch.cat([y_h, y_v], dim=-1)  # (B, H, W, 2*C)
        y_fused = self.fusion(y_combined)           # (B, H, W, C)
        
        # 残差连接和归一化
        out = self.norm(y_fused + x)
        
        return out


class VisionMambaLiteBlock(nn.Module):
    """轻量级Vision Mamba Block"""
    def __init__(self, dim, expand=1.5, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm = norm_layer(dim)
        self.ss2d_lite = SS2D_Lite(d_model=dim, expand=expand)

    def forward(self, x):
        """
        x: (B, L, C) 其中L = num_patches + 1 (包含cls_token)
        """
        B, L, C = x.shape
        
        # 分离cls_token和patch tokens
        cls_token = x[:, 0:1, :]  # (B, 1, C)
        patch_tokens = x[:, 1:, :]  # (B, num_patches, C)
        
        # 计算patch的H, W
        num_patches = patch_tokens.shape[1]
        H = W = int(math.sqrt(num_patches))
        
        if H * W != num_patches:
            raise ValueError(f"Number of patches {num_patches} is not a perfect square")
        
        # 重组patch tokens为2D
        patch_2d = patch_tokens.view(B, H, W, C)
        
        # 应用轻量级SS2D
        patch_2d_out = self.ss2d_lite(self.norm(patch_2d))
        
        # 重组回序列格式
        patch_tokens_out = patch_2d_out.view(B, num_patches, C)
        
        # 对cls_token简单处理
        cls_token_out = cls_token + self.norm(cls_token)
        
        # 重新组合
        out = torch.cat([cls_token_out, patch_tokens_out], dim=1)
        
        return out


class VisionMambaLite(nn.Module):
    """轻量级Vision Mamba"""
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        stride_size=16,
        in_chans=3,
        num_classes=1000,
        embed_dim=512,  # 减小嵌入维度
        depth=6,        # 减少层数
        expand=1.5,     # 减小扩展比例
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
        
        # 轻量级Vision Mamba blocks
        self.blocks = nn.ModuleList([
            VisionMambaLiteBlock(
                dim=embed_dim,
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
        # 使用HE初始化cls_token和pos_embed
        nn.init.kaiming_normal_(self.cls_token, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.pos_embed, mode='fan_out', nonlinearity='relu')
        self.apply(self._init_weights_module)

    def _init_weights_module(self, m):
        if isinstance(m, nn.Linear):
            # Linear层使用HE uniform初始化 (适合SiLU激活)
            nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            # Conv2d层使用HE normal初始化
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            # Conv1d层使用HE normal初始化
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            # LayerNorm保持标准初始化
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Parameter):
            # 对于直接的Parameter，使用HE normal
            if len(m.shape) >= 2:
                nn.init.kaiming_normal_(m, mode='fan_out', nonlinearity='relu')
            else:
                # 1D参数使用小范围uniform初始化
                nn.init.uniform_(m, -0.1, 0.1)

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        
        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add position embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
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
        """Load pretrained parameters"""
        if not model_path or model_path == '':
            print('No pretrained model provided for Vision Mamba Lite, using random initialization')
            return
            
        try:
            print(f'Loading pretrained Vision Mamba Lite model from {model_path}')
            param_dict = torch.load(model_path, map_location='cpu')
            
            if 'model' in param_dict:
                param_dict = param_dict['model']
            elif 'state_dict' in param_dict:
                param_dict = param_dict['state_dict']
            
            model_dict = self.state_dict()
            pretrained_dict = {}
            
            for k, v in param_dict.items():
                key = k.replace('module.', '')
                if key in model_dict and model_dict[key].shape == v.shape:
                    pretrained_dict[key] = v
            
            if pretrained_dict:
                model_dict.update(pretrained_dict)
                self.load_state_dict(model_dict, strict=False)
                print(f'Successfully loaded {len(pretrained_dict)} parameters')
            else:
                print('No compatible parameters found, using random initialization')
                
        except Exception as e:
            print(f'Failed to load pretrained model: {e}')
            print('Using random initialization...')


def vision_mamba_lite_small_patch16_224_FSRA(img_size=(256, 128), stride_size=16, drop_rate=0., local_feature=False, **kwargs):
    """轻量级Vision Mamba Small for FSRA"""
    model = VisionMambaLite(
        img_size=img_size,
        patch_size=16,
        stride_size=stride_size,
        embed_dim=512,  # 比原来小
        depth=6,        # 比原来少
        expand=1.5,     # 比原来小
        drop_rate=drop_rate,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        local_feature=local_feature,
        **kwargs
    )
    return model 