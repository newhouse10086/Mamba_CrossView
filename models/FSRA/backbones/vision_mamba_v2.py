"""
Vision Mamba v2 - 更接近真实Mamba特性的实现

基于官方VMamba的核心思想，实现了：
1. 2D选择性扫描（SS2D）
2. 四个方向的扫描路径
3. cumsum实现的高效selective scan
4. 更好的收敛特性
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from typing import Optional
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


def selective_scan_fn(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False):
    """
    高效的selective scan实现，基于cumsum的方法
    
    参数:
    u: (B, L, D) 输入序列
    delta: (B, L, D) 时间步长
    A: (D, N) 状态矩阵
    B: (B, L, N) 输入到状态的投影
    C: (B, L, N) 状态到输出的投影
    D: (D,) 直接连接权重
    z: (B, L, D) 门控
    """
    B, L, D = u.shape
    N = A.shape[1]
    
    # 应用delta_bias和softplus
    if delta_bias is not None:
        delta = delta + delta_bias.view(1, 1, -1)
    if delta_softplus:
        delta = F.softplus(delta)
    
    # 计算discretized A和B
    # dA = exp(delta * A) [B, L, D, N]
    dA = torch.exp(torch.einsum('bld,dn->bldn', delta, A))
    
    # dB_u = delta * u * B [B, L, D, N]  
    dB_u = torch.einsum('bld,bld,bln->bldn', delta, u, B)
    
    # 使用cumsum实现selective scan
    # 计算cumulative product of dA (反向累积然后正向)
    dA_cumsum = F.pad(dA[:, 1:], (0, 0, 0, 0, 0, 1)).flip(1).cumsum(1).exp().flip(1)
    
    # 状态计算
    x = dB_u * dA_cumsum
    x = x.cumsum(1) / (dA_cumsum + 1e-12)
    
    # 输出计算
    y = torch.einsum('bldn,bln->bld', x, C)
    
    # 添加直接连接
    if D is not None:
        y = y + u * D.view(1, 1, -1)
    
    # 应用门控
    if z is not None:
        y = y * F.silu(z)
    
    return y


class SS2D(nn.Module):
    """
    2D Selective Scan模块
    实现四个方向的扫描：从左到右，从右到左，从上到下，从下到上
    """
    def __init__(self, d_model, d_state=16, d_conv=3, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        
        # 输入投影
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        
        # 卷积层（每个方向一个）
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=True,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
        )
        
        # 状态空间参数
        self.x_proj = nn.Linear(self.d_inner, (self.d_state + self.d_state + self.d_model) * 4, bias=False)
        self.dt_proj = nn.Linear(self.d_model, self.d_inner * 4, bias=True)
        
        # A矩阵初始化
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32),
            "n -> d n", d=self.d_inner
        ).contiguous()
        A_log = torch.log(A)  # 保持A稳定
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True
        
        # D参数
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.D._no_weight_decay = True
        
        # 输出投影
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        """
        x: (B, H, W, C)
        """
        B, H, W, C = x.shape
        
        # 输入投影
        xz = self.in_proj(x)  # (B, H, W, 2*d_inner)
        x_inner, z = xz.chunk(2, dim=-1)  # 每个 (B, H, W, d_inner)
        
        # 2D卷积
        x_inner = x_inner.permute(0, 3, 1, 2).contiguous()  # (B, d_inner, H, W)
        x_inner = self.conv2d(x_inner)  # (B, d_inner, H, W)
        x_inner = x_inner.permute(0, 2, 3, 1).contiguous()  # (B, H, W, d_inner)
        x_inner = F.silu(x_inner)
        
        # 准备状态空间参数
        x_dbl = self.x_proj(x_inner)  # (B, H, W, (d_state*2+d_model)*4)
        dt, B_ssm, C_ssm = torch.split(x_dbl, [self.d_model*4, self.d_state*4, self.d_state*4], dim=-1)
        dt = self.dt_proj(dt.view(B, H, W, 4, self.d_model).mean(3))  # (B, H, W, d_inner*4)
        
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        
        # 四个方向的扫描
        ys = []
        for direction in range(4):
            # 获取当前方向的参数
            dt_i = dt[:, :, :, direction*self.d_inner:(direction+1)*self.d_inner]  # (B, H, W, d_inner)
            B_i = B_ssm[:, :, :, direction*self.d_state:(direction+1)*self.d_state]  # (B, H, W, d_state)
            C_i = C_ssm[:, :, :, direction*self.d_state:(direction+1)*self.d_state]  # (B, H, W, d_state)
            
            # 重组为序列格式进行扫描
            if direction == 0:  # 从左到右
                x_seq = x_inner.flatten(1, 2)  # (B, H*W, d_inner)
                dt_seq = dt_i.flatten(1, 2)    # (B, H*W, d_inner)
                B_seq = B_i.flatten(1, 2)      # (B, H*W, d_state)
                C_seq = C_i.flatten(1, 2)      # (B, H*W, d_state)
            elif direction == 1:  # 从右到左
                x_seq = x_inner.flip([2]).flatten(1, 2)  # (B, H*W, d_inner)
                dt_seq = dt_i.flip([2]).flatten(1, 2)    # (B, H*W, d_inner)
                B_seq = B_i.flip([2]).flatten(1, 2)      # (B, H*W, d_state)
                C_seq = C_i.flip([2]).flatten(1, 2)      # (B, H*W, d_state)
            elif direction == 2:  # 从上到下
                x_seq = x_inner.transpose(1, 2).flatten(1, 2)  # (B, W*H, d_inner)
                dt_seq = dt_i.transpose(1, 2).flatten(1, 2)    # (B, W*H, d_inner)
                B_seq = B_i.transpose(1, 2).flatten(1, 2)      # (B, W*H, d_state)
                C_seq = C_i.transpose(1, 2).flatten(1, 2)      # (B, W*H, d_state)
            else:  # 从下到上
                x_seq = x_inner.flip([1]).transpose(1, 2).flatten(1, 2)  # (B, W*H, d_inner)
                dt_seq = dt_i.flip([1]).transpose(1, 2).flatten(1, 2)    # (B, W*H, d_inner)
                B_seq = B_i.flip([1]).transpose(1, 2).flatten(1, 2)      # (B, W*H, d_state)
                C_seq = C_i.flip([1]).transpose(1, 2).flatten(1, 2)      # (B, W*H, d_state)
            
            # 执行selective scan
            y_seq = selective_scan_fn(x_seq, dt_seq, A, B_seq, C_seq, self.D)  # (B, L, d_inner)
            
            # 重组回2D格式
            if direction == 0:  # 从左到右
                y_2d = y_seq.view(B, H, W, self.d_inner)
            elif direction == 1:  # 从右到左
                y_2d = y_seq.view(B, H, W, self.d_inner).flip([2])
            elif direction == 2:  # 从上到下
                y_2d = y_seq.view(B, W, H, self.d_inner).transpose(1, 2)
            else:  # 从下到上
                y_2d = y_seq.view(B, W, H, self.d_inner).transpose(1, 2).flip([1])
            
            ys.append(y_2d)
        
        # 融合四个方向的结果
        y = sum(ys) / 4  # 简单平均
        
        # 门控
        y = y * F.silu(z)
        
        # 输出投影
        out = self.out_proj(y)
        out = self.dropout(out)
        
        return out


def repeat(tensor, pattern, **axes_lengths):
    """简化版的einops repeat"""
    if pattern == "n -> d n":
        d = axes_lengths['d']
        return tensor.unsqueeze(0).repeat(d, 1)
    else:
        raise NotImplementedError(f"Pattern {pattern} not implemented")


class VisionMambaBlock(nn.Module):
    """Vision Mamba Block"""
    def __init__(self, dim, d_state=16, d_conv=3, expand=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm = norm_layer(dim)
        self.ss2d = SS2D(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)

    def forward(self, x):
        """
        x: (B, L, C) -> 重组为 (B, H, W, C) -> SS2D -> 重组回 (B, L, C)
        """
        B, L, C = x.shape
        H = W = int(math.sqrt(L))  # 假设是正方形
        
        # 重组为2D
        x_2d = x.view(B, H, W, C)
        
        # 应用SS2D
        x_2d = x_2d + self.ss2d(self.norm(x_2d))
        
        # 重组回序列
        out = x_2d.view(B, L, C)
        return out


class VisionMambaV2(nn.Module):
    """Vision Mamba v2 for FSRA framework"""
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
        d_conv=3,
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
        
        # Vision Mamba blocks
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
            print('No pretrained model provided for Vision Mamba v2, using random initialization')
            return
            
        if not os.path.exists(model_path):
            print(f'Pretrained model path {model_path} does not exist, using random initialization')
            return
            
        try:
            print(f'Loading pretrained Vision Mamba v2 model from {model_path}')
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


def vision_mamba_v2_small(**kwargs):
    """Vision Mamba v2 Small model"""
    model = VisionMambaV2(
        embed_dim=768,
        depth=8,
        d_state=16,
        d_conv=3,
        expand=2,
        **kwargs
    )
    return model


def vision_mamba_v2_small_patch16_224_FSRA(img_size=(256, 128), stride_size=16, drop_rate=0., local_feature=False, **kwargs):
    """Vision Mamba v2 Small for FSRA with specific configuration"""
    model = VisionMambaV2(
        img_size=img_size,
        patch_size=16,
        stride_size=stride_size,
        embed_dim=768,
        depth=8,
        d_state=16,
        d_conv=3,
        expand=2,
        drop_rate=drop_rate,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        local_feature=local_feature,
        **kwargs
    )
    return model 