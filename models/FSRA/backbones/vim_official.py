"""
Official Vision Mamba (Vim) Implementation
基于官方 https://github.com/hustvl/Vim 和 https://github.com/doodleima/vision_mamba

Vision Mamba: Efficient Visual Representation Learning with Bidirectional State Space Model
支持加载官方预训练权重：vim_t_midclstok_ft_78p3acc.pth

Key Features:
- Bidirectional State Space Model
- Position embeddings for image sequences  
- Middle class token design (midclstok)
- Efficient hardware-aware designs
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from typing import Optional
import os
import warnings

# 尝试导入mamba相关模块，如果没有则使用fallback实现
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn
    from mamba_ssm.modules.mamba_simple import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
    warnings.warn("mamba_ssm not available, using fallback implementation")

def to_2tuple(x):
    if isinstance(x, (list, tuple)):
        return x
    return (x, x)


class PatchEmbed(nn.Module):
    """Image to Patch Embedding"""
    def __init__(self, img_size=224, patch_size=16, stride_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        stride_size = to_2tuple(stride_size)
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.stride_size = stride_size
        
        # 计算patch数量
        self.num_x = (img_size[1] - patch_size[1]) // stride_size[1] + 1
        self.num_y = (img_size[0] - patch_size[0]) // stride_size[0] + 1
        self.num_patches = self.num_x * self.num_y
        
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride_size)
        
    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)  # (B, embed_dim, num_y, num_x)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        return x


class VimBlock(nn.Module):
    """
    Vision Mamba Block with Bidirectional State Space Model
    """
    def __init__(
        self,
        dim,
        mixer_cls,
        norm_cls=nn.LayerNorm,
        fused_add_norm=False,
        residual_in_fp32=False,
        drop_path=0.0,
    ):
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(self, hidden_states: torch.Tensor, residual: Optional[torch.Tensor] = None):
        if not self.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + self.drop_path(hidden_states)
            
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            if residual is None:
                hidden_states, residual = fused_add_norm_fn(
                    hidden_states,
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )
            else:
                hidden_states, residual = fused_add_norm_fn(
                    self.drop_path(hidden_states),
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )    
        hidden_states = self.mixer(hidden_states)
        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)


class BidirectionalMamba(nn.Module):
    """
    Bidirectional Mamba implementation for Vision
    """
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True,
        layer_idx=None,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx

        if MAMBA_AVAILABLE:
            # 使用官方Mamba实现
            self.forward_mamba = Mamba(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                dt_rank=dt_rank,
                dt_min=dt_min,
                dt_max=dt_max,
                dt_init=dt_init,
                dt_scale=dt_scale,
                dt_init_floor=dt_init_floor,
                conv_bias=conv_bias,
                bias=bias,
                use_fast_path=use_fast_path,
                layer_idx=layer_idx,
                device=device,
                dtype=dtype,
            )
            self.backward_mamba = Mamba(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                dt_rank=dt_rank,
                dt_min=dt_min,
                dt_max=dt_max,
                dt_init=dt_init,
                dt_scale=dt_scale,
                dt_init_floor=dt_init_floor,
                conv_bias=conv_bias,
                bias=bias,
                use_fast_path=use_fast_path,
                layer_idx=layer_idx,
                device=device,
                dtype=dtype,
            )
        else:
            # Fallback实现
            self.forward_proj = nn.Linear(d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
            self.backward_proj = nn.Linear(d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
            self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias, **factory_kwargs)
            
            # 简化的状态空间参数
            self.dt_proj_forward = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)
            self.dt_proj_backward = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)
            
    def forward(self, hidden_states):
        batch, seqlen, dim = hidden_states.shape
        
        if MAMBA_AVAILABLE:
            # 使用官方Mamba进行双向处理
            # Forward direction
            forward_output = self.forward_mamba(hidden_states)
            
            # Backward direction
            backward_input = torch.flip(hidden_states, dims=[1])  # 反转序列
            backward_output = self.backward_mamba(backward_input)
            backward_output = torch.flip(backward_output, dims=[1])  # 恢复序列顺序
            
            # 融合双向输出
            output = (forward_output + backward_output) / 2
        else:
            # Fallback实现
            # Forward direction
            forward_states = self.forward_proj(hidden_states)
            forward_x, forward_z = forward_states.chunk(2, dim=-1)
            forward_x = F.silu(forward_x)
            forward_output = forward_x * F.silu(forward_z)
            
            # Backward direction  
            backward_input = torch.flip(hidden_states, dims=[1])
            backward_states = self.backward_proj(backward_input)
            backward_x, backward_z = backward_states.chunk(2, dim=-1)
            backward_x = F.silu(backward_x)
            backward_output = backward_x * F.silu(backward_z)
            backward_output = torch.flip(backward_output, dims=[1])
            
            # 融合双向输出
            combined = (forward_output + backward_output) / 2
            output = self.out_proj(combined)
            
        return output

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        if MAMBA_AVAILABLE:
            forward_cache = self.forward_mamba.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs) 
            backward_cache = self.backward_mamba.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            return {"forward": forward_cache, "backward": backward_cache}
        else:
            return {}


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  
        output = x.div(keep_prob) * random_tensor
        return output


def create_block(
    d_model,
    ssm_cfg=None,
    norm_epsilon=1e-5,
    drop_path=0.0,
    rms_norm=False,
    residual_in_fp32=False,
    fused_add_norm=False,
    layer_idx=None,
    device=None,
    dtype=None,
):
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    mixer_cls = partial(BidirectionalMamba, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    block = VimBlock(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        drop_path=drop_path,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block


class VisionMambaOfficial(nn.Module):
    """
    Official Vision Mamba (Vim) Implementation
    Support loading pretrained weights from vim_t_midclstok_ft_78p3acc.pth
    """
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        stride_size=16,
        depth=24,
        embed_dim=192,
        channels=3,
        num_classes=1000,
        ssm_cfg=None,
        drop_rate=0.0,
        drop_path_rate=0.1,
        norm_epsilon=1e-5,
        rms_norm=False,
        initializer_cfg=None,
        fused_add_norm=False,
        residual_in_fp32=False,
        device=None,
        dtype=None,
        ft_seq_len=None,
        pt_hw_seq_len=14,
        if_bidirectional=True,
        final_pool_type='none',
        if_abs_pos_embed=True,
        if_rope=False,
        if_rope_residual=False,
        flip_img_sequences_ratio=-1.,
        if_bimamba=False,
        bimamba_type="none",
        if_cls_token=True,
        if_devide_out=True,
        init_layer_scale=None,
        use_middle_cls_token=True,
        local_feature=False,
        **kwargs
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.if_bidirectional = if_bidirectional
        self.final_pool_type = final_pool_type
        self.if_abs_pos_embed = if_abs_pos_embed
        self.if_rope = if_rope
        self.if_rope_residual = if_rope_residual
        self.flip_img_sequences_ratio = flip_img_sequences_ratio
        self.if_cls_token = if_cls_token
        self.use_middle_cls_token = use_middle_cls_token
        self.num_tokens = 1 if if_cls_token else 0
        self.local_feature = local_feature
        
        # 关键配置
        self.num_classes = num_classes
        self.d_model = self.num_features = self.embed_dim = embed_dim
        
        # Patch embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size, 
            patch_size=patch_size, 
            stride_size=stride_size,
            in_chans=channels, 
            embed_dim=embed_dim
        )
        num_patches = self.patch_embed.num_patches

        # 位置编码和cls token
        if if_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
            
        if if_abs_pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, self.embed_dim))
            self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  
        
        # transformer blocks
        self.layers = nn.ModuleList(
            [
                create_block(
                    embed_dim,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    drop_path=dpr[i],
                    **factory_kwargs,
                )
                for i in range(depth)
            ]
        )
        
        # output head
        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            embed_dim, eps=norm_epsilon, **factory_kwargs
        )
        
        # 分类头（用于预训练，FSRA会替换）
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        # 权重初始化
        if if_abs_pos_embed:
            trunc_normal_(self.pos_embed, std=.02)
        if if_cls_token:
            trunc_normal_(self.cls_token, std=.02)
            
        self.apply(
            partial(
                _init_weights,
                n_layer=depth,
                **(initializer_cfg if initializer_cfg is not None else {}),
                n_residuals_per_layer=1 if if_bidirectional else 2,
            )
        )

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    def forward_features(self, x, inference_params=None):
        x = self.patch_embed(x)
        B, M, _ = x.shape
        
        if self.if_cls_token:
            cls_token = self.cls_token.expand(B, -1, -1)
            if self.use_middle_cls_token:
                # 将cls token放在中间位置
                mid_idx = M // 2
                x = torch.cat([x[:, :mid_idx, :], cls_token, x[:, mid_idx:, :]], dim=1) 
            else:
                x = torch.cat([cls_token, x], dim=1)
            
        if self.if_abs_pos_embed:
            x = x + self.pos_embed
            x = self.pos_drop(x)

        # 通过Mamba layers
        residual = None
        for layer in self.layers:
            if self.local_feature and layer == self.layers[-1]:
                break
            x, residual = layer(x, residual, inference_params=inference_params)
            
        if not self.fused_add_norm:
            if residual is None:
                residual = x
            x = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            x = fused_add_norm_fn(
                x,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )
        return x

    def forward(self, x, return_features=False, inference_params=None):
        x = self.forward_features(x, inference_params)
        
        if return_features:
            return x
            
        if self.final_pool_type == 'none':
            return x
        elif self.final_pool_type == 'mean':
            return x.mean(dim=1)
        elif self.final_pool_type == 'max':
            return x.max(dim=1)[0]
        else:
            raise NotImplementedError(f"pool_type {self.final_pool_type} not implemented")

    def load_param(self, model_path):
        """加载官方预训练权重"""
        if not model_path or model_path == '':
            print('No pretrained model provided for Vision Mamba, using random initialization')
            return
            
        if not os.path.exists(model_path):
            print(f'Pretrained model path {model_path} does not exist, using random initialization')
            return
            
        try:
            print(f'Loading pretrained Vision Mamba model from {model_path}')
            checkpoint = torch.load(model_path, map_location='cpu')
            
            # 处理不同的checkpoint格式
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict'] 
            else:
                state_dict = checkpoint
            
            # 移除前缀
            new_state_dict = {}
            for k, v in state_dict.items():
                new_k = k.replace('module.', '') if k.startswith('module.') else k
                new_state_dict[new_k] = v
            
            # 加载权重，允许部分匹配
            missing_keys, unexpected_keys = self.load_state_dict(new_state_dict, strict=False)
            
            print(f'Successfully loaded pretrained Vision Mamba model')
            if missing_keys:
                print(f'Missing keys: {missing_keys[:5]}{"..." if len(missing_keys) > 5 else ""}')
            if unexpected_keys:
                print(f'Unexpected keys: {unexpected_keys[:5]}{"..." if len(unexpected_keys) > 5 else ""}')
                
        except Exception as e:
            print(f'Failed to load pretrained model from {model_path}: {e}')
            print('Using random initialization instead...')


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    """Truncated normal initialization"""
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.", stacklevel=2)

    l = norm_cdf((a - mean) / std)
    u = norm_cdf((b - mean) / std)
    tensor.uniform_(2 * l - 1, 2 * u - 1)
    tensor.erfinv_()
    tensor.mul_(std * math.sqrt(2.))
    tensor.add_(mean)
    tensor.clamp_(min=a, max=b)
    return tensor


def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,  
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


# FSRA兼容接口
def vim_tiny_patch16_224_FSRA(img_size=(256, 256), stride_size=16, drop_rate=0., local_feature=False, **kwargs):
    """Vision Mamba Tiny for FSRA framework"""
    model = VisionMambaOfficial(
        img_size=img_size,
        patch_size=16,
        stride_size=stride_size,
        depth=24,
        embed_dim=192,
        drop_rate=drop_rate,
        drop_path_rate=0.0,
        if_cls_token=True,
        use_middle_cls_token=True,
        final_pool_type='none',
        if_abs_pos_embed=True,
        local_feature=local_feature,
        **kwargs
    )
    return model


def vim_small_patch16_224_FSRA(img_size=(256, 256), stride_size=16, drop_rate=0., local_feature=False, **kwargs):
    """Vision Mamba Small for FSRA framework"""
    model = VisionMambaOfficial(
        img_size=img_size,
        patch_size=16,
        stride_size=stride_size, 
        depth=24,
        embed_dim=384,
        drop_rate=drop_rate,
        drop_path_rate=0.05,
        if_cls_token=True,
        use_middle_cls_token=True,
        final_pool_type='none',
        if_abs_pos_embed=True,
        local_feature=local_feature,
        **kwargs
    )
    return model 