import os
import sys
sys.path.append(os.path.split(sys.path[0])[0])

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.modeling_utils import ModelMixin
from diffusers.utils import BaseOutput
from diffusers.utils.import_utils import is_xformers_available
from diffusers.models.attention import CrossAttention, FeedForward, AdaLayerNorm

from einops import rearrange, repeat


@dataclass
class Transformer3DModelOutput(BaseOutput):
    sample: torch.FloatTensor


# xformers 导入
xformers = None
if is_xformers_available():
    import xformers
    import xformers.ops


class Transformer3DModel(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(self, num_attention_heads=16, attention_head_dim=88, in_channels=None, num_layers=1, 
                 dropout=0.0, norm_num_groups=32, cross_attention_dim=None, attention_bias=False,
                 activation_fn="geglu", num_embeds_ada_norm=None, use_linear_projection=False,
                 only_cross_attention=False, upcast_attention=False, use_first_frame=False):
        super().__init__()

        inner_dim = num_attention_heads * attention_head_dim
        self.use_linear_projection = use_linear_projection
        self.norm = nn.GroupNorm(num_groups=norm_num_groups, num_channels=in_channels, eps=1e-6, affine=True)
        self.proj_in = (nn.Linear(in_channels, inner_dim) if use_linear_projection 
                        else nn.Conv2d(in_channels, inner_dim, kernel_size=1))

        self.transformer_blocks = nn.ModuleList([
            BasicTransformerBlock(inner_dim, num_attention_heads, attention_head_dim, dropout, cross_attention_dim,
                                  activation_fn, num_embeds_ada_norm, attention_bias, only_cross_attention, upcast_attention, 
                                  use_first_frame)
            for _ in range(num_layers)
        ])

        self.proj_out = (nn.Linear(in_channels, inner_dim) if use_linear_projection
                         else nn.Conv2d(inner_dim, in_channels, kernel_size=1))

    def process_input(self, hidden_states, video_length, reshape_exemplar=False, exemplar_latent=None):
        hidden_states = rearrange(hidden_states, "b c f h w -> (b f) c h w")
        if reshape_exemplar:
            exemplar_latent = rearrange(exemplar_latent, "b c f h w -> (b f) c h w")
        return hidden_states, exemplar_latent

    def project_states(self, hidden_states, batch, height, weight, inner_dim):
        hidden_states = self.proj_in(hidden_states)
        hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * weight, inner_dim)
        return hidden_states

    def forward(self, hidden_states, encoder_hidden_states=None, exemplar_latent=None, timestep=None, reshape_exemplar=False, 
                return_dict=True, exemplar_encoder_hidden_states=None):

        # 检查输入维度
        assert hidden_states.dim() == 5, f"Expected hidden_states to have ndim=5, but got ndim={hidden_states.dim()}."

        video_length = hidden_states.shape[2]
        hidden_states, exemplar_latent = self.process_input(hidden_states, video_length, reshape_exemplar, exemplar_latent)

        batch, channel, height, weight = hidden_states.shape
        residual = hidden_states
        hidden_states = self.norm(hidden_states)

        # 处理输入投影
        hidden_states = self.project_states(hidden_states, batch, height, weight, self.num_attention_heads * self.attention_head_dim)

        if reshape_exemplar:
            exemplar_latent = self.norm(exemplar_latent)
            exemplar_latent = self.project_states(exemplar_latent, batch, height, weight, self.num_attention_heads * self.attention_head_dim)

        # Transformer Block
        for block in self.transformer_blocks:
            hidden_states = block(hidden_states, encoder_hidden_states, timestep, video_length, exemplar_latent)

        # 恢复维度
        hidden_states = hidden_states.reshape(batch, height, weight, self.num_attention_heads * self.attention_head_dim).permute(0, 3, 1, 2).contiguous()
        hidden_states = self.proj_out(hidden_states) + residual

        output = rearrange(hidden_states, "(b f) c h w -> b c f h w", f=video_length)
        if not return_dict:
            return output

        return Transformer3DModelOutput(sample=output)


class BasicTransformerBlock(nn.Module):
    def __init__(self, dim, num_attention_heads, attention_head_dim, dropout=0.0, cross_attention_dim=None, 
                 activation_fn="geglu", num_embeds_ada_norm=None, attention_bias=False, only_cross_attention=False, 
                 upcast_attention=False, use_first_frame=False):
        super().__init__()
        self.only_cross_attention = only_cross_attention
        self.use_ada_layer_norm = num_embeds_ada_norm is not None
        self.use_first_frame = use_first_frame

        self.attn1 = CrossAttention(query_dim=dim, heads=num_attention_heads, dim_head=attention_head_dim, 
                                    dropout=dropout, bias=attention_bias, cross_attention_dim=(cross_attention_dim if only_cross_attention else None),
                                    upcast_attention=upcast_attention)

        self.norm1 = AdaLayerNorm(dim, num_embeds_ada_norm) if self.use_ada_layer_norm else nn.LayerNorm(dim)
        self.ff = FeedForward(dim, dropout=dropout, activation_fn=activation_fn)
        self.norm3 = nn.LayerNorm(dim)
        self.attn_temp = CrossAttention(query_dim=dim, heads=num_attention_heads, dim_head=attention_head_dim, 
                                        dropout=dropout, bias=attention_bias, upcast_attention=upcast_attention)
        nn.init.zeros_(self.attn_temp.to_out[0].weight.data)
        self.norm_temp = AdaLayerNorm(dim, num_embeds_ada_norm) if self.use_ada_layer_norm else nn.LayerNorm(dim)

    def forward(self, hidden_states, encoder_hidden_states=None, timestep=None, attention_mask=None, video_length=None, exemplar_latent=None):
        norm_hidden_states = self.norm1(hidden_states, timestep) if self.use_ada_layer_norm else self.norm1(hidden_states)

        if self.only_cross_attention:
            hidden_states = self.attn1(norm_hidden_states, encoder_hidden_states, attention_mask=attention_mask) + hidden_states
        else:
            hidden_states = self.attn1(norm_hidden_states, attention_mask=attention_mask) + hidden_states

        # 前馈
        hidden_states = self.ff(self.norm3(hidden_states)) + hidden_states

        # 时间注意力
        d = hidden_states.shape[1]
        hidden_states = rearrange(hidden_states, "(b f) d c -> (b d) f c", f=video_length)
        norm_hidden_states = self.norm_temp(hidden_states, timestep) if self.use_ada_layer_norm else self.norm_temp(hidden_states)
        hidden_states = self.attn_temp(norm_hidden_states) + hidden_states
        hidden_states = rearrange(hidden_states, "(b d) f c -> (b f) d c", d=d)

        return hidden_states
