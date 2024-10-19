import os
import sys
sys.path.append(os.path.split(sys.path[0])[0])

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class InflatedConv3d(nn.Conv2d):  # 修正继承错误
    def forward(self, x):  # 修正拼写错误
        video_length = x.shape[2]
        x = rearrange(x, "b c f h w -> (b f) c h w")
        x = super().forward(x)  # 调用父类的forward
        x = rearrange(x, "(b f) c h w -> b c f h w", f=video_length)
        return x

class Upsample3D(nn.Module):
    def __init__(self, channels, use_conv=False, use_conv_transpose=False, out_channels=None, name="conv"):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_conv_transpose = use_conv_transpose
        self.name = name

        self.conv = None
        if use_conv_transpose:
            raise NotImplementedError
        elif use_conv:
            self.conv = InflatedConv3d(self.channels, self.out_channels, 3, padding=1)

    def forward(self, hidden_states, output_size=None):
        assert hidden_states.shape[1] == self.channels

        if self.use_conv_transpose:
            raise NotImplementedError

        dtype = hidden_states.dtype
        if dtype == torch.bfloat16:
            hidden_states = hidden_states.to(torch.float32)

        if hidden_states.shape[0] > 64:
            hidden_states = hidden_states.contiguous()

        if output_size is None:
            hidden_states = F.interpolate(hidden_states, scale_factor=[1.0, 2.0, 2.0], mode="nearest")
        else:
            hidden_states = F.interpolate(hidden_states, size=output_size, mode="nearest")

        if dtype == torch.bfloat16:
            hidden_states = hidden_states.to(torch.bfloat16)

        if self.use_conv:
            hidden_states = self.conv(hidden_states)

        return hidden_states

class Downsample3D(nn.Module):
    def __init__(self, channels, use_conv=False, use_conv_transpose=False, out_channels=None, name="conv"):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_conv_transpose = use_conv_transpose
        self.name = name
        self.conv = None
        if use_conv:
            self.conv = InflatedConv3d(self.channels, self.out_channels, 3, padding=1)
        elif use_conv_transpose:
            raise NotImplementedError

    def forward(self, hidden_states):
        assert hidden_states.shape[1] == self.channels

        if self.use_conv:
            hidden_states = self.conv(hidden_states)

        return hidden_states

class ResnetBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels=None, conv_shortcut=False, dropout=0.0, temb_channels=512,
                 groups=32, groups_out=None, pre_norm=True, eps=1e-6, non_linearity="swish",
                 time_embedding_norm="default", output_scale_factor=1.0, use_in_shortcut=None):
        super().__init__()
        self.pre_norm = pre_norm
        self.input_channel = in_channels
        self.output_channel = self.input_channel if out_channels is None else out_channels
        self.conv_shortcut = conv_shortcut
        self.time_embedding_norm = time_embedding_norm
        self.output_scale_factor = output_scale_factor

        if groups_out is None:
            groups_out = groups

        self.norm1 = nn.GroupNorm(num_groups=groups, num_channels=in_channels, eps=eps, affine=True)
        self.conv1 = InflatedConv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

        if temb_channels is not None:
            if self.time_embedding_norm == "default":
                time_emb_proj_out_channels = out_channels
            elif self.time_embedding_norm == "scale_shift":
                time_emb_proj_out_channels = out_channels * 2
            else:
                raise ValueError("Unknown time embedding norm")
            self.time_proj_emb = nn.Linear(temb_channels, time_emb_proj_out_channels)
        else:
            self.time_proj_emb = None

        self.norm2 = nn.GroupNorm(num_groups=groups_out, num_channels=out_channels, eps=eps, affine=True)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = InflatedConv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        if non_linearity == "swish":
            self.nonlinearity = F.silu
        elif non_linearity == "mish":
            self.nonlinearity = Mish()
        elif non_linearity == "silu":
            self.nonlinearity = nn.SiLU()

        self.use_in_shortcut = self.input_channel != self.output_channel if use_in_shortcut is None else use_in_shortcut
        self.conv_shortcut = InflatedConv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0) if self.use_in_shortcut else None

    def forward(self, input, temb):
        out = self.norm1(input)
        out = self.nonlinearity(out)
        out = self.conv1(out)

        if temb is not None:
            temb = self.time_proj_emb(self.nonlinearity(temb))[:, :, None, None, None]

        if temb is not None and self.time_embedding_norm == "default":
            out = out + temb
        out = self.norm2(out)

        if temb is not None and self.time_embedding_norm == "scale_shift":
            scale, shift = torch.chunk(temb, 2, dim=1)
            out = out * (1 + scale) + shift

        out = self.nonlinearity(out)
        out = self.dropout(out)
        out = self.conv2(out)

        if self.conv_shortcut is not None:
            input = self.conv_shortcut(input)

        out = (input + out) / self.output_scale_factor
        return out

class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))
