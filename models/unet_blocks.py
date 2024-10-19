import os
import sys
import torch
from torch import nn

# Add the parent directory to the path
sys.path.append(os.path.split(sys.path[0])[0])

# Import necessary components
try:
    from .attention import Transformer3DModel
    from .resnet import Downsample3D, ResnetBlock3D, Upsample3D
except ImportError:
    from attention import Transformer3DModel
    from resnet import Downsample3D, ResnetBlock3D, Upsample3D


def create_down_block(down_block_type, **kwargs):
    """Create a down block based on the specified type."""
    down_block_type = down_block_type[7:] if down_block_type.startswith("UNetRes") else down_block_type
    if down_block_type == "DownBlock3D":
        return DownBlock3D(**kwargs)
    elif down_block_type == "CrossAttnDownBlock3D":
        if kwargs['cross_attention_dim'] is None:
            raise ValueError("cross_attention_dim must be specified for CrossAttnDownBlock3D")
        return CrossAttnDownBlock3D(**kwargs)
    raise ValueError(f"{down_block_type} does not exist.")


def create_up_block(up_block_type, **kwargs):
    """Create an up block based on the specified type."""
    up_block_type = up_block_type[7:] if up_block_type.startswith("UNetRes") else up_block_type
    if up_block_type == "UpBlock3D":
        return UpBlock3D(**kwargs)
    elif up_block_type == "CrossAttnUpBlock3D":
        if kwargs['cross_attention_dim'] is None:
            raise ValueError("cross_attention_dim must be specified for CrossAttnUpBlock3D")
        return CrossAttnUpBlock3D(**kwargs)
    raise ValueError(f"{up_block_type} does not exist.")


class CrossAttnDownBlock3D(nn.Module):
    """3D Down Block with Cross Attention."""

    def __init__(self, in_channels, out_channels, temb_channels, num_layers=1, **kwargs):
        super().__init__()
        self.attentions = nn.ModuleList()
        self.resnets = nn.ModuleList()
        self.has_cross_attention = True
        self.attn_num_head_channels = kwargs.get('attn_num_head_channels', 1)

        for i in range(num_layers):
            resnet_block = ResnetBlock3D(
                in_channels=in_channels if i == 0 else out_channels,
                out_channels=out_channels,
                temb_channels=temb_channels,
                **kwargs
            )
            self.resnets.append(resnet_block)

            attn_block = Transformer3DModel(
                self.attn_num_head_channels,
                out_channels // self.attn_num_head_channels,
                in_channels=out_channels,
                num_layers=1,
                **kwargs
            )
            self.attentions.append(attn_block)

        if kwargs.get('add_downsample', True):
            self.downsamplers = nn.ModuleList([
                Downsample3D(out_channels, use_conv=True, out_channels=out_channels,
                             padding=kwargs.get('downsample_padding', 1))
            ])
        else:
            self.downsamplers = None

    def forward(self, hidden_states, temb=None, **kwargs):
        output_states = ()
        output_exemplar_latents = ()

        for resnet, attn in zip(self.resnets, self.attentions):
            hidden_states = self._process_layer(hidden_states, resnet, attn, temb, **kwargs)
            output_states += (hidden_states,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)
                output_states += (hidden_states,)

        return hidden_states, output_states

    def _process_layer(self, hidden_states, resnet, attn, temb, **kwargs):
        """Process each layer with ResNet and attention."""
        if self.training and kwargs.get('gradient_checkpointing', False):
            hidden_states = self._checkpoint_forward(resnet, hidden_states, temb)
            hidden_states = self._checkpoint_forward(attn, hidden_states, **kwargs)
        else:
            hidden_states = resnet(hidden_states, temb)
            hidden_states = attn(hidden_states, **kwargs)

        return hidden_states

    def _checkpoint_forward(self, module, *inputs):
        """Use gradient checkpointing to save memory."""
        return torch.utils.checkpoint.checkpoint(module, *inputs)


class CrossAttnUpBlock3D(nn.Module):
    """3D Up Block with Cross Attention."""

    def __init__(self, in_channels, out_channels, prev_output_channel, temb_channels, num_layers=1, **kwargs):
        super().__init__()
        self.attentions = nn.ModuleList()
        self.resnets = nn.ModuleList()

        for i in range(num_layers):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            resnet_block = ResnetBlock3D(
                in_channels=resnet_in_channels + res_skip_channels,
                out_channels=out_channels,
                temb_channels=temb_channels,
                **kwargs
            )
            self.resnets.append(resnet_block)

            attn_block = Transformer3DModel(
                kwargs.get('attn_num_head_channels', 1),
                out_channels // kwargs.get('attn_num_head_channels', 1),
                in_channels=out_channels,
                num_layers=1,
                **kwargs
            )
            self.attentions.append(attn_block)

        if kwargs.get('add_upsample', True):
            self.upsamplers = nn.ModuleList([Upsample3D(out_channels, use_conv=True, out_channels=out_channels)])
        else:
            self.upsamplers = None

    def forward(self, hidden_states, res_hidden_states_tuple, temb=None, **kwargs):
        for resnet, attn in zip(self.resnets, self.attentions):
            hidden_states, res_hidden_states_tuple = self._process_layer(hidden_states, res_hidden_states_tuple, resnet,
                                                                         attn, temb, **kwargs)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, kwargs.get('upsample_size', None))

        return hidden_states

    def _process_layer(self, hidden_states, res_hidden_states_tuple, resnet, attn, temb, **kwargs):
        """Process each layer with ResNet and attention for upsampling."""
        res_hidden_states = res_hidden_states_tuple[-1]
        res_hidden_states_tuple = res_hidden_states_tuple[:-1]
        hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

        hidden_states = resnet(hidden_states, temb)
        hidden_states = attn(hidden_states, **kwargs)

        return hidden_states, res_hidden_states_tuple
