import logging
from dataclasses import dataclass

import sys
import os
from typing import List,Optional,Union,Tuple
import torch
import torch.nn as nn

sys.path.append(os.path.split(sys.path[0])[0])
from diffusers.utils import BaseOutput
from diffusers.modeling_utils import ModelMixin
from diffusers.configuration_utils import ConfigMixin,register_to_config
from diffusers.models.embeddings import TimestepEmbedding,Timesteps
try:
    from .unet_block import (
        CrossAttnDownBlock3D,
        CrossAttnUpBlock3D,
        DownBlock3D,
        UNetMidBlock3DCrossAttn,
        UpBlock3D,
        get_down_block,
        get_up_block,
    )
    from .resnet import InflatedConv3d
except:
    from .unet_block import (
        CrossAttnDownBlock3D,
        CrossAttnUpBlock3D,
        DownBlock3D,
        UNetMidBlock3DCrossAttn,
        UpBlock3D,
        get_down_block,
        get_up_block,
    )
    from resnet import InflatedConv3d

logger=logging.get_logger(__name__)
@dataclass
def UNet3DConditionOutput(BaseOutput):
    sample:torch.FloatTensor

class UNet3DConditionModelWaterMark(ModelMixin, ConfigMixin):
    _support_gradient_checkpointing=True

    @register_to_config
    def __init__(
        self,
        sample_size: Optional[int]=None,
        in_channels:int=4,
        out_channels:int=4,
        center_input_sample:bool=False,
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,
        down_block_types:Tuple[str]=(
            "CrossAttnDownBlock3D",
            "CrossAttnDownBlock3D",
            "CrossAttnDownBlock3D",
            "DownBlock3D"
        ),
        mid_block_types:str="UNetMidBlock3DCrossAttn",
        Up_block_types:Tuple[str]=(
            "UpBlock3D",
            "CrossAttnUpBlock3D",
            "CrossAttnUpBlock3D",
            "CrossAttnUpBlock3D",
        ),
        only_cross_attention:Union[bool,Tuple[bool]]=False,
        block_out_channels:Tuple[int]=(320,640,1280,1280),
        layers_per_block:int=2,
        downsample_padding:int=1,
        mid_block_scale_factor:float=1,
        act_fn:str="silu",
        norm_num_groups:int=32,
        norm_eps:float=1e-5,
        cross_attention_dim: int = 1280,
        attention_head_dim: Union[int, Tuple[int]] = 8,

        dual_cross_attention:bool=False,
        use_linear_projection:bool=False,
        class_embed_type:Optional[str]=None,
        num_class_embeds:Optional[int]=None,
        upcast_attention:bool=False,
        resnet_time_scale_shift:str="default",
        use_first_frame:bool=False,
    ):
        super.__init__()
        self.sample_size = sample_size,
        time_embed_dim=block_out_channels[0]*4
        #input
        self.conv_in=InflatedConv3d(in_channels,block_out_channels[0],kernel_size=3,padding=(1,1))

        #time
        self.time_proj=Timesteps(block_out_channels[0],flip_sin_to_cos,freq_shift)
        timestep_input_dim=block_out_channels[0]

        self.time_embedding=TimestepEmbedding(timestep_input_dim,time_embed_dim)

        if class_embed_type is None and num_class_embeds is not None:
            self.class_embedding=nn.Embedding(num_class_embeds,time_embed_dim)
        elif class_embed_type=="timestep":
            self.class_embedding=TimestepEmbedding(timestep_input_dim,time_embed_dim)
        elif class_embed_type=="identity":
            self.class_embedding=nn.Identity(time_embed_dim,time_embed_dim)
        else:
            self.class_embedding=None

        self.down_blocks=nn.ModuleList([])
        self.mid_block=None
        self.up_blocks=nn.ModuleList([])

        if isinstance(only_cross_attention,bool):
            only_cross_attention=[only_cross_attention]*len(down_block_types)

        if isinstance(attention_head_dim,int):
            attention_head_dim=(attention_head_dim,)*len(down_block_types)

        output_channel=block_out_channels[0]
        for i,down_block_type in enumerate(down_block_types):
            input_channel=output_channel
            output_channel=block_out_channels[i]
            is_final_block=i==len(block_out_channels)-1

            down_block=get_down_block(
                down_block_type,
                num_layers=layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=time_embed_dim,
                add_downsample=not is_final_block,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                cross_attention_dim=cross_attention_dim,
                attn_num_head_channels=attention_head_dim[i],
                downsample_padding=downsample_padding,
                dual_cross_attention=dual_cross_attention,
                use_linear_projection=use_linear_projection,
                only_cross_attention=only_cross_attention[i],
                upcast_attention=upcast_attention,
                resnet_time_scale_shift=resnet_time_scale_shift,
                use_first_frame=use_first_frame,
            )
            self.down_blocks.append(down_block)

            if mid_block_types=="UNetMidBlock3DCrossAttn":
                self.mid_block=UNetMidBlock3DCrossAttn(
                    in_channels=block_out_channels[-1],
                    temb_channels=time_embed_dim,
                    resnet_eps=time_embed_dim,
                    resnet_act_fn=act_fn,
                    output_scale_factor=mid_block_scale_factor,
                    attn_num_head_channels=attention_head_dim[-1],
                    resnet_groups=norm_num_groups,
                    dual_cross_attention=dual_cross_attention,
                    use_linear_projection=use_linear_projection,
                    upcast_attention=upcast_attention,
                    use_first_frame=use_first_frame,
                )
            else:
                raise ValueError(f"unknown mid_block_type:{mid_block_types}")

        self.num_upsamples=0

        #up
        reversed_block_out_channels=list(reversed(block_out_channels))
        reversed_attention_head_dim=list(reversed(attention_head_dim))
        only_cross_attention=list(reversed(only_cross_attention))
        output_channel=reversed_block_out_channels[0]
        for i,up_block_type in enumerate(Up_block_types):
            is_final_block=i==len(block_out_channels)-1
            prev_output_channel=output_channel
            output_channel=reversed_block_out_channels[i]
            input_channel=reversed_block_out_channels[min(i+1,len(block_out_channels)-1)]

            if not is_final_block:
                add_upsample=True
                self.num_upsamplers+=1
            else:
                add_upsample=False

            up_block=get_up_block(
                up_block_type,
                num_layers=layers_per_block+1,
                in_channels=input_channel,
                out_channels=output_channel,
                prev_output_channel=prev_output_channel,
                temb_channels=time_embed_dim,
                add_upsample=add_upsample,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                cross_attention_dim=cross_attention_dim,
                attn_num_head_channels=reversed_attention_head_dim[i],
                dual_cross_attention=dual_cross_attention,
                use_linear_projection=use_linear_projection,
                only_cross_attention=only_cross_attention[i],
                upcast_attention=upcast_attention,
                resnet_time_scale_shift=resnet_time_scale_shift,
                use_first_frame=use_first_frame,
            )
            self.up_blocks.append(up_block)
            prev_output_channel=output_channel

        #out
        #post-process
        self.watermark_down_blocks=nn.ModuleList([])
        self.watermark_up_blocks=nn.ModuleList([])

        watermark_down_block_types=['DownBlock3D','DownBlock3D','DownBlock3D']

        watermark_block_out_channels=[320,640,640]
        output_channel=watermark_block_out_channels[0]
        for i,watermark_down_block_type in enumerate(watermark_down_block_types):
            input_channel=output_channel
            output_channel=watermark_down_out_channels[i]
            is_final_block=i==len(watermark_down_block_types)-1
            down_block = get_down_block(
                watermark_down_block_type,
                num_layers=layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=time_embed_dim,
                add_downsample=not is_final_block,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                cross_attention_dim=cross_attention_dim,
                attn_num_head_channels=attention_head_dim[i],
                downsample_padding=downsample_padding,
                dual_cross_attention=dual_cross_attention,
                use_linear_projection=use_linear_projection,
                only_cross_attention=only_cross_attention[i],
                upcast_attention=upcast_attention,
                resnet_time_scale_shift=resnet_time_scale_shift,
                use_first_frame=use_first_frame,
            )
            self.watermark_down_blocks.append(down_block)

        reversed_watermark_block_out_channels=list(reversed(watermark_down_block_types))
        watermark_up_block_types=["UpBlock3D","UpBlock3D","UpBlock3D"]
        output_channel=reversed_watermark_block_out_channels[0]
        for i,watermark_up_block_type in enumerate(watermark_up_block_types):
            is_final_block=i==len(watermark_block_out_channels)-1

            prev_output_channel=output_channel
            output_channel=reversed_watermark_block_out_channels[i]
            input_channel=reversed_watermark_block_out_channels[min(i+1,len(watermark_block_out_channels)-1)]
            if not is_final_block:
                add_upsample=True
            else:
                add_upsample=False
            up_block=get_up_block(
                watermark_up_block_type,
                num_layers=layers_per_block+1,
                input_channel=input_channel,
                out_channel=output_channel,
                prev_output_channel=prev_output_channel,
                temb_channels=time_embed_dim,
                add_upsample=add_upsample,
                resnet_ep=norm_eps,
                resnet_fn=act_fn,
                resnet_groups=norm_num_groups,
                cross_attention_dim=cross_attention_dim,
                attn_num_head_channels=reversed_attention_head_dim[i],
                dual_cross_attention=dual_cross_attention,
                use_linear_projection=use_linear_projection,
                only_cross_attention=only_cross_attention[i],
                upcast_attention=upcast_attention,
                resnet_time_scale_shift=resnet_time_scale_shift,
                use_first_frame=use_first_frame,
            )
            self.watermark_up_blocks.append(up_block)
        self.watermark_zero_out=zero_module(InflatedConv3d(block_out_channels[0],block_out_channels[0],kernel_size=1,padding=0))
        self.conv_norm_out=nn.GroupNorm(num_channels=block_out_channels[0],num_groups=norm_num_groups,eps=norm_eps)
        self.conv_act=nn.SiLU()
        self.conv_out=InflatedConv3d(block_out_channels[0],out_channels,kernel_size=3, padding=1)


    def set_attention_slice(self,slice_size):
        slicable_head_dims=[]

        def fn_recursive_retrieve_slicable_dims(module:torch.nn.Module):
            if hasattr(module,"set_attention_slice"):
                slicable_head_dims.append(module.slicable_head_dim)

            for child in module.children:
                fn_recursive_retrieve_slicable_dims(child)

        for module in self.children():
            fn_recursive_retrieve_slicable_dims(module)
        num_slicable_layers=len(slicable_head_dims)
        if slice_size=="auto":
            slice_size=[dim//2 for dim in slicable_head_dims]
        else:
            slice_size=num_slicable_layers*[1]

        slice_size=num_slicable_layers*[slice_size] if not isinstance(slice_size,list) else slice_size

        if len(slice_size)!=len(slicable_head_dims):
            raise ValueError(
                f"You have provided {len(slice_size)}, but {self.config} has {len(slicable_head_dims)} different"
                f" attention layers. Make sure to match `len(slice_size)` to be {len(slicable_head_dims)}."
        )

        for i in range(len(slice_size)):
            size=slice_size[i]
            dim=slicable_head_dims[i]
            if size is not None and size>dim:
                raise ValueError(f"size {size} has to be smaller or equal to {dim}.")

        def fn_recursive_set_attention_slice(module:torch.nn.Module,slice_size:List[int]):
            if hasattr(module,"set_attention_slice"):
                module.set_attention_slice(slice_size.pop())

            for child in module.children():
                fn_recursive_set_attention_slice(child,slice_size)

        reversed_slice_size=list(reversed(slice_size))
        for module in self.children():
            fn_recursive_set_attention_slice(module,reversed_slice_size)

    def _set_gradient_checkpointing(self,module,value=False):
        if isinstance(module,(CrossAttnDownBlock3D,DownBlock3D,CrossAttnUpBlock3D,UpBlock3D)):
            module.gradient_checkpointing=value

    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        # encoder_hidden_states: torch.Tensor,
        encoder_hidden_states=None,
        exemplar_latent=None,
        exemplar_timestep=None,
        exemplar_encoder_hidden_states=None,
        class_labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    )->Union[UNet3DConditionOutput,Tuple]:
        default_overall_up_factor=2**self.num_upsamples

        forward_upsample_size = False
        upsample_size = None

        if any(s%default_overall_up_factor!=0 for s in sample.shape[-2:]):
            logger.info("Forward upsample size to force interpolation output size.")
            forward_upsample_size=True

        if attention_mask is not None:
            attention_mask=(1-attention_mask.to(sample.dtype))*-10000.0
            attention_mask=attention_mask.unsqueeze(1)

        if self.config.center_input_sample:
            sample=2*sample-1.0

        #time
        timesteps=timestep
        if not torch.is_tensor(timesteps):
            is_mps=sample.device.type=='mps'
            if isinstance(timesteps,float):
                dtype=torch.float32 if is_mps else torch.float64
            else:
                dtype=torch.int32 if is_mps else torch.int64
            timesteps=torch.tensor([timesteps],dtype=dtype,device=sample.device)
        elif len(timesteps.shape)==0:
            timesteps=timesteps[None].to(sample.device)

        timesteps=timesteps.expand(sample.shape[0])

        t_emb= self.time_proj(timesteps)
        t_emb=t_emb.to(dtype=self.dtype)

        emb=self.time_embedding(t_emb)

        if self.class_embedding is not None:
            if class_labels is None:
                raise ValueError("class labels should not be None")

            if self.config.class_embed_type=="timesteps":
                class_labels=self.time_proj(class_labels)

            class_emb=self.class_embedding(class_labels).to(dtype=self.dtype)
            emb=emb+class_emb

        #pre_process
        sample=self.conv_in(sample)

        if (exemplar_latent is not None) and (exemplar_timestep is not None):
            examplar_latent=self.conv_in(exemplar_latent)
            #time
            exemplar_timesteps=exemplar_timestep
            if not torch.is_tensor(exemplar_timesteps):
                is_mps = sample.device.type == 'mps'
                if isinstance(exemplar_timesteps, float):
                    dtype = torch.float32 if is_mps else torch.float64
                else:
                    dtype = torch.int32 if is_mps else torch.int64
                exemplar_timesteps = torch.tensor([exemplar_timesteps], dtype=dtype, device=sample.device)
            elif len(exemplar_timesteps.shape) ==0:
                exemplar_timesteps = exemplar_timesteps[None].to(examplar_latent.device)

            exemplar_timesteps = exemplar_timesteps.expand(exemplar_latent.shape[0])

            exemplar_t_emb=self.time_proj(exemplar_timesteps)
            exemplar_t_emb=exemplar_t_emb.to(self.dtype)
            exemplar_emb=self.time_embedding(exemplar_t_emb)
        else:
            exemplar_emb=None

        #down
        down_res_sample=(sample,)
        if (exemplar_latent is not None) and (exemplar_timestep is not None):
            down_res_sample_exemplar=(exemplar_latent,)
        for down_block in self.down_blocks:
            if hasattr(down_block,"has_cross_attention") and down_block.has_cross_attention:
                if (exemplar_latent is not None) and (exemplar_timestep is not None):
                    sample,res_samples,exemplar_latent,res_exemplar_latent=down_block(
                        hidden_states=sample,
                        temb=emb,
                        encoder_hidden_states=encoder_hidden_states,
                        attention_mask=attention_mask,
                        examplar_latent=examplar_latent,
                        exemplar_temb=exemplar_emb,
                        exemplar_encoder_hidden_states=exemplar_encoder_hidden_states,
                    )
                else:
                    sample,res_samples=down_block(
                        hidden_states=sample,
                        temb=emb,
                        attention_mask=attention_mask,
                        encoder_hidden_states=encoder_hidden_states,
                        exemplar_latent=exemplar_latent,
                        exemplar_temb=exemplar_emb,

                    )
            else:
                sample,res_samples=down_block(hidden_states=sample,temb=emb)
                if (exemplar_latent is not None) and (exemplar_timestep is not None):
                    examplar_latent,res_exemplar_latent=down_block(hidden_states=exemplar_latent,t_emb=exemplar_emb)
            down_res_sample+=res_samples
            if (exemplar_latent is not None) and (exemplar_timestep is not None):
                down_res_sample_exemplar+=res_exemplar_latent

        #mid
        if (exemplar_latent is not None) and (exemplar_timestep is not None):
            sample,res_samples,exemplar_latent,res_exemplar_latent=self.mid_block(
                sample,
                emb,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                exemplar_latent=exemplar_latent,
                exemplar_temb=exemplar_emb,
                exemplar_encoder_hidden_states=exemplar_encoder_hidden_states,
            )

            # up
            for i, upsample_block in enumerate(self.up_blocks):
                is_final_block = i == len(self.up_blocks) - 1

                res_samples = down_res_sample[-len(upsample_block.resnets):]
                down_res_sample = down_res_sample[: -len(upsample_block.resnets)]

                if (exemplar_latent is not None) and (exemplar_timestep is not None):
                    res_exemplar_latents = down_res_sample_exemplar[-len(upsample_block.resnets):]
                    down_res_sample_exemplar = down_res_sample_exemplar[: -len(upsample_block.resnets)]

                # if we have not reached the final block and need to forward the
                # upsample size, we do it here
                if not is_final_block and forward_upsample_size:
                    upsample_size = down_res_sample[-1].shape[2:]

                if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                    if (exemplar_latent is not None) and (exemplar_timestep is not None):
                        sample, exemplar_latent = upsample_block(
                            hidden_states=sample,
                            temb=emb,
                            res_hidden_states_tuple=res_samples,
                            encoder_hidden_states=encoder_hidden_states,
                            upsample_size=upsample_size,
                            attention_mask=attention_mask,
                            exemplar_latent=exemplar_latent,
                            res_exemplar_latents_tuple=res_exemplar_latents,
                            exemplar_temb=exemplar_emb,
                            exemplar_encoder_hidden_states=exemplar_encoder_hidden_states
                        )
                    else:
                        sample = upsample_block(
                            hidden_states=sample,
                            temb=emb,
                            res_hidden_states_tuple=res_samples,
                            encoder_hidden_states=encoder_hidden_states,
                            upsample_size=upsample_size,
                            attention_mask=attention_mask,
                            exemplar_latent=exemplar_latent
                        )
                else:
                    if (exemplar_latent is not None) and (exemplar_timestep is not None):
                        exemplar_latent = upsample_block(
                            hidden_states=exemplar_latent, temb=exemplar_emb,
                            res_hidden_states_tuple=res_exemplar_latents, upsample_size=upsample_size
                        )
                    sample = upsample_block(
                        hidden_states=sample, temb=emb, res_hidden_states_tuple=res_samples, upsample_size=upsample_size
                    )

                sample_ori_branch=sample.clone()

                #post process
                water_mark_down_block_res_samples=(sample,)
                for watermark_downsample_block in self.watermark_down_blocks:
                    sample,res_samples=watermark_downsample_block(
                        hidden_states=sample,
                        temb=emb,
                    )
                water_mark_down_block_res_samples+=res_samples
                for i,watermark_upsample_block in self.watermark_up_blocks:
                    is_final_block=i==len(self.watermark_up_blocks)-1
                    res_samples=water_mark_down_block_res_samples[-len(watermark_upsample_block.resnet):]
                    water_mark_down_block_res_samples=water_mark_down_block_res_samples[:-len(water_mark_down_block_res_samples)]
                    if not is_final_block:
                        upsample_size=water_mark_down_block_res_samples[-1].shape[-2:]
                    sample=watermark_upsample_block(
                        hidden_states=sample,
                        temb=emb,
                        res_hiddent_states_tuple=res_samples,
                        upsample_size=upsample_size,
                    )
                sample=self.watermark_zero_out(sample)
                sample=self.conv_norm_out(sample_ori_branch+sample)
                sample=self.conv_act(sample)
                sample=self.conv_out(sample)
                if not return_dict:
                    return (sample,)
                return UNet3DConditionOutput(sample=sample)

def zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module

