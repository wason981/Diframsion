import os
import math
import torch
import torch.nn as nn
from einops import repeat

#################################################################################
#                                  Unet Utils                                   #
#################################################################################

def checkpoint(func, inputs, params, flag: bool) -> torch.Tensor:
    """
    Evaluate a function without caching intermediate activations, allowing for
    reduced memory at the expense of extra compute in the backward pass.

    :param func: The function to evaluate.
    :param inputs: The argument sequence to pass to `func`.
    :param params: A sequence of parameters `func` depends on but does not
                   explicitly take as arguments.
    :param flag: If False, disable gradient checkpointing.
    :return: The output of the evaluated function.
    """
    if flag:
        args = tuple(inputs) + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), *args)
    else:
        return func(*inputs)


class CheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])

        with torch.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]
        with torch.enable_grad():
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)
        input_grads = torch.autograd.grad(
            output_tensors,
            ctx.input_tensors + ctx.input_params,
            output_grads,
            allow_unused=True,
        )
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors
        return (None, None) + input_grads


def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000, repeat_only: bool = False, device: str = 'cpu') -> torch.Tensor:
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: A 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: The dimension of the output.
    :param max_period: Controls the minimum frequency of the embeddings.
    :param device: The device to create the tensors on.
    :return: An [N x dim] Tensor of positional embeddings.
    """
    if not repeat_only:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device)
        args = timesteps[:, None].float() * freqs[None].to(device)
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    else:
        embedding = repeat(timesteps, 'b -> b d', d=dim).contiguous()
    return embedding


def zero_module(module: nn.Module) -> nn.Module:
    """
    Zero out the parameters of a module and return it.

    :param module: The nn.Module to zero out.
    :return: The module with zeroed parameters.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def scale_module(module: nn.Module, scale: float) -> nn.Module:
    """
    Scale the parameters of a module and return it.

    :param module: The nn.Module to scale.
    :param scale: The scaling factor.
    :return: The scaled module.
    """
    for p in module.parameters():
        p.detach().mul_(scale)
    return module


def mean_flat(tensor: torch.Tensor) -> torch.Tensor:
    """
    Take the mean over all non-batch dimensions.

    :param tensor: Input tensor.
    :return: Mean of the input tensor.
    """
    return tensor.mean(dim=list(range(1, tensor.ndim)))


def normalization(channels: int) -> nn.Module:
    """
    Create a standard normalization layer.

    :param channels: Number of input channels.
    :return: An nn.Module for normalization.
    """
    return GroupNorm32(32, channels)


class SiLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class GroupNorm32(nn.GroupNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.float()).type(x.dtype)


def conv_nd(dims: int, *args, **kwargs) -> nn.Module:
    """
    Create a 1D, 2D, or 3D convolution module.

    :param dims: The number of dimensions (1, 2, or 3).
    :return: The convolutional layer.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"Unsupported dimensions: {dims}, must be 1, 2, or 3.")


def linear(*args, **kwargs) -> nn.Module:
    """
    Create a linear module.

    :return: The linear layer.
    """
    return nn.Linear(*args, **kwargs)


def avg_pool_nd(dims: int, *args, **kwargs) -> nn.Module:
    """
    Create a 1D, 2D, or 3D average pooling module.

    :param dims: The number of dimensions (1, 2, or 3).
    :return: The average pooling layer.
    """
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"Unsupported dimensions: {dims}, must be 1, 2, or 3.")


def noise_like(shape: tuple, device: str, repeat: bool = False) -> torch.Tensor:
    """
    Generate noise of a specified shape.

    :param shape: The shape of the noise tensor.
    :param device: The device to create the tensor on.
    :param repeat: If True, repeat the noise across the batch dimension.
    :return: A tensor filled with random noise.
    """
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()


def count_flops_attn(model: nn.Module, _x: tuple, y: tuple) -> None:
    """
    A counter for the `thop` package to count the operations in an attention operation.

    :param model: The model to profile.
    :param _x: The input tensor(s).
    :param y: The output tensor(s).
    """
    b, c, *spatial = y[0].shape
    num_spatial = int(np.prod(spatial))
    matmul_ops = 2 * b * (num_spatial ** 2) * c
    model.total_ops += torch.DoubleTensor([matmul_ops])


def count_params(model: nn.Module, verbose: bool = False) -> int:
    """
    Count the total number of parameters in a model.

    :param model: The model to count parameters for.
    :param verbose: If True, print the number of parameters.
    :return: The total number of parameters.
    """
    total_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"{model.__class__.__name__} has {total_params * 1.e-6:.2f} M params.")
    return total_params
