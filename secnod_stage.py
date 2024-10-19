import os
import glob
import json
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
import torch.nn.functional as F
from time import time
from einops import rearrange


def setup_ddp(args):
    """Setup distributed data parallel (DDP) environment."""
    dist.init_process_group("nccl")
    assert args.global_batch_size % dist.get_world_size() == 0, "Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    torch.manual_seed(args.global_seed + rank)
    torch.cuda.set_device(device)
    return rank, device


def create_experiment_directory(args, rank):
    """Create an experiment directory and return its path."""
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)
        experiment_index = len(glob.glob(f"{args.results_dir}/*"))
        num_frame_string = f'F{args.num_frames}S{args.frame_interval}'
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{num_frame_string}-{args.dataset}"

        if args.class_guided:
            experiment_dir += '-Class'
        if args.use_compile:
            experiment_dir += '-Compile'
        if args.use_timecross_transformer:
            experiment_dir += '-TimeCross'

        checkpoint_dir = f"{experiment_dir}/checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        tb_writer = create_tensorboard(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")

        with open(f'{experiment_dir}/config.json', 'wt') as f:
            json.dump(vars(args), f, indent=4)

        return experiment_dir, checkpoint_dir, logger, tb_writer
    else:
        return None, None, create_logger(None), None


def load_models(device, args):
    """Load necessary models and return them."""
    unet = UNet3DConditionModel.from_pretrained_2d(pretrained_model_path='./pretrained/stable-diffusion-v1-4',
                                                   subfolder="unet").to(device)
    state_dict = torch.load(args.pretrained_t2v_model, map_location='cuda:0')["ema"]
    unet.load_state_dict(state_dict)

    diffusion = create_diffusion(timestep_respacing="")
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(device)

    # Modify text encoder for custom forward
    for module in text_encoder.modules():
        if isinstance(module, CLIPTextTransformer):
            module.__call__ = inj_forward_text

    image_encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    return unet, diffusion, vae, tokenizer, text_encoder, image_encoder


def initialize_mapper(unet, args):
    """Initialize the mapper and modify necessary layers."""
    mapper = Mapper(input_dim=1024, output_dim=768).to(device)

    for name, module in unet.named_modules():
        if isinstance(module, CrossAttention):
            if 'attn1' in name or 'attn_temp' in name:
                continue
            module.forward = inj_forward_crossattention
            add_mapper_layers(module, mapper, name)

        if isinstance(module, SparseCausalAttention):
            if 'attn2' in name or 'attn_temp' in name:
                continue
            module.forward = inj_forward_stattention
            add_mapper_layers(module, mapper, name)

    return mapper


def add_mapper_layers(module, mapper, name):
    """Add necessary layers to the mapper."""
    for layer_type in ['to_k', 'to_v']:
        shape = getattr(module, layer_type).weight.shape
        new_layer = nn.Linear(shape[1], shape[0], bias=False)
        new_layer.weight.data = getattr(module, layer_type).weight.data.clone()
        mapper.add_module(f'{name.replace(".", "_")}_{layer_type}', new_layer)
        module.add_module(f'{layer_type}_global', new_layer)


def main(args):
    """Main training function."""
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    rank, device = setup_ddp(args)
    experiment_dir, checkpoint_dir, logger, tb_writer = create_experiment_directory(args, rank)

    # Model and data loading
    unet, diffusion, vae, tokenizer, text_encoder, image_encoder = load_models(device, args)
    mapper = initialize_mapper(unet, args)

    # Freeze models and prepare data loader
    freeze_params(vae.parameters())
    freeze_params(unet.parameters())
    freeze_params(text_encoder.parameters())
    freeze_params(image_encoder.parameters())
    unfreeze_params(mapper.parameters())
    mapper = DDP(mapper.to(device), device_ids=[rank])

    dataset = get_dataset(args)
    sampler = DistributedSampler(dataset, num_replicas=dist.get_world_size(), rank=rank, shuffle=True,
                                 seed=args.global_seed)
    loader = DataLoader(dataset, batch_size=args.global_batch_size // dist.get_world_size(), sampler=sampler,
                        num_workers=args.num_workers, pin_memory=True, drop_last=True)

    logger.info(f"Dataset contains {len(dataset):,} videos.")

    # Training loop
    start_time = time()
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        for video_data in loader:
            # Process each video batch
            process_video_batch(video_data, device, mapper, unet, vae, text_encoder, image_encoder, tokenizer,
                                diffusion, logger)


def process_video_batch(video_data, device, mapper, unet, vae, text_encoder, image_encoder, tokenizer, diffusion,
                        logger):

    pass



