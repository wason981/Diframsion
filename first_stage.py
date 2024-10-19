import os
import json
import glob
import argparse
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler
from time import time
from torchvision import transforms
from your_module import (UNet3DConditionModel, AutoencoderKL,
                         create_logger, create_tensorboard,
                         get_dataset, update_ema,
                         validation, requires_grad, freeze_params,
                         unfreeze_params, write_tensorboard,
                         create_diffusion, clip_grad_norm_,
                         Mapper, inj_forward_text, inj_forward_crossattention)


def setup_distributed_training(args):
    """Initialize distributed training."""
    assert torch.cuda.is_available(), "Training requires at least one GPU."
    dist.init_process_group("nccl")
    assert args.global_batch_size % dist.get_world_size() == 0, "Batch size must be divisible by world size."


def create_experiment_directory(args, rank):
    """Create the experiment directory for saving results."""
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)
        experiment_index = len(glob.glob(f"{args.results_dir}/*"))
        num_frame_string = f'F{args.num_frames}S{args.frame_interval}'
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{num_frame_string}-{args.dataset}"

        # Add suffixes based on flags
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

        with open(f'{experiment_dir}/config.json', 'wt') as f:
            json.dump(vars(args), f, indent=4)

        logger.info(f"Experiment directory created at {experiment_dir}")
        return experiment_dir, checkpoint_dir, logger, tb_writer

    return None, None, None, None


def initialize_model_and_tokenizer(args, device):
    """Initialize the model and tokenizer."""
    latent_size = args.image_size // 8
    args.latent_size = latent_size

    unet = UNet3DConditionModel.from_pretrained_2d(pretrained_model_path='./pretrained/stable-diffusion-v1-4',
                                                   subfolder="unet").to(device)
    state_dict = torch.load(args.pretrained_t2v_model, map_location=device)["ema"]
    unet.load_state_dict(state_dict)

    diffusion = create_diffusion(timestep_respacing="")
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(device)

    for _module in text_encoder.modules():
        if isinstance(_module, CLIPTextTransformer):
            _module.__class__.__call__ = inj_forward_text

    image_encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14").to(device)

    mapper = Mapper(input_dim=1024, output_dim=768).to(device)
    setup_cross_attention(unet, mapper, args.global_mapper_path)

    return unet, vae, tokenizer, text_encoder, image_encoder, mapper


def setup_cross_attention(unet, mapper, global_mapper_path):
    """Setup cross-attention layers in the UNet model."""
    for _name, _module in unet.named_modules():
        if isinstance(_module, CrossAttention):
            if 'attn1' in _name or 'attn_temp' in _name:
                continue

            _module.__class__.forward = inj_forward_crossattention
            mapper.add_module(f'{_name.replace(".", "_")}_to_k', create_global_layer(_module.to_k))
            mapper.add_module(f'{_name.replace(".", "_")}_to_v', create_global_layer(_module.to_v))

            if global_mapper_path is None:
                _module.add_module('to_k_global', mapper[f'{_name.replace(".", "_")}_to_k'])
                _module.add_module('to_v_global', mapper[f'{_name.replace(".", "_")}_to_v'])
            else:
                load_global_mapper(global_mapper_path, mapper, _module, _name)


def create_global_layer(layer):
    """Create a global layer from the existing layer."""
    shape = layer.weight.shape
    global_layer = nn.Linear(shape[1], shape[0], bias=False)
    global_layer.weight.data = layer.weight.data.clone()
    return global_layer


def load_global_mapper(global_mapper_path, mapper, _module, _name):
    """Load global mapper state dict."""
    state_dict = torch.load(global_mapper_path, map_location='cpu')
    for k, v in mapper.state_dict().items():
        if 'to_k' in k or 'to_v' in k:
            state_dict[k] = v
    mapper.load_state_dict(state_dict)

    _module.add_module('to_k_global', getattr(mapper, f'{_name.replace(".", "_")}_to_k'))
    _module.add_module('to_v_global', getattr(mapper, f'{_name.replace(".", "_")}_to_v'))


def setup_optimizer(mapper):
    """Setup optimizer for the model."""
    return torch.optim.AdamW(mapper.parameters(), lr=1e-4, weight_decay=0)


def prepare_data_loader(args, dataset, rank):
    """Prepare the data loader."""
    sampler = DistributedSampler(dataset, num_replicas=dist.get_world_size(), rank=rank, shuffle=True,
                                 seed=args.global_seed)
    return DataLoader(dataset, batch_size=int(args.global_batch_size // dist.get_world_size()), shuffle=False,
                      sampler=sampler, num_workers=args.num_workers, pin_memory=True, drop_last=True)


def log_training_info(train_steps, epoch, running_loss, running_loss_ldm, running_loss_reg, running_loss_reg_text,
                      log_steps, start_time, args, logger, tb_writer):
    """Log training information."""
    if train_steps % args.log_every == 0:
        torch.cuda.synchronize()
        end_time = time()
        steps_per_sec = log_steps / (end_time - start_time)

        avg_loss = running_loss / log_steps
        avg_loss_ldm = running_loss_ldm / log_steps
        avg_loss_reg = running_loss_reg / log_steps
        avg_loss_reg_text = running_loss_reg_text / log_steps

        logger.info(
            f"(step={train_steps:07d}/epoch={epoch:04d}) Train Loss: {avg_loss:.4f}, Loss ldm: {avg_loss_ldm:.4f}, Loss reg: {avg_loss_reg:.4f}, Loss reg text: {avg_loss_reg_text:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")

        write_tensorboard(tb_writer, 'Train Loss', avg_loss, train_steps)
        write_tensorboard(tb_writer, 'Loss ldm', avg_loss_ldm, train_steps)
        write_tensorboard(tb_writer, 'Loss reg', avg_loss_reg, train_steps)
        write_tensorboard(tb_writer, 'Loss reg text', avg_loss_reg_text, train_steps)

        return 0, 0, 0, 0, start_time  # Reset metrics


def main(args):
    setup_distributed_training(args)
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)

    experiment_dir, checkpoint_dir, logger, tb_writer = create_experiment_directory(args, rank)

    unet, vae, tokenizer, text_encoder, image_encoder, mapper = initialize_model_and_tokenizer(args, device)

    # Freeze models
    freeze_params(vae.parameters())
    freeze_params(unet.parameters())
    freeze_params(text_encoder.parameters())
    freeze_params(image_encoder.parameters())
    unfreeze_params(mapper.parameters())

    logger.info(f"Model Parameters: {sum(p.numel() for p in mapper.parameters()):,}")
    opt = setup_optimizer(mapper)

    dataset = get_dataset(args)
    loader = prepare_data_loader(args, dataset, rank)

    logger.info(f"Dataset contains {len(dataset):,} videos.")

    # Training loop
    train_steps = 0
    log_steps = 0
    running_loss = 0
    running_loss_ldm = 0
    running_loss_reg = 0
    running_loss_reg_text = 0
    start_time = time()

    diffusion = create_diffusion(str(args.num_sampling_steps))
    normalize_exemplar = transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                              (0.26862954, 0.26130258, 0.27577711))

    for epoch in range(args.num_epochs):
        for step, (input, text_input, motion_input, image_input) in enumerate(loader):
            input, text_input, motion_input, image_input = input.to(device), text_input.to(device), motion_input.to(
                device), image_input.to(device)

            if step < args.num_warmup_steps:
                requires_grad(mapper.parameters(), False)
            else:
                requires_grad(mapper.parameters(), True)

            output = mapper(input, text_input, motion_input, diffusion)
            loss = calculate_loss(output, image_input)

            opt.zero_grad()
            loss.backward()
            clip_grad_norm_(mapper.parameters(), max_norm=1.0)
            opt.step()

            # Update running loss
            running_loss += loss.item()
            running_loss_ldm += ldm_loss(output)
            running_loss_reg += reg_loss(output)
            running_loss_reg_text += reg_text_loss(output)

            train_steps += 1
            log_steps += 1

            # Log training info
            if log_steps >= args.log_every:
                running_loss, running_loss_ldm, running_loss_reg, running_loss_reg_text, start_time = log_training_info(
                    train_steps, epoch, running_loss, running_loss_ldm, running_loss_reg, running_loss_reg_text,
                    log_steps, start_time, args, logger, tb_writer)

        if epoch % args.save_every == 0 and rank == 0:
            save_checkpoint(mapper, checkpoint_dir, epoch)

    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DiT Training")
    # Add your arguments here
    args = parser.parse_args()
    main(args)
