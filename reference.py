import os
import random
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
from omegaconf import OmegaConf
from torch import nn
from einops import rearrange


# Define the validation function
def validation(diffusion, unet, mapper, image_encoder, tokenizer, video_data, text_encoder, vae, cfg_scale, device,
               save_dir):
    """Validate the model by generating videos based on input data."""
    normalize_exemplar = transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                              (0.26862954, 0.26130258, 0.27577711))

    z = torch.randn(video_data['video'].size(0), 4, 16, 32, 32).to(device)
    image = torch.nn.functional.interpolate(video_data["masked_first_frame"].to(device), (224, 224), mode='bilinear')
    image = normalize_exemplar(image)

    # Get the image embeddings
    image_features = image_encoder(image, output_hidden_states=True)
    image_embeddings = [image_features[0], image_features[2][4], image_features[2][8], image_features[2][12],
                        image_features[2][16]]
    image_embeddings = [emb.detach() for emb in image_embeddings]
    inj_embedding = mapper(image_embeddings)

    # Tokenize video names and word prompts
    original_ids = tokenizer(
        video_data['video_name'],
        padding="max_length",
        truncation=True,
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    )["input_ids"].to(device)

    placeholder_idx = tokenizer(video_data['word_prompt'], add_special_tokens=False)["input_ids"]

    # Get text embeddings for conditioning
    encoder_hidden_states_con = text_encoder({'input_ids': original_ids,
                                              "inj_embedding": inj_embedding,
                                              "inj_index": placeholder_idx})[0]

    # Get null text embeddings
    null_ids = tokenizer(["None"] * video_data['video'].size(0),
                         padding="max_length",
                         truncation=True,
                         max_length=tokenizer.model_max_length,
                         return_tensors="pt",
                         )["input_ids"].to(device)

    encoder_hidden_states_uncon = text_encoder({'input_ids': null_ids})[0]
    encoder_hidden_states = torch.cat([encoder_hidden_states_con, encoder_hidden_states_uncon], dim=0)

    # Get the exemplar latent representation
    x = video_data['video'].to(device)
    with torch.no_grad():
        b, _, _, _, _ = x.shape
        x = rearrange(x, 'b f c h w -> (b f) c h w').contiguous()
        x = vae.encode(x).latent_dist.sample().mul_(0.18215)
        x = rearrange(x, '(b f) c h w -> b c f h w', b=b).contiguous()  # for tav unet; b c f h w is for conv3d
        exemplar_latent = x[:, :, :1, :, :].repeat(1, 1, 16, 1, 1)

    model_kwargs = dict(encoder_hidden_states=encoder_hidden_states, class_labels=None, cfg_scale=cfg_scale,
                        exemplar_latent_ori=exemplar_latent, add_noise_to_exemplar=True)

    z = torch.cat([z, z], 0)

    # Sample videos using the diffusion process
    samples = diffusion.p_sample_loop(
        unet.forward_with_cfg_with_exemplar, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True,
        device=device)

    samples, _ = samples.chunk(2, dim=0)

    # Decode the samples
    samples = rearrange(samples, 'b c f h w -> (b f) c h w')
    samples = vae.decode(samples / 0.18215).sample
    samples = rearrange(samples, '(b f) c h w -> b f c h w', b=b)

    os.makedirs(save_dir, exist_ok=True)

    # Save the generated videos and images
    for batch_idx in range(b):
        video_ = ((samples[batch_idx] * 0.5 + 0.5) * 255).clamp_(0, 255).to(dtype=torch.uint8).cpu().permute(0, 2, 3,
                                                                                                             1).contiguous()
        torchvision.io.write_video(f'{save_dir}/sampled_video_{batch_idx}.mp4', video_, fps=8)
        torchvision.utils.save_image(video_data["masked_first_frame"][batch_idx],
                                     f'{save_dir}/image_prompt_{batch_idx}.png', normalize=True, value_range=(0, 1))

    # Save prompts to text files
    with open(f'{save_dir}/prompts.txt', 'w') as file:
        for prompt in video_data['video_name']:
            file.write(f'{prompt}\n')

    with open(f'{save_dir}/replaced_words.txt', 'w') as file:
        for word in video_data['word_prompt']:
            file.write(f'{word}\n')


def load_data_pair(config):
    """Load and preprocess data pairs of images and masks."""
    try:
        img_random_trans = transforms.Compose([transforms.Resize([224, 224]), ])
        first_frame_random_trans = transforms.Compose([transforms.Resize([256, 256]), ])

        mask = np.array(Image.open(config.mask_path))
        first_frame = np.array(Image.open(config.img_path))

        # Prepare the masked first frame
        masked_first_frame = first_frame.copy()
        masked_first_frame[mask == 0] = 255
        x1, y1, x2, y2 = config.bbox
        masked_first_frame = masked_first_frame[int(y1):int(y2), int(x1):int(x2), :]

        # Convert to tensor and pad
        masked_first_frame = torch.from_numpy(masked_first_frame).permute(2, 0, 1).contiguous()
        height, width = masked_first_frame.size(1), masked_first_frame.size(2)

        if height < width:
            diff = width - height
            top_pad = diff // 2
            down_pad = diff - top_pad
            padding_size = [0, top_pad, 0, down_pad]  # [left, top, right, bottom]
            masked_first_frame = F.pad(masked_first_frame, padding=padding_size, fill=255)
        else:
            diff = height - width
            left_pad = diff // 2
            right_pad = diff - left_pad
            padding_size = [left_pad, 0, right_pad, 0]
            masked_first_frame = F.pad(masked_first_frame, padding=padding_size, fill=255)

        # Normalize and augment
        masked_first_frame_ori = masked_first_frame.clone()
        masked_first_frame = img_random_trans(masked_first_frame) / 255.0
        aug_first_frame = first_frame_random_trans(masked_first_frame_ori).unsqueeze(0) / 127.5 - 1

        return {'video': aug_first_frame, 'masked_first_frame': masked_first_frame}
    except Exception as e:
        print(f"Error loading data: {e}")
        return None


def set_random_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main(args):
    """Main function to run the validation process."""
    config = OmegaConf.load(args.config)

    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load models
    pretrained_t2v_model_path = './pretrained/watermark_remove_module.pt'
    unet = UNet3DConditionModelWaterMark.from_pretrained_2d(pretrained_model_path='./pretrained/stable-diffusion-v1-4',
                                                            subfolder="unet").to(device)
    state_dict = torch.load(pretrained_tre'f2v_model_path, map_location='cpu')["ema_unet"]
    unet.load_state_dict(state_dict, strict=False)

    # Set up diffusion model and other components
    num_sampling_steps = 250
    diffusion = create_diffusion(str(num_sampling_steps))  # default: 1000 steps, linear noise schedule
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(device)
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(device)

    # Modify the text encoder if necessary
    for _module in text_encoder.modules():
        if _module.__class__.__name__ == "CLIPTextTransformer":
            _module.__class__.__call__ = inj_forward_text

    image_encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    mapper = Mapper(input_dim=1024, output_dim=768).to(device)

    # Adjust UNet layers
    for _name, _module in unet.named_modules():
        if _module.__class__.__name__ == "CrossAttention":
            if 'attn1' in _name or 'attn_temp' in _name:
