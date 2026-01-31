"""
SDS-based Image Simplification using SDXL (1024x1024)

This version uses Stable Diffusion XL for higher resolution processing.
- Supports 1024x1024 native resolution (vs 512x512 for SD 1.5)
- Loads from local .safetensors checkpoint
- Better detail preservation for high-resolution images

For app3/v3 experimental pipeline.
"""

from typing import Tuple, Union, Optional, List
import torch
from torch.optim.sgd import SGD
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel
import numpy as np
from PIL import Image
from tqdm import tqdm

T = torch.Tensor
TN = Optional[T]
TS = Union[Tuple[T, ...], List[T]]

# SDXL model - use HuggingFace for reliable separate component loading
# Alternative: local checkpoint (has issues with CPU offload)
# SDXL_CHECKPOINT_PATH = "checkpoints/sdxl/realvisxlV50_v50Bakedvae.safetensors"
SDXL_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"

# Use fp16-fixed VAE to avoid NaN issues with float16 inference
# The standard SDXL VAE has numerical instability in fp16
SDXL_VAE_FP16_FIX = "madebyollin/sdxl-vae-fp16-fix"

# SDXL native resolution
SDXL_RESOLUTION = 1024


def load_image_sdxl(image_path: str, target_size: int = SDXL_RESOLUTION, left=0, right=0, top=0, bottom=0):
    """
    Load and resize image for SDXL processing.
    Preserves aspect ratio by padding to square, then resizing.

    Args:
        image_path: Path to input image
        target_size: Target resolution (default 1024 for SDXL)

    Returns:
        numpy array of shape (target_size, target_size, 3)
    """
    # Load original image
    img = Image.open(image_path).convert('RGB')
    orig_width, orig_height = img.size

    # Calculate aspect ratio and determine padding
    max_dim = max(orig_width, orig_height)

    # Create square canvas with white background
    square_img = Image.new('RGB', (max_dim, max_dim), (255, 255, 255))

    # Paste original image centered
    paste_x = (max_dim - orig_width) // 2
    paste_y = (max_dim - orig_height) // 2
    square_img.paste(img, (paste_x, paste_y))

    # Resize to target size
    resized = square_img.resize((target_size, target_size), Image.LANCZOS)

    # Convert to numpy array
    image = np.array(resized)[:, :, :3]

    return image, (orig_width, orig_height, paste_x, paste_y, max_dim)


def load_image_sdxl_stretch(image_path: str, target_size: int = SDXL_RESOLUTION):
    """
    Load and resize image for SDXL processing (stretch method - original behavior).
    Simply resizes to square, distorting aspect ratio.

    Args:
        image_path: Path to input image
        target_size: Target resolution (default 1024 for SDXL)

    Returns:
        numpy array of shape (target_size, target_size, 3)
    """
    image = np.array(Image.open(image_path).convert('RGB').resize((target_size, target_size)))[:, :, :3]
    return image, None


@torch.no_grad()
def get_text_embeddings_sdxl(device, pipe: StableDiffusionXLPipeline, text: str) -> Tuple[T, T]:
    """
    Get text embeddings for SDXL (requires both text encoders).
    """
    # SDXL uses two text encoders
    tokens_1 = pipe.tokenizer(
        [text],
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt"
    ).input_ids.to(device)

    tokens_2 = pipe.tokenizer_2(
        [text],
        padding="max_length",
        max_length=pipe.tokenizer_2.model_max_length,
        truncation=True,
        return_tensors="pt"
    ).input_ids.to(device)

    # Get embeddings from both encoders
    prompt_embeds_1 = pipe.text_encoder(tokens_1, output_hidden_states=True)
    prompt_embeds_2 = pipe.text_encoder_2(tokens_2, output_hidden_states=True)

    # SDXL combines embeddings from both encoders
    # Use hidden_states[-2] (penultimate layer) as per diffusers implementation
    prompt_embeds = torch.cat([
        prompt_embeds_1.hidden_states[-2],
        prompt_embeds_2.hidden_states[-2]
    ], dim=-1)

    # Pooled embeddings come from text_encoder_2's pooler_output (not hidden_states!)
    # This is used in added_cond_kwargs["text_embeds"]
    pooled_prompt_embeds = prompt_embeds_2.text_embeds

    return prompt_embeds.detach(), pooled_prompt_embeds.detach()


@torch.no_grad()
def denormalize(image):
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    image = (image * 255).astype(np.uint8)
    return image[0]


@torch.no_grad()
def decode_sdxl(latent: T, pipe: StableDiffusionXLPipeline):
    """Decode latent to image using SDXL VAE."""
    # SDXL uses scaling factor of 0.13025 (different from SD 1.5's 0.18215)
    # Convert to float16 for VAE decoding (VAE is in float16)
    latent_fp16 = latent.half() / pipe.vae.config.scaling_factor
    image = pipe.vae.decode(latent_fp16, return_dict=False)[0]
    image = denormalize(image)
    return image


def init_pipe_sdxl(device, dtype, unet, scheduler) -> Tuple[UNet2DConditionModel, T, T]:
    """Initialize SDXL pipeline components."""
    with torch.inference_mode():
        alphas = torch.sqrt(scheduler.alphas_cumprod).to(device, dtype=dtype)
        sigmas = torch.sqrt(1 - scheduler.alphas_cumprod).to(device, dtype=dtype)
    for p in unet.parameters():
        p.requires_grad = False
    return unet, alphas, sigmas


class SDSLossSDXL:
    """SDS Loss for SDXL models."""

    def noise_input(self, z, eps=None, timestep: Optional[int] = None):
        if timestep is None:
            b = z.shape[0]
            timestep = torch.randint(
                low=self.t_min,
                high=min(self.t_max, 1000) - 1,
                size=(b,),
                device=z.device, dtype=torch.long)
        if eps is None:
            eps = torch.randn_like(z)
        alpha_t = self.alphas[timestep, None, None, None]
        sigma_t = self.sigmas[timestep, None, None, None]
        z_t = alpha_t * z + sigma_t * eps
        return z_t, eps, timestep, alpha_t, sigma_t

    def get_eps_prediction(self, z_t: T, timestep: T, text_embeddings: T, pooled_embeddings: T,
                           alpha_t: T, sigma_t: T, get_raw=False, guidance_scale=1,
                           add_time_ids: T = None):
        """Get noise prediction from SDXL UNet.

        DTYPE STRATEGY:
        - Convert all UNet inputs to float16 explicitly (no autocast)
        - Run UNet forward pass in float16
        - Convert output to float32 for post-processing
        - This avoids autocast mixed precision issues that cause NaN/Inf
        """
        # For guidance_scale=0, we only need unconditional prediction (single batch)
        # This matches SD 1.5 behavior where e_t = e_t_uncond when guidance_scale=0
        if guidance_scale == 0:
            # Single batch - just unconditional
            latent_input = z_t
            timestep_input = timestep
            # Use only the first (null) embedding
            embedd = text_embeddings[0:1] if text_embeddings.shape[0] > 1 else text_embeddings
            pooled = pooled_embeddings[0:1] if pooled_embeddings.shape[0] > 1 else pooled_embeddings
        else:
            # Double batch for CFG
            latent_input = torch.cat([z_t] * 2)
            timestep_input = torch.cat([timestep] * 2)
            embedd = text_embeddings
            pooled = pooled_embeddings

        # SDXL requires additional time embeddings
        if add_time_ids is None:
            add_time_ids = torch.tensor([[1024, 1024, 0, 0, 1024, 1024]], device=z_t.device, dtype=torch.float16)
        if guidance_scale != 0 and add_time_ids.shape[0] == 1:
            add_time_ids = add_time_ids.repeat(2, 1)

        # Explicitly convert all inputs to float16 for UNet (avoids autocast issues)
        latent_input_fp16 = latent_input.half()
        embedd_fp16 = embedd.half()
        pooled_fp16 = pooled.half()
        add_time_ids_fp16 = add_time_ids.half()

        added_cond_kwargs = {
            "text_embeds": pooled_fp16,
            "time_ids": add_time_ids_fp16
        }

        # Debug: print input shapes and values on first call
        if not hasattr(self, '_debug_printed'):
            self._debug_printed = True
            print(f"DEBUG UNet inputs (all fp16):")
            print(f"  latent_input: shape={latent_input_fp16.shape}, dtype={latent_input_fp16.dtype}, "
                  f"min={latent_input_fp16.min().item():.4f}, max={latent_input_fp16.max().item():.4f}, "
                  f"has_nan={torch.isnan(latent_input_fp16).any().item()}")
            print(f"  timestep_input: {timestep_input}, dtype={timestep_input.dtype}")
            print(f"  embedd: shape={embedd_fp16.shape}, dtype={embedd_fp16.dtype}, "
                  f"has_nan={torch.isnan(embedd_fp16).any().item()}")
            print(f"  pooled: shape={pooled_fp16.shape}, dtype={pooled_fp16.dtype}, "
                  f"has_nan={torch.isnan(pooled_fp16).any().item()}")
            print(f"  add_time_ids: shape={add_time_ids_fp16.shape}, dtype={add_time_ids_fp16.dtype}")
            print(f"  prediction_type: {self.prediction_type}")

        # Run UNet without autocast - inputs are already fp16
        e_t = self.unet(
            latent_input_fp16,
            timestep_input,
            encoder_hidden_states=embedd_fp16,
            added_cond_kwargs=added_cond_kwargs
        ).sample

        # Convert to float32 for stable post-processing
        e_t = e_t.float()

        if not hasattr(self, '_debug_printed2'):
            self._debug_printed2 = True
            print(f"  e_t after UNet (fp32): shape={e_t.shape}, dtype={e_t.dtype}, "
                  f"has_nan={torch.isnan(e_t).any().item()}, has_inf={torch.isinf(e_t).any().item()}")

        # Post-processing in float32 for numerical stability
        if self.prediction_type == 'v_prediction':
            e_t = alpha_t.float() * e_t + sigma_t.float() * latent_input.float()

        if guidance_scale != 0:
            e_t_uncond, e_t_cond = e_t.chunk(2)
            e_t = e_t_uncond + guidance_scale * (e_t_cond - e_t_uncond)

        if not torch.isfinite(e_t).all():
            print(f"ERROR: NaN/Inf detected in e_t after processing!")
            e_t = torch.nan_to_num(e_t, nan=0.0, posinf=1e4, neginf=-1e4)

        if get_raw:
            return e_t
        pred_z0 = (z_t.float() - sigma_t.float() * e_t) / alpha_t.float()
        return e_t, pred_z0

    def get_sds_loss(self, z: T, text_embeddings: T, pooled_embeddings: T,
                    eps: TN = None, mask=None, t=None,
                    timestep: Optional[int] = None, guidance_scale=0,
                    add_time_ids: T = None) -> TS:
        with torch.inference_mode():
            z_t, eps, timestep, alpha_t, sigma_t = self.noise_input(z, eps=eps, timestep=timestep)
            e_t, _ = self.get_eps_prediction(
                z_t, timestep, text_embeddings, pooled_embeddings,
                alpha_t, sigma_t, guidance_scale=guidance_scale,
                add_time_ids=add_time_ids
            )
            grad_z = (alpha_t ** self.alpha_exp) * (sigma_t ** self.sigma_exp) * (e_t - eps)
            assert torch.isfinite(grad_z).all()
            grad_z = torch.nan_to_num(grad_z.detach(), 0.0, 0.0, 0.0)
            if mask is not None:
                grad_z = grad_z * mask
            log_loss = (grad_z ** 2).mean()
        sds_loss = grad_z.clone() * z
        del grad_z
        return sds_loss.sum() / (z.shape[2] * z.shape[3]), log_loss

    def __init__(self, device, pipe: StableDiffusionXLPipeline, dtype=torch.float32):
        self.t_min = 50
        self.t_max = 950
        self.alpha_exp = 0
        self.sigma_exp = 0
        self.dtype = dtype
        self.unet, self.alphas, self.sigmas = init_pipe_sdxl(device, dtype, pipe.unet, pipe.scheduler)
        self.prediction_type = pipe.scheduler.config.prediction_type


def image_optimization_sdxl(device, pipe: StableDiffusionXLPipeline, image: np.ndarray,
                            text_target: str, num_iters: int = 200):
    """
    Optimize image using SDS loss with SDXL.
    Follows the same pattern as SD 1.5 version for reliability.

    DTYPE STRATEGY: Use float32 throughout for numerical stability.
    - VAE encoding: float32 input -> float32 latents
    - z_target: float32 for gradient computation
    - UNet: autocast handles internal float16 conversion safely
    - Embeddings: float32 to match z_target
    """
    sds_loss = SDSLossSDXL(device, pipe)

    # Prepare image as float32
    image_source = torch.from_numpy(image).float().permute(2, 0, 1) / 127.5 - 1
    image_source = image_source.unsqueeze(0).to(device)

    with torch.no_grad():
        # Encode to latent space - use float32 for VAE encoding to avoid precision loss
        # The VAE internally uses float16 but we convert output to float32 for stability
        vae_input = image_source.half()  # VAE expects float16
        z_source = pipe.vae.encode(vae_input).latent_dist.mean
        z_source = z_source.float() * pipe.vae.config.scaling_factor  # Convert to float32 BEFORE scaling

        print(f"VAE encoding: input dtype={vae_input.dtype}, output dtype={z_source.dtype}, "
              f"z_source range=[{z_source.min().item():.4f}, {z_source.max().item():.4f}]")

        # Get text embeddings (SDXL needs both regular and pooled)
        embedding_text, pooled_text = get_text_embeddings_sdxl(device, pipe, text_target)
        embedding_null, pooled_null = get_text_embeddings_sdxl(device, pipe, "")

        # Stack for classifier-free guidance: [unconditional, conditional]
        # Keep embeddings as float32 for consistency
        embedding_target = torch.cat([embedding_null, embedding_text], dim=0).float()
        pooled_target = torch.cat([pooled_null, pooled_text], dim=0).float()

    # z_target is float32 for gradient computation
    z_target = z_source.clone()  # Already float32
    z_target.requires_grad = True
    # SDXL has 128x128 latent space (vs 64x64 for SD 1.5), needs higher LR for visible changes
    optimizer = SGD(params=[z_target], lr=0.5)

    print(f"z_target dtype: {z_target.dtype}, device: {z_target.device}")
    print(f"embedding_target dtype: {embedding_target.dtype}, pooled_target dtype: {pooled_target.dtype}")

    # Prepare time ids for SDXL - use float32 to match other tensors
    add_time_ids = torch.tensor([[SDXL_RESOLUTION, SDXL_RESOLUTION, 0, 0, SDXL_RESOLUTION, SDXL_RESOLUTION]],
                                 device=device, dtype=torch.float32)

    simp_img_seq = []
    z_initial = z_target.detach().clone()  # Store initial latent for comparison

    with tqdm(total=num_iters, desc="SDXL SDS Optimization", unit="iter") as pbar:
        for i in range(num_iters):
            loss, log_loss = sds_loss.get_sds_loss(
                z_target, embedding_target, pooled_target,
                guidance_scale=0, add_time_ids=add_time_ids
            )
            optimizer.zero_grad()
            (2000 * loss).backward()

            # Debug: print gradient info for first few iterations
            if i < 3 or i % 20 == 0:
                grad_norm = z_target.grad.norm().item() if z_target.grad is not None else 0
                z_change = (z_target - z_initial).abs().mean().item()
                print(f"Iter {i}: loss={loss.item():.6f}, log_loss={log_loss.item():.6f}, "
                      f"grad_norm={grad_norm:.6f}, z_change={z_change:.6f}")

            optimizer.step()
            out = decode_sdxl(z_target, pipe)
            simp_img_seq.append(out)
            pbar.update(1)

    # Final diagnostic
    z_total_change = (z_target - z_initial).abs().mean().item()
    z_max_change = (z_target - z_initial).abs().max().item()
    print(f"Optimization complete: total z_change (mean)={z_total_change:.6f}, max_change={z_max_change:.6f}")

    return simp_img_seq


def sds_based_simplification_sdxl(device, image_path: str, simp_img_seq_indexs: List[int],
                                   simp_img_seq_save_path: str,
                                   all_simp_img_seq_save_path: str = "-1",
                                   preserve_aspect_ratio: bool = True):
    """
    SDS-based simplification using SDXL at 1024x1024 resolution.

    Args:
        device: torch device
        image_path: Path to input image
        simp_img_seq_indexs: Indices of simplified images to use [80, 60, 40, 20, 0]
        simp_img_seq_save_path: Path to save selected simplified images
        all_simp_img_seq_save_path: Path to save all simplified images (or "-1" to skip)
        preserve_aspect_ratio: If True, pad to square. If False, stretch.

    Returns:
        List of simplified images as numpy arrays
    """
    # Load image
    if preserve_aspect_ratio:
        image, padding_info = load_image_sdxl(image_path, SDXL_RESOLUTION)
    else:
        image, padding_info = load_image_sdxl_stretch(image_path, SDXL_RESOLUTION)

    prompt = " "  # Empty prompt for simplification

    # Load SDXL pipeline with fp16-fixed VAE to avoid NaN issues
    # The standard SDXL VAE has numerical instability in fp16, so we use a fine-tuned version
    from diffusers import AutoencoderKL

    print(f"Loading fp16-fixed VAE from: {SDXL_VAE_FP16_FIX}")
    vae = AutoencoderKL.from_pretrained(
        SDXL_VAE_FP16_FIX,
        torch_dtype=torch.float16
    )

    print(f"Loading SDXL from: {SDXL_MODEL_ID}")
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        SDXL_MODEL_ID,
        vae=vae,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16"
    ).to(device)

    print(f"SDXL loaded on {device} with fp16-fixed VAE")

    num_iters = simp_img_seq_indexs[0]
    print(f"Running {num_iters} SDS iterations at {SDXL_RESOLUTION}x{SDXL_RESOLUTION}...")
    all_simp_img_seq = image_optimization_sdxl(device, pipeline, image, prompt, num_iters)

    # Save images
    simp_img_seq = [image]
    img_pil = Image.fromarray(image)
    img_pil.save(f"{simp_img_seq_save_path}/0.png")

    if all_simp_img_seq_save_path != "-1":
        img_pil.save(f"{all_simp_img_seq_save_path}/0.png")

    for i, simp_img in enumerate(all_simp_img_seq):
        if i + 1 in simp_img_seq_indexs:
            simp_img_seq.append(simp_img)
            simp_img_pil = Image.fromarray(simp_img)
            simp_img_pil.save(f"{simp_img_seq_save_path}/{i+1}.png")

        if all_simp_img_seq_save_path != "-1":
            simp_img_pil = Image.fromarray(simp_img)
            simp_img_pil.save(f"{all_simp_img_seq_save_path}/{i+1}.png")

    # Clean up to free VRAM
    del pipeline
    torch.cuda.empty_cache()

    return simp_img_seq


# Backward compatible function name
def sds_based_simplification(device, image: str, simp_img_seq_indexs: List[int],
                             simp_img_seq_save_path: str,
                             all_simp_img_seq_save_path: str = "-1"):
    """Wrapper for backward compatibility."""
    return sds_based_simplification_sdxl(
        device, image, simp_img_seq_indexs,
        simp_img_seq_save_path, all_simp_img_seq_save_path,
        preserve_aspect_ratio=False
    )
