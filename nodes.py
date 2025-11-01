import torch
import torch.nn.functional as F
import math
from comfy.utils import ProgressBar
import comfy.model_management as mm

device = mm.get_torch_device()
offload_device = mm.unet_offload_device()

VAE_STRIDE = (4, 8, 8)

class WanVideoVACEEncode4090:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "vae": ("WANVAE",),
            "width": ("INT", {"default": 832, "min": 64, "max": 8096, "step": 8, "tooltip": "Width of the image to encode"}),
            "height": ("INT", {"default": 480, "min": 64, "max": 8096, "step": 8, "tooltip": "Height of the image to encode"}),
            "num_frames": ("INT", {"default": 81, "min": 1, "max": 10000, "step": 4, "tooltip": "Number of frames to encode"}),
            "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}),
            "vace_start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Start percent of the steps to apply VACE"}),
            "vace_end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "End percent of the steps to apply VACE"}),
            },
            "optional": {
                "input_frames": ("IMAGE",),
                "ref_images": ("IMAGE",),
                "input_masks": ("MASK",),
                "prev_vace_embeds": ("WANVIDIMAGE_EMBEDS",),
                "tiled_vae": ("BOOLEAN", {"default": False, "tooltip": "Use optimized tiled VAE encoding for better memory utilization"}),
                "tile_size_multiplier": ("FLOAT", {"default": 2.0, "min": 0.5, "max": 4.0, "step": 0.1, "tooltip": "Multiplier for tile size, larger values use more VRAM but are faster"}),
                "min_tile_size": ("INT", {"default": 256, "min": 64, "max": 1024, "step": 16, "tooltip": "Minimum tile size in pixels"}),
            },
        }

    RETURN_TYPES = ("WANVIDIMAGE_EMBEDS", )
    RETURN_NAMES = ("vace_embeds",)
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "Optimized VACE encoder for 4090-class GPUs with improved tiling performance"

    def process(self, vae, width, height, num_frames, strength, vace_start_percent, vace_end_percent, 
                input_frames=None, ref_images=None, input_masks=None, prev_vace_embeds=None, 
                tiled_vae=False, tile_size_multiplier=2.0, min_tile_size=256):
        
        width = (width // 16) * 16
        height = (height // 16) * 16

        target_shape = (16, (num_frames - 1) // VAE_STRIDE[0] + 1,
                        height // VAE_STRIDE[1],
                        width // VAE_STRIDE[2])
        
        # VACE context encode with optimized tiling
        if input_frames is None:
            input_frames = torch.zeros((1, 3, num_frames, height, width), device=device, dtype=vae.dtype)
        else:
            input_frames = input_frames.clone()[:num_frames, :, :, :3]
            # Use more efficient upscaling
            B, H_orig, W_orig, C = input_frames.shape
            if H_orig != height or W_orig != width:
                input_frames = input_frames.permute(0, 3, 1, 2)  # B, C, H, W
                input_frames = F.interpolate(input_frames, size=(height, width), mode='bilinear', align_corners=False)
                input_frames = input_frames.permute(0, 2, 3, 1)  # B, H, W, C
            input_frames = input_frames.to(vae.dtype).to(device).unsqueeze(0).permute(0, 4, 1, 2, 3) # B, C, T, H, W
            input_frames = input_frames * 2 - 1
        
        if input_masks is None:
            input_masks = torch.ones_like(input_frames, device=device)
        else:
            input_masks = input_masks[:num_frames]
            B, H_orig, W_orig = input_masks.shape
            if H_orig != height or W_orig != width:
                input_masks = input_masks.unsqueeze(1)  # B, 1, H, W
                input_masks = F.interpolate(input_masks, size=(height, width), mode='nearest-exact')
                input_masks = input_masks.squeeze(1)
            input_masks = input_masks.to(vae.dtype).to(device)
            input_masks = input_masks.unsqueeze(-1).unsqueeze(0).permute(0, 4, 1, 2, 3).repeat(1, 3, 1, 1, 1) # B, C, T, H, W

        if ref_images is not None:
            ref_images = ref_images.clone()[..., :3]
            # Optimized ref image processing
            if ref_images.shape[0] > 1:
                ref_images = torch.cat([ref_images[i] for i in range(ref_images.shape[0])], dim=1).unsqueeze(0)
            
            B, H_ref, W_ref, C = ref_images.shape
            if H_ref != height or W_ref != width:
                ref_images = ref_images.permute(0, 3, 1, 2)  # B, C, H, W
                ref_images = F.interpolate(ref_images, size=(height, width), mode='bilinear', align_corners=False)
                ref_images = ref_images.permute(0, 2, 3, 1)  # B, H, W, C
            
            ref_images = ref_images.to(vae.dtype).to(device).unsqueeze(0).permute(0, 4, 1, 2, 3).unsqueeze(0)
            ref_images = ref_images * 2 - 1

        vae = vae.to(device)
        
        # Optimized tiling parameters for 4090-class GPUs
        if tiled_vae:
            # Calculate optimal tile size based on available VRAM and multiplier
            base_tile_size = max(min_tile_size, min(height, width) // 2)
            optimized_tile_size = min(int(base_tile_size * tile_size_multiplier), min(height, width))
            
            # Ensure tile size is divisible by 16 for alignment
            optimized_tile_size = (optimized_tile_size // 16) * 16
            optimized_tile_size = max(optimized_tile_size, min_tile_size)
            
            # Use larger stride for fewer overlaps (faster but potential seams)
            tile_stride = max(optimized_tile_size // 2, optimized_tile_size - 32)
            
            print(f"Using optimized tiling: size={optimized_tile_size}, stride={tile_stride}")
            
            z0 = self.vace_encode_frames_optimized(
                vae, input_frames, ref_images, masks=input_masks, 
                tile_size=(optimized_tile_size, optimized_tile_size),
                tile_stride=(tile_stride, tile_stride)
            )
        else:
            z0 = self.vace_encode_frames(vae, input_frames, ref_images, masks=input_masks, tiled_vae=False)
        
        m0 = self.vace_encode_masks(input_masks, ref_images)
        z = self.vace_latent(z0, m0)
        vae.to(offload_device)

        vace_input = {
            "vace_context": z,
            "vace_scale": strength,
            "has_ref": ref_images is not None,
            "num_frames": num_frames,
            "target_shape": target_shape,
            "vace_start_percent": vace_start_percent,
            "vace_end_percent": vace_end_percent,
            "vace_seq_len": math.ceil((z[0].shape[2] * z[0].shape[3]) / 4 * z[0].shape[1]),
            "additional_vace_inputs": [],
        }

        if prev_vace_embeds is not None:
            if "additional_vace_inputs" in prev_vace_embeds and prev_vace_embeds["additional_vace_inputs"]:
                vace_input["additional_vace_embeds"] = prev_vace_embeds["additional_vace_inputs"].copy()
            vace_input["additional_vace_inputs"].append(prev_vace_embeds)
    
        return (vace_input,)
    
    def vace_encode_frames_optimized(self, vae, frames, ref_images, masks=None, tile_size=(512, 512), tile_stride=(256, 256)):
        """Optimized frame encoding with larger tiles for better performance"""
        if ref_images is None:
            ref_images = [None] * len(frames)
        else:
            assert len(frames) == len(ref_images)

        pbar = ProgressBar(len(frames))
        if masks is None:
            # Use optimized tiling with larger blocks
            latents = vae.encode(frames, device=device, tiled=True, 
                               tile_size=(tile_size[0]//vae.upsampling_factor, tile_size[1]//vae.upsampling_factor),
                               tile_stride=(tile_stride[0]//vae.upsampling_factor, tile_stride[1]//vae.upsampling_factor))
        else:
            # Process larger chunks at once
            inactive = [i * (1 - m) + 0 * m for i, m in zip(frames, masks)]
            reactive = [i * m + 0 * (1 - m) for i, m in zip(frames, masks)]
            del frames
            
            # Batch process for better GPU utilization
            inactive = vae.encode(inactive, device=device, tiled=True,
                                tile_size=(tile_size[0]//vae.upsampling_factor, tile_size[1]//vae.upsampling_factor),
                                tile_stride=(tile_stride[0]//vae.upsampling_factor, tile_stride[1]//vae.upsampling_factor))
            reactive = vae.encode(reactive, device=device, tiled=True,
                                tile_size=(tile_size[0]//vae.upsampling_factor, tile_size[1]//vae.upsampling_factor),
                                tile_stride=(tile_stride[0]//vae.upsampling_factor, tile_stride[1]//vae.upsampling_factor))
            latents = [torch.cat((u, c), dim=0) for u, c in zip(inactive, reactive)]
            del inactive, reactive
        
        # Process reference images with same optimized tiling
        cat_latents = []
        for latent, refs in zip(latents, ref_images):
            if refs is not None:
                if masks is None:
                    ref_latent = vae.encode(refs, device=device, tiled=True,
                                          tile_size=(tile_size[0]//vae.upsampling_factor, tile_size[1]//vae.upsampling_factor),
                                          tile_stride=(tile_stride[0]//vae.upsampling_factor, tile_stride[1]//vae.upsampling_factor))
                else:
                    ref_latent = vae.encode(refs, device=device, tiled=True,
                                          tile_size=(tile_size[0]//vae.upsampling_factor, tile_size[1]//vae.upsampling_factor),
                                          tile_stride=(tile_stride[0]//vae.upsampling_factor, tile_stride[1]//vae.upsampling_factor))
                    ref_latent = [torch.cat((u, torch.zeros_like(u)), dim=0) for u in ref_latent]
                assert all([x.shape[1] == 1 for x in ref_latent])
                latent = torch.cat([*ref_latent, latent], dim=1)
            cat_latents.append(latent)
            pbar.update(1)
        return cat_latents

    def vace_encode_frames(self, vae, frames, ref_images, masks=None, tiled_vae=False):
        """Original frame encoding method for compatibility"""
        if ref_images is None:
            ref_images = [None] * len(frames)
        else:
            assert len(frames) == len(ref_images)

        pbar = ProgressBar(len(frames))
        if masks is None:
            latents = vae.encode(frames, device=device, tiled=tiled_vae)
        else:
            inactive = [i * (1 - m) + 0 * m for i, m in zip(frames, masks)]
            reactive = [i * m + 0 * (1 - m) for i, m in zip(frames, masks)]
            del frames
            inactive = vae.encode(inactive, device=device, tiled=tiled_vae)
            reactive = vae.encode(reactive, device=device, tiled=tiled_vae)
            latents = [torch.cat((u, c), dim=0) for u, c in zip(inactive, reactive)]
            del inactive, reactive
        
        cat_latents = []
        for latent, refs in zip(latents, ref_images):
            if refs is not None:
                if masks is None:
                    ref_latent = vae.encode(refs, device=device, tiled=tiled_vae)
                else:
                    ref_latent = vae.encode(refs, device=device, tiled=tiled_vae)
                    ref_latent = [torch.cat((u, torch.zeros_like(u)), dim=0) for u in ref_latent]
                assert all([x.shape[1] == 1 for x in ref_latent])
                latent = torch.cat([*ref_latent, latent], dim=1)
            cat_latents.append(latent)
            pbar.update(1)
        return cat_latents

    def vace_encode_masks(self, masks, ref_images=None):
        """Optimized mask encoding"""
        if ref_images is None:
            ref_images = [None] * len(masks)
        else:
            assert len(masks) == len(ref_images)

        result_masks = []
        pbar = ProgressBar(len(masks))
        for mask, refs in zip(masks, ref_images):
            _c, depth, height, width = mask.shape
            new_depth = int((depth + 3) // VAE_STRIDE[0])
            height = 2 * (int(height) // (VAE_STRIDE[1] * 2))
            width = 2 * (int(width) // (VAE_STRIDE[2] * 2))

            # Optimized reshape and interpolation
            mask = mask[0, :, :, :]
            mask = mask.view(
                depth, height, VAE_STRIDE[1], width, VAE_STRIDE[1]
            )
            mask = mask.permute(2, 4, 0, 1, 3)
            mask = mask.reshape(
                VAE_STRIDE[1] * VAE_STRIDE[2], depth, height, width
            )

            # Use more efficient interpolation
            mask = F.interpolate(mask.unsqueeze(0), size=(new_depth, height, width), 
                               mode='nearest-exact').squeeze(0)

            if refs is not None:
                length = len(refs)
                mask_pad = torch.zeros_like(mask[:, :length, :, :])
                mask = torch.cat((mask_pad, mask), dim=1)
            result_masks.append(mask)
            pbar.update(1)
        return result_masks

    def vace_latent(self, z, m):
        return [torch.cat([zz, mm], dim=0) for zz, mm in zip(z, m)]

NODE_CLASS_MAPPINGS = {
    "WanVideoVACEEncode4090": WanVideoVACEEncode4090,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WanVideoVACEEncode4090": "WanVideo VACE Encode 4090",
}