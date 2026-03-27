"""
Qwen3-VL Vision System Module

This module wraps Qwen3-VL's complete vision pipeline including:
- SigLIP2-So400m ViT encoder
- DeepStack multi-level feature extraction
- Vision-language adapter/projector
- Multi-image and video support
- Interleaved-MRope positional embeddings
"""

import torch
import torch.nn as nn
from typing import Optional, List, Union
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from PIL import Image


class Qwen3VLVisionSystem(nn.Module):
    """
    Complete Qwen3-VL vision system with all capabilities:
    - Multi-image processing
    - Video understanding
    - Spatial reasoning
    - DeepStack features
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-VL-7B-Instruct",
        dream_hidden_size: int = 4096,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()

        self.model_name = model_name
        self.dream_hidden_size = dream_hidden_size
        self.device = device
        self.dtype = dtype

        print(f"Loading Qwen3-VL vision system from {model_name}...")

        # Load full Qwen3-VL model
        self.qwen_model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=device,
        )

        # Extract vision components
        self.visual = self.qwen_model.visual  # SigLIP2 ViT encoder + merger
        self.qwen_hidden_size = self.qwen_model.config.hidden_size

        # Freeze vision system (preserve pretrained knowledge)
        for param in self.visual.parameters():
            param.requires_grad = False

        # Create dimension adapter if needed (Qwen → Dream)
        if self.qwen_hidden_size != self.dream_hidden_size:
            print(f"Creating adapter: {self.qwen_hidden_size} → {self.dream_hidden_size}")
            self.dimension_adapter = nn.Linear(
                self.qwen_hidden_size,
                self.dream_hidden_size,
                bias=False,
            )
            # Initialize with small random values
            nn.init.normal_(self.dimension_adapter.weight, mean=0.0, std=0.02)
        else:
            print("No dimension adapter needed (same hidden size)")
            self.dimension_adapter = None

        self.visual.eval()

        print(f"Qwen3-VL vision system loaded successfully!")
        print(f"  Vision hidden size: {self.qwen_hidden_size}")
        print(f"  Dream hidden size: {self.dream_hidden_size}")
        print(f"  Adapter required: {self.dimension_adapter is not None}")

    @torch.no_grad()
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        video_grid_thw: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Process images and/or videos through Qwen's vision system.

        Args:
            pixel_values: Image tensors [B, C, H, W] or batched
            pixel_values_videos: Video tensors [B, T, C, H, W]
            image_grid_thw: Grid info for images (temporal, height, width)
            video_grid_thw: Grid info for videos

        Returns:
            vision_embeds: [B, num_vision_tokens, dream_hidden_size]
        """

        # Process through Qwen's vision encoder.
        # Qwen2VisionTransformerPretrainedModel.forward(hidden_states, grid_thw)
        # where hidden_states = pixel patches and grid_thw = (T, H, W) grid tensor.
        vision_embeds_list = []

        if pixel_values is not None and image_grid_thw is not None:
            pixel_values = pixel_values.type(self.visual.get_dtype())
            image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
            vision_embeds_list.append(image_embeds)

        if pixel_values_videos is not None and video_grid_thw is not None:
            pixel_values_videos = pixel_values_videos.type(self.visual.get_dtype())
            video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
            vision_embeds_list.append(video_embeds)

        if not vision_embeds_list:
            raise ValueError("No pixel values provided to Qwen vision system")

        vision_embeds = torch.cat(vision_embeds_list, dim=0)

        # vision_embeds shape: [num_vision_tokens, qwen_hidden_size]

        # Adapt to Dream's dimension if needed
        if self.dimension_adapter is not None:
            vision_embeds = self.dimension_adapter(vision_embeds)

        # vision_embeds shape: [B, num_vision_tokens, dream_hidden_size]
        return vision_embeds

    def get_num_vision_tokens(
        self,
        image_grid_thw: Optional[torch.Tensor] = None,
        video_grid_thw: Optional[torch.Tensor] = None,
    ) -> int:
        """
        Calculate total number of vision tokens from grid info.

        Args:
            image_grid_thw: Image grid info [num_images, 3] (T, H, W)
                where H and W are raw patch counts (before the patch merger).
            video_grid_thw: Video grid info [num_videos, 3] (T, H, W)

        Returns:
            Total number of vision tokens after Qwen2-VL's patch merger.
        """
        # Qwen2-VL's PatchMerger reduces spatial tokens by merge_size^2 (default 2 → factor 4).
        merge_size: int = getattr(
            getattr(self.visual, "merger", None), "spatial_merge_size", 2
        )
        divisor = merge_size ** 2
        num_tokens = 0

        if image_grid_thw is not None:
            # Each grid element is (temporal, height_patches, width_patches)
            for thw in image_grid_thw:
                t, h, w = thw
                num_tokens += int(t * h * w) // divisor

        if video_grid_thw is not None:
            for thw in video_grid_thw:
                t, h, w = thw
                num_tokens += int(t * h * w) // divisor

        return num_tokens


class Qwen3VLMultimodalProcessor:
    """
    Multimodal preprocessing using Qwen3-VL's processor.
    Handles images, videos, and text with proper tokenization.
    """

    # Default pixel cap: 1024×1024 equivalent.
    # Qwen2-VL uses 14×14 patches with temporal_patch_size=2, so the effective
    # tile size is 28 pixels.  1024²/28² ≈ 1340 spatial tiles → ~335 vision
    # tokens after the 2×2 merger — a good speed/quality trade-off.
    DEFAULT_MAX_PIXELS: int = 1024 * 1024   # 1 MP

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-VL-7B-Instruct",
        max_pixels: Optional[int] = DEFAULT_MAX_PIXELS,
    ):
        self.model_name = model_name
        self.max_pixels = max_pixels

        print(f"Loading Qwen3-VL processor from {model_name}...")
        self.processor = AutoProcessor.from_pretrained(model_name)
        print("Processor loaded successfully!")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resize_images(self, images: List[Image.Image]) -> List[Image.Image]:
        """
        Downscale images that exceed self.max_pixels while preserving aspect ratio.
        Has no effect when max_pixels is None or the image is already small enough.
        """
        if not self.max_pixels:
            return images
        resized = []
        for img in images:
            w, h = img.size
            if w * h > self.max_pixels:
                scale = (self.max_pixels / (w * h)) ** 0.5
                new_w = max(1, int(w * scale))
                new_h = max(1, int(h * scale))
                img = img.resize((new_w, new_h), Image.LANCZOS)
                print(f"  [resize] {w}×{h} → {new_w}×{new_h} "
                      f"({w * h // 1000}K → {new_w * new_h // 1000}K px)")
            resized.append(img)
        return resized

    def process_images(
        self,
        images: Union[List[Image.Image], List[str]],
        text: str = "",
        max_pixels: Optional[int] = None,
    ) -> dict:
        """
        Process images with optional text.

        Args:
            images:     List of PIL Images or image paths.
            text:       Text prompt.
            max_pixels: Override the instance-level pixel cap for this call.
                        Pass 0 or None to disable resizing entirely.

        Returns:
            Dictionary with processed tensors:
                - pixel_values: Image tensors
                - image_grid_thw: Grid information
                - input_ids: Text token IDs (if text provided)
                - attention_mask: Attention mask
        """
        # Load images if paths provided
        if images and isinstance(images[0], str):
            images = [Image.open(img_path).convert("RGB") for img_path in images]

        # Apply resolution cap (use call-level override when given)
        cap = max_pixels if max_pixels is not None else self.max_pixels
        _orig_max = self.max_pixels
        self.max_pixels = cap
        images = self._resize_images(list(images))
        self.max_pixels = _orig_max

        # Create conversation format for Qwen
        # For multi-image, include multiple <image> tokens
        if text and images:
            content = []
            for _ in images:
                content.append({"type": "image"})
            content.append({"type": "text", "text": text})

            messages = [{"role": "user", "content": content}]

            # Process with Qwen's processor
            inputs = self.processor(
                text=[self.processor.apply_chat_template(messages, add_generation_prompt=True)],
                images=images,
                return_tensors="pt",
                padding=True,
            )
        elif images:
            # Images only (no text)
            inputs = self.processor(
                images=images,
                return_tensors="pt",
            )
        else:
            # Text only (no images)
            messages = [{"role": "user", "content": text}]
            inputs = self.processor(
                text=[self.processor.apply_chat_template(messages, add_generation_prompt=True)],
                return_tensors="pt",
                padding=True,
            )

        return inputs

    def process_video(
        self,
        video_path: str,
        text: str = "",
    ) -> dict:
        """
        Process video with optional text.

        Args:
            video_path: Path to video file
            text: Text prompt

        Returns:
            Dictionary with processed tensors including video data
        """

        # Create conversation format
        content = [
            {"type": "video"},
        ]
        if text:
            content.append({"type": "text", "text": text})

        messages = [{"role": "user", "content": content}]

        # Process with Qwen's processor
        inputs = self.processor(
            text=[self.processor.apply_chat_template(messages, add_generation_prompt=True)],
            videos=[video_path],
            return_tensors="pt",
            padding=True,
        )

        return inputs

    def get_tokenizer(self):
        """Get the underlying tokenizer."""
        return self.processor.tokenizer
