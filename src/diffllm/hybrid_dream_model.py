"""
Hybrid Dream Model: Qwen3-VL Vision + Dream-7B Diffusion Text Generation

This module combines:
- Qwen3-VL's complete vision system (frozen)
- Dream-7B's diffusion-based text generation (frozen)
- Vision features condition the diffusion process
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
from transformers import AutoModel


class HybridDreamModel(nn.Module):
    """
    Vision-Language Model with Diffusion-based Text Generation.

    Architecture:
        Image/Video → [Qwen Vision Encoder] → Vision Embeddings
                                                      ↓
        Text Prompt → [Tokenizer] → Text Embeddings  ↓
                                                      ↓
                            [Concatenate: vision | text]
                                                      ↓
                            [Dream Transformer (28 layers)]
                                                      ↓
                                Logits (full sequence)
                                                      ↓
                            [Extract text logits only]
                                                      ↓
                            [Diffusion sampling/denoising]
    """

    def __init__(
        self,
        dream_model_name: str = "Dream-org/Dream-v0-Instruct-7B",
        qwen_model_name: str = "Qwen/Qwen2-VL-7B-Instruct",
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()

        self.device = device
        self.dtype = dtype

        print("=" * 60)
        print("Initializing Hybrid Dream Model")
        print("=" * 60)

        # Load Dream-7B model
        print(f"\n[1/2] Loading Dream-7B from {dream_model_name}...")
        self.dream_model = AutoModel.from_pretrained(
            dream_model_name,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
        self.dream_model = self.dream_model.to(device).eval()

        # Freeze Dream model
        for param in self.dream_model.parameters():
            param.requires_grad = False

        print(f"✓ Dream-7B loaded successfully!")
        print(f"  Hidden size: {self.dream_model.config.hidden_size}")
        print(f"  Num layers: {self.dream_model.config.num_hidden_layers}")
        print(f"  Vocab size: {self.dream_model.config.vocab_size}")

        # Load Qwen3-VL vision system
        print(f"\n[2/2] Loading Qwen3-VL vision system...")
        from .qwen_vision_system import Qwen3VLVisionSystem

        self.vision_system = Qwen3VLVisionSystem(
            model_name=qwen_model_name,
            dream_hidden_size=self.dream_model.config.hidden_size,
            device=device,
            dtype=dtype,
        )

        print("\n" + "=" * 60)
        print("Hybrid Dream Model initialized successfully!")
        print("=" * 60)

    def encode_vision(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        video_grid_thw: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode visual inputs through Qwen's vision system.

        Returns:
            vision_embeds: [B, num_vision_tokens, hidden_size]
        """
        return self.vision_system(
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
        )

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        vision_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass through hybrid model.

        Two modes:
        1. With vision: Concatenate vision embeddings before text
        2. Without vision: Standard Dream forward pass

        Args:
            input_ids: Text token IDs [B, L]
            inputs_embeds: Pre-computed text embeddings [B, L, H]
            attention_mask: Attention mask [B, L]
            vision_embeds: Vision embeddings [B, V, H] (optional)

        Returns:
            logits: [B, total_length, vocab_size]
        """

        # Get text embeddings if not provided
        if inputs_embeds is None:
            inputs_embeds = self.dream_model.model.embed_tokens(input_ids)

        # Concatenate vision embeddings if provided
        if vision_embeds is not None:
            # inputs_embeds: [B, L, H]
            # vision_embeds: [B, V, H]
            # combined: [B, V+L, H]
            inputs_embeds = torch.cat([vision_embeds, inputs_embeds], dim=1)

            # Extend a 1D/2D attention mask to cover prepended vision tokens.
            # Skip if the mask is already 4D (pre-computed by the generation utility
            # to include vision positions) or if it's None (full attention).
            if attention_mask is not None and attention_mask.dim() <= 2:
                batch_size = vision_embeds.shape[0]
                num_vision_tokens = vision_embeds.shape[1]

                vision_mask = torch.ones(
                    (batch_size, num_vision_tokens),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                attention_mask = torch.cat([vision_mask, attention_mask], dim=1)

        # Forward through Dream transformer
        outputs = self.dream_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **kwargs,
        )

        # Extract logits
        logits = outputs.logits if hasattr(outputs, "logits") else outputs

        return logits

    def get_num_vision_tokens(
        self,
        image_grid_thw: Optional[torch.Tensor] = None,
        video_grid_thw: Optional[torch.Tensor] = None,
    ) -> int:
        """Get number of vision tokens from grid info."""
        return self.vision_system.get_num_vision_tokens(
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
        )

    def get_embedding_layer(self):
        """Get Dream's token embedding layer."""
        return self.dream_model.model.embed_tokens

    @property
    def config(self):
        """Access Dream's config."""
        return self.dream_model.config

    @property
    def model(self):
        """Access Dream's model for compatibility."""
        return self.dream_model.model
