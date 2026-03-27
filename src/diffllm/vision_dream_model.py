"""
VisionDreamModel — Unified Vision + Diffusion Language Model

Architecture (injection, not prepend):
    Image  →  [Qwen2-VL Visual Encoder]  →  vision_embeds [N_vis, H]
                                                     │
    Prompt  →  [Tokenizer]  →  input_ids             │
    (includes <|image_pad|> tokens at image slots)   │
              ↓                                       │
    [Dream embed_tokens]  →  inputs_embeds [B,L,H]   │
              ↓  replace image_pad positions with ────┘
    inputs_embeds (vision injected)  [B, L, H]
              ↓
    [Dream Transformer — 28 layers, bidirectional]
              ↓
    logits  →  [Masked Diffusion Sampling]  →  generated text

Why injection instead of prepend:
    Dream was trained with token embeddings at every position.  If we simply
    prepend vision tokens, Dream's RoPE positions and attention patterns are
    misaligned — it has no learned prior for a "vision prefix".
    Injection replaces the placeholder <|image_pad|> embeddings with the
    actual vision features at the *same sequence positions*, preserving the
    conversation structure Dream expects.

Training:
    Both vision encoder and Dream are frozen.
    Only `vision_proj` (optional linear) is trainable.
    For the no-projection baseline (same hidden dim) the model runs
    zero-shot — no training needed.
"""

import gc
from typing import Optional, List

import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
from transformers import (
    AutoModel, AutoTokenizer, AutoProcessor,
    Qwen2VLForConditionalGeneration,
)


# ---------------------------------------------------------------------------
# Numerics safety: prevent NaN in multinomial sampling
# ---------------------------------------------------------------------------
def _patch_multinomial() -> None:
    def _sanitize(p: torch.Tensor) -> torch.Tensor:
        if not torch.is_floating_point(p):
            return p
        p = torch.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
        s = p.sum(dim=-1, keepdim=True) if p.ndim > 1 else p.sum()
        zero = (s <= 0)
        if (zero if p.ndim == 1 else zero.any()):
            p = torch.where(
                (zero.expand_as(p) if p.ndim > 1 else zero),
                torch.full_like(p, 1.0 / p.shape[-1]),
                p,
            )
            s = p.sum(dim=-1, keepdim=True) if p.ndim > 1 else p.sum()
        return p / s

    _orig = torch.multinomial

    def _safe(p, n, replacement=False, *, generator=None, out=None):
        if p.ndim > 2:
            sh = p.shape
            return _orig(_sanitize(p.contiguous().view(-1, sh[-1])),
                         n, replacement, generator=generator, out=out).view(*sh[:-1], n)
        return _orig(_sanitize(p), n, replacement, generator=generator, out=out)

    if not getattr(torch.multinomial, "_vd_patched", False):
        _safe._vd_patched = True
        torch.multinomial = _safe


_patch_multinomial()


# ---------------------------------------------------------------------------
# Helper: extract only Qwen's visual encoder, drop the LLM layers
# ---------------------------------------------------------------------------
def _load_qwen_visual_encoder(model_name: str, device: str, dtype: torch.dtype) -> nn.Module:
    print(f"  Loading Qwen2-VL from {model_name} (will discard LLM layers)…")
    full = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name, dtype=dtype, device_map=device,
    )
    visual = full.visual          # SigLIP2 ViT + PatchMerger
    visual.eval()
    for p in visual.parameters():
        p.requires_grad = False

    # Drop Qwen's language model to free VRAM (~8 GB)
    del full.model
    del full.lm_head
    del full
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    return visual


# ---------------------------------------------------------------------------
# VisionDreamModel
# ---------------------------------------------------------------------------
class VisionDreamModel(nn.Module):
    """
    Vision-conditioned diffusion language model.

    Parameters
    ----------
    use_proj : bool
        If True and hidden dims differ, insert a trainable linear projection
        between vision encoder output and Dream's embedding space.
        When dims match (both 3584 for 7B models) this has no effect.
    max_pixels : int or None
        Resize images to at most this many pixels before processing.
        Default 1 MP gives ~300–500 vision tokens — good speed/quality.
    """

    DREAM_MODEL = "Dream-org/Dream-v0-Instruct-7B"
    QWEN_MODEL  = "Qwen/Qwen2-VL-7B-Instruct"

    def __init__(
        self,
        dream_model_name: str = DREAM_MODEL,
        qwen_model_name:  str = QWEN_MODEL,
        device: str = "cuda",
        dtype: Optional[torch.dtype] = None,
        use_proj: bool = False,
        max_pixels: int = 1024 * 1024,
    ):
        super().__init__()
        self.device     = device
        self.max_pixels = max_pixels
        self.dtype = dtype or (
            torch.bfloat16 if device == "cuda" else
            torch.float16  if device == "mps"  else
            torch.float32
        )

        print("=" * 60)
        print("VisionDreamModel (injection architecture)")
        print("=" * 60)

        # ── 1. Dream-7B (frozen) ──────────────────────────────────────────
        print(f"\n[1/3] Dream-7B  ← {dream_model_name}")
        self.dream = AutoModel.from_pretrained(
            dream_model_name, dtype=self.dtype, trust_remote_code=True,
        ).to(device).eval()
        for p in self.dream.parameters():
            p.requires_grad = False

        dream_hidden = self.dream.config.hidden_size
        print(f"  hidden={dream_hidden}  layers={self.dream.config.num_hidden_layers}  "
              f"vocab={self.dream.config.vocab_size}")

        # ── 2. Qwen visual encoder (frozen) ──────────────────────────────
        print(f"\n[2/3] Qwen visual encoder  ← {qwen_model_name}")
        self.visual = _load_qwen_visual_encoder(qwen_model_name, device, self.dtype)

        # ── 3. Tokenisers / processor (must come before _probe_vision_dim) ──
        print(f"\n[3/3] Tokenizers & processor")
        self.processor = AutoProcessor.from_pretrained(qwen_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(dream_model_name, trust_remote_code=True)

        # Discover vision output dim via a tiny dummy forward pass
        self._vision_dim = self._probe_vision_dim()
        print(f"  vision output dim: {self._vision_dim}")

        # ── Optional projection (trainable) ──────────────────────────────
        if use_proj and self._vision_dim != dream_hidden:
            print(f"  Adding trainable projection: {self._vision_dim} → {dream_hidden}")
            self.vision_proj: Optional[nn.Linear] = nn.Linear(
                self._vision_dim, dream_hidden, bias=False,
            ).to(device).to(self.dtype)
            nn.init.normal_(self.vision_proj.weight, std=0.02)
        else:
            print(f"  No projection (dims match: {self._vision_dim} == {dream_hidden})")
            self.vision_proj = None

        # <|image_pad|>: the placeholder Qwen puts in input_ids for each
        # vision token.  We replace these embeddings with vision features.
        self.image_pad_id = self.processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        self.mask_token_id = self.tokenizer.mask_token_id or 151666
        print(f"  image_pad_id={self.image_pad_id}  mask_token_id={self.mask_token_id}")
        print(f"  Dream vocab={len(self.tokenizer)}  Qwen vocab={len(self.processor.tokenizer)}")

        print("\n" + "=" * 60)
        print("VisionDreamModel ready!")
        print("=" * 60 + "\n")

    # ------------------------------------------------------------------ #
    # Vision encoding                                                       #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def _probe_vision_dim(self) -> int:
        dummy = Image.new("RGB", (56, 56), (128, 128, 128))
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "x"}]}]
        inp = self.processor(
            text=[self.processor.apply_chat_template(msgs, add_generation_prompt=True)],
            images=[dummy], return_tensors="pt",
        )
        pv   = inp["pixel_values"].to(self.device).type(self.visual.get_dtype())
        grid = inp["image_grid_thw"].to(self.device)
        return self.visual(pv, grid_thw=grid).shape[-1]

    @torch.no_grad()
    def encode_vision(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode images through Qwen's visual encoder.

        Returns:
            vision_embeds: [N_vis, dream_hidden_size]
                where N_vis == number of <|image_pad|> tokens in input_ids
        """
        pv  = pixel_values.type(self.visual.get_dtype())
        out = self.visual(pv, grid_thw=image_grid_thw)   # [N_vis, vision_dim]
        if self.vision_proj is not None:
            out = self.vision_proj(out.to(self.dtype))
        return out.to(self.dtype)

    # ------------------------------------------------------------------ #
    # Core embedding with vision injection                                  #
    # ------------------------------------------------------------------ #

    def embed_with_vision(
        self,
        input_ids: torch.Tensor,
        vision_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute token embeddings and replace <|image_pad|> positions with
        the corresponding vision feature vectors.

        Args:
            input_ids:     [B, L]  — may contain image_pad_id tokens
            vision_embeds: [N_vis, H]  — precomputed from encode_vision()

        Returns:
            inputs_embeds: [B, L, H]
        """
        embeds = self.dream.model.embed_tokens(input_ids)   # [B, L, H]

        if vision_embeds is not None:
            pad_mask = (input_ids == self.image_pad_id)     # [B, L] bool
            n_pads = pad_mask.sum().item()
            if n_pads > 0:
                if n_pads != vision_embeds.shape[0]:
                    raise ValueError(
                        f"Mismatch: {n_pads} image_pad tokens in input_ids "
                        f"but vision_embeds has {vision_embeds.shape[0]} rows. "
                        "Check that pixel_values and input_ids come from the same processor call."
                    )
                embeds = embeds.clone()
                embeds[pad_mask] = vision_embeds            # inject

        return embeds

    # ------------------------------------------------------------------ #
    # Model forward                                                         #
    # ------------------------------------------------------------------ #

    def forward(
        self,
        input_ids: torch.Tensor,
        vision_embeds: Optional[torch.Tensor] = None,
        attention_mask=None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass with vision injection.

        Returns:
            logits: [B, L, vocab_size]
        """
        inputs_embeds = self.embed_with_vision(input_ids, vision_embeds)
        outputs = self.dream(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **kwargs,
        )
        return outputs.logits if hasattr(outputs, "logits") else outputs

    # ------------------------------------------------------------------ #
    # Image preprocessing                                                   #
    # ------------------------------------------------------------------ #

    def _resize(self, images: List[Image.Image]) -> List[Image.Image]:
        if not self.max_pixels:
            return images
        out = []
        for img in images:
            w, h = img.size
            if w * h > self.max_pixels:
                s = (self.max_pixels / (w * h)) ** 0.5
                nw, nh = max(1, int(w * s)), max(1, int(h * s))
                print(f"  [resize] {w}×{h} → {nw}×{nh}")
                img = img.resize((nw, nh), Image.LANCZOS)
            out.append(img)
        return out

    def preprocess(self, text: str, images: Optional[List[Image.Image]] = None) -> dict:
        """
        Build processor inputs from text + optional images.

        Returns a dict with input_ids, attention_mask, and optionally
        pixel_values / image_grid_thw — everything from the *same*
        Qwen processor call so token counts align perfectly.
        """
        if images:
            images = self._resize(list(images))
            content = [{"type": "image"} for _ in images] + [{"type": "text", "text": text}]
            msgs = [{"role": "user", "content": content}]
            return self.processor(
                text=[self.processor.apply_chat_template(msgs, add_generation_prompt=True)],
                images=images,
                return_tensors="pt",
                padding=True,
            )
        msgs = [{"role": "user", "content": text}]
        return self.processor(
            text=[self.processor.apply_chat_template(msgs, add_generation_prompt=True)],
            return_tensors="pt",
            padding=True,
        )

    # ------------------------------------------------------------------ #
    # High-level generate()                                                #
    # ------------------------------------------------------------------ #

    def generate(
        self,
        text: str,
        images: Optional[List[Image.Image]] = None,
        max_new_tokens: int = 128,
        steps: int = 128,
        temperature: float = 0.3,   # greedy (0.0) stalls on vision inputs — 0.2–0.3 works best
        top_p: float = 0.95,
        top_k: Optional[int] = None,
        alg: str = "entropy",
        verbose: bool = True,
    ) -> str:
        inputs = self.preprocess(text, images)
        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                  for k, v in inputs.items()}

        # Pre-compute vision embeddings once
        vision_embeds = None
        if "pixel_values" in inputs:
            vision_embeds = self.encode_vision(
                inputs["pixel_values"], inputs["image_grid_thw"]
            )
            if verbose:
                n = vision_embeds.shape[0]
                pads = (inputs["input_ids"] == self.image_pad_id).sum().item()
                print(f"  vision_embeds: {vision_embeds.shape}  "
                      f"image_pad tokens in prompt: {pads}  (match={n == pads})")

        output = _diffusion_generate(
            model=self,
            tokenizer=self.tokenizer,
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            vision_embeds=vision_embeds,
            max_new_tokens=max_new_tokens,
            steps=steps,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            alg=alg,
            device=self.device,
            verbose=verbose,
        )

        prompt_len = inputs["input_ids"].shape[1]
        generated_ids = output["sequences"][0, prompt_len:]
        return self.tokenizer.decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        ).strip()

    # ------------------------------------------------------------------ #
    # Properties for compatibility with multimodal_generation_utils        #
    # ------------------------------------------------------------------ #

    @property
    def config(self):
        return self.dream.config

    @property
    def model(self):
        return self.dream.model

    def get_embedding_layer(self):
        return self.dream.model.embed_tokens

    def __repr__(self) -> str:
        nd = sum(p.numel() for p in self.dream.parameters()) / 1e9
        nv = sum(p.numel() for p in self.visual.parameters()) / 1e9
        np_ = (sum(p.numel() for p in self.vision_proj.parameters()) / 1e6
               if self.vision_proj else 0)
        return (
            f"VisionDreamModel(\n"
            f"  dream_params={nd:.2f}B (frozen)\n"
            f"  vision_params={nv:.2f}B (frozen)\n"
            f"  proj_params={np_:.1f}M ({'trainable' if self.vision_proj else 'none'})\n"
            f"  device={self.device}  dtype={self.dtype}\n"
            f"  image_pad_id={self.image_pad_id}  mask_token_id={self.mask_token_id}\n"
            f")"
        )


# ---------------------------------------------------------------------------
# Diffusion generation loop (injection-aware)
# ---------------------------------------------------------------------------

def _sample_tokens(logits, temperature=0.0, top_p=None, top_k=None,
                   margin_confidence=False, neg_entropy=False):
    """Sample tokens from logits, return (confidence, token_ids)."""
    import torch.nn.functional as F
    if temperature > 0:
        logits = logits / temperature
    if top_p is not None and top_p < 1.0:
        sorted_l, sorted_i = torch.sort(logits, descending=True)
        cum = torch.cumsum(F.softmax(sorted_l, dim=-1), dim=-1)
        remove = cum > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        mask = torch.zeros_like(logits, dtype=torch.bool)
        mask.scatter_(-1, sorted_i, remove)
        logits = logits.masked_fill(mask, torch.finfo(logits.dtype).min)
    if top_k is not None:
        top_k = min(top_k, logits.size(-1))
        remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits = logits.masked_fill(remove, torch.finfo(logits.dtype).min)

    probs = F.softmax(logits, dim=-1)
    if temperature > 0:
        try:
            x0 = torch.multinomial(probs, num_samples=1).squeeze(-1)
            conf = torch.gather(probs, -1, x0.unsqueeze(-1)).squeeze(-1)
        except Exception:
            conf, x0 = probs.max(dim=-1)
    else:
        conf, x0 = probs.max(dim=-1)

    if margin_confidence:
        sp, _ = torch.sort(probs, dim=-1, descending=True)
        conf = sp[:, 0] - sp[:, 1]
    if neg_entropy:
        log_p = torch.log(probs + 1e-10)
        conf = (probs * log_p).sum(dim=-1)

    return conf, x0


def _diffusion_generate(
    model: "VisionDreamModel",
    tokenizer,
    input_ids: torch.Tensor,
    attention_mask,
    vision_embeds: Optional[torch.Tensor],
    max_new_tokens: int = 128,
    steps: int = 64,
    eps: float = 1e-3,
    temperature: float = 0.0,
    top_p: float = 0.95,
    top_k=None,
    alg: str = "entropy",
    alg_temp=None,
    device: str = "cuda",
    generation_tokens_hook_func=None,
    output_history: bool = False,
    verbose: bool = True,
) -> dict:
    """
    Masked-diffusion generation with vision injection.

    Key difference from the old "prepend" approach:
        - sequence length is always  L + G  (no extra V prepended)
        - vision features are injected at image_pad positions inside forward()
        - logit extraction always starts at index prompt_length
    """
    import torch.nn.functional as F

    batch_size    = input_ids.shape[0]
    prompt_length = input_ids.shape[1]

    # ── initialise generation sequence ───────────────────────────────────
    mask_id = model.mask_token_id
    gen_ids = torch.full((batch_size, max_new_tokens), mask_id,
                         dtype=torch.long, device=device)
    x = torch.cat([input_ids, gen_ids], dim=1)        # [B, L+G]

    # ── attention mask ────────────────────────────────────────────────────
    has_padding = (attention_mask is not None) and torch.any(attention_mask == 0).item()
    if has_padding:
        gen_mask = torch.ones((batch_size, max_new_tokens),
                              dtype=attention_mask.dtype, device=device)
        full_1d = torch.cat([attention_mask, gen_mask], dim=1)
        tok_idx = full_1d.long().cumsum(-1) - 1
        tok_idx.masked_fill_(full_1d == 0, 1)
        dream_attn = torch.logical_and(
            full_1d.unsqueeze(1).unsqueeze(-2),
            full_1d.unsqueeze(1).unsqueeze(-1),
        )
        dream_pos = tok_idx
    else:
        dream_attn = None
        dream_pos  = None

    # ── timestep schedule (mirrors Dream's _sample) ──────────────────────
    timesteps = torch.linspace(1.0, eps, steps + 1, device=device)
    histories = [] if output_history else None

    if verbose:
        print(f"Diffusion: {steps} steps, {max_new_tokens} tokens, alg={alg}")

    # ── denoising loop ────────────────────────────────────────────────────
    for i in range(steps):
        mask_index = (x == mask_id)    # [B, L+G]

        # Forward: embed x, inject vision at image_pad positions, run Dream
        logits = model(
            input_ids=x,
            vision_embeds=vision_embeds,
            attention_mask=dream_attn,
            position_ids=dream_pos,
        )
        # logits shape: [B, L+G, vocab]

        # Dream's one-position logit shift (matches Dream's _sample)
        logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)

        # Extract logits for the generation region only [prompt_length:]
        gen_logits = logits[:, prompt_length:, :]            # [B, G, vocab]

        # Logits for currently masked generation positions
        gen_mask_index = mask_index[:, prompt_length:]       # [B, G]
        mask_logits = gen_logits[gen_mask_index]             # [N_masked, vocab]

        t = timesteps[i]
        s = timesteps[i + 1]

        if alg == "origin":
            p_transfer = (1 - s / t) if i < steps - 1 else 1.0
            x0 = torch.full(
                (mask_logits.shape[0],), mask_id,
                dtype=torch.long, device=device,
            )
            transfer = torch.rand(x0.shape, device=device) < p_transfer
            if transfer.any():
                _, sampled = _sample_tokens(
                    mask_logits[transfer], temperature=temperature,
                    top_p=top_p, top_k=top_k,
                )
                x0[transfer] = sampled
            x[:, prompt_length:][gen_mask_index] = x0

        else:
            margin  = (alg == "topk_margin")
            neg_ent = (alg == "entropy")
            conf, x0 = _sample_tokens(
                mask_logits, temperature=temperature, top_p=top_p, top_k=top_k,
                margin_confidence=margin, neg_entropy=neg_ent,
            )

            num_mask   = gen_mask_index.sum() / batch_size
            n_transfer = int(num_mask * (1 - s / t)) if i < steps - 1 else int(num_mask)

            if n_transfer > 0:
                full_conf = torch.full(
                    (batch_size, max_new_tokens), float("-inf"),
                    dtype=logits.dtype, device=device,
                )
                full_conf[gen_mask_index] = conf

                if alg_temp is None or alg_temp == 0:
                    _, transfer_idx = torch.topk(full_conf, n_transfer)
                else:
                    full_conf = F.softmax(full_conf / alg_temp, dim=-1)
                    transfer_idx = torch.multinomial(full_conf, num_samples=n_transfer)

                x0_full = torch.full(
                    (batch_size, max_new_tokens), mask_id,
                    dtype=torch.long, device=device,
                )
                x0_full[gen_mask_index] = x0

                row_idx = torch.arange(batch_size, device=device).unsqueeze(1).expand_as(transfer_idx)
                x[:, prompt_length:][row_idx, transfer_idx] = x0_full[row_idx, transfer_idx]

        if generation_tokens_hook_func is not None:
            x = generation_tokens_hook_func(i, x, logits)

        if output_history:
            histories.append(x.clone())

        if verbose and ((i + 1) % max(1, steps // 10) == 0 or i == steps - 1):
            remaining = (x[:, prompt_length:] == mask_id).sum().item()
            print(f"  Step {i+1}/{steps}: {remaining} mask tokens remaining")

    if verbose:
        print("Generation complete!")

    result = {"sequences": x}
    if output_history:
        result["history"] = histories
    return result
