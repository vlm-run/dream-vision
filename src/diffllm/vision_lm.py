"""
VisionLM — Unified vision-conditioned LM with three backends.

┌──────────┬────────────────────────────────┬────────┬────────┬──────────────┐
│ Backend  │ Model                          │ H-size │ Gen    │ Vision inject│
├──────────┼────────────────────────────────┼────────┼────────┼──────────────┤
│ dream    │ Dream-v0-Instruct-7B           │  3584  │ mdiff  │ image_pad    │
│ nbdiff   │ NBDiff-7B-Instruct (PanGu)    │  4096  │ causal │ prepend      │
│ llada    │ LLaDA2.1-mini (MoE, diffusion)│  2048  │ mdiff  │ prepend      │
└──────────┴────────────────────────────────┴────────┴────────┴──────────────┘

Vision encoder: Qwen2-VL-7B ViT (frozen, 3584-dim output)
Projection    : nn.Linear(3584 → lm_hidden) when dims differ (frozen zero-shot)
"""

import gc
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List
from PIL import Image
from transformers import (
    AutoModel, AutoModelForCausalLM, AutoTokenizer, AutoProcessor,
    Qwen2VLForConditionalGeneration,
)
import transformers

from .vision_dream_model import _patch_multinomial, _diffusion_generate

_patch_multinomial()

# ── Compatibility shim: LossKwargs missing from this transformers build ──────
if not hasattr(transformers.utils, "LossKwargs"):
    from typing import TypedDict

    class _LossKwargs(TypedDict, total=False):
        num_items_in_batch: Optional[int]

    transformers.utils.LossKwargs = _LossKwargs
    import sys
    _m = sys.modules.get("transformers.utils")
    if _m is not None:
        _m.LossKwargs = _LossKwargs


# ---------------------------------------------------------------------------
# Shared: load only Qwen's visual encoder
# ---------------------------------------------------------------------------
def _load_qwen_visual(qwen_id: str, device: str, dtype: torch.dtype) -> nn.Module:
    print(f"  Loading Qwen ViT from {qwen_id} …")
    full = Qwen2VLForConditionalGeneration.from_pretrained(
        qwen_id, dtype=dtype, device_map=device,
    )
    vis = full.visual.eval()
    for p in vis.parameters():
        p.requires_grad = False
    del full.model, full.lm_head, full
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return vis


# ---------------------------------------------------------------------------
# VisionLM
# ---------------------------------------------------------------------------
class VisionLM(nn.Module):

    DREAM_ID  = "Dream-org/Dream-v0-Instruct-7B"
    NBDIFF_ID = "yuchuantian/NBDiff-7B-Instruct"
    LLADA_ID  = "inclusionAI/LLaDA2.1-mini"
    QWEN_ID   = "Qwen/Qwen2-VL-7B-Instruct"

    def __init__(
        self,
        backend: str = "dream",
        qwen_model_name: str = QWEN_ID,
        device: str = "cuda",
        dtype: Optional[torch.dtype] = None,
        max_pixels: int = 1024 * 1024,
    ):
        super().__init__()
        assert backend in ("dream", "nbdiff", "llada"), \
            f"backend must be 'dream', 'nbdiff', or 'llada'"
        self.backend    = backend
        self.device     = device
        self.max_pixels = max_pixels
        self.dtype = dtype or (
            torch.bfloat16 if device == "cuda" else
            torch.float16  if device == "mps"  else
            torch.float32
        )

        print("=" * 60)
        print(f"VisionLM  backend={backend.upper()}")
        print("=" * 60)

        # ── 1. Qwen ViT ───────────────────────────────────────────────────
        print(f"\n[1/3] Qwen ViT ← {qwen_model_name}")
        self.qwen_processor = AutoProcessor.from_pretrained(qwen_model_name)
        self.visual = _load_qwen_visual(qwen_model_name, device, self.dtype)
        self._vision_dim = self._probe_vision_dim()
        print(f"  vision_dim={self._vision_dim}")

        # ── 2. LLM backend ────────────────────────────────────────────────
        lm_id = {
            "dream":  self.DREAM_ID,
            "nbdiff": self.NBDIFF_ID,
            "llada":  self.LLADA_ID,
        }[backend]
        print(f"\n[2/3] LLM ← {lm_id}")

        if backend == "dream":
            self.lm = AutoModel.from_pretrained(
                lm_id, dtype=self.dtype, trust_remote_code=True,
            ).to(device).eval()
            self.tokenizer = AutoTokenizer.from_pretrained(lm_id, trust_remote_code=True)
            self.mask_token_id = self.tokenizer.mask_token_id or 151666
            self.image_pad_id  = self.qwen_processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
            self._gen_mode = "diffusion"

        elif backend == "nbdiff":
            self.lm = AutoModelForCausalLM.from_pretrained(
                lm_id, dtype=self.dtype, trust_remote_code=True,
                device_map=device,
            ).eval()
            self.tokenizer  = AutoTokenizer.from_pretrained(lm_id, trust_remote_code=True)
            self.eos_id     = self.tokenizer.eos_token_id   # 45892
            self._gen_mode  = "causal"

        else:  # llada — need full LM model (with lm_head) to get logits
            self.lm = AutoModelForCausalLM.from_pretrained(
                lm_id, dtype=self.dtype, trust_remote_code=True,
                device_map=device,
            ).eval()
            self.tokenizer = AutoTokenizer.from_pretrained(lm_id, trust_remote_code=True)
            # Discover mask token: try common names then fall back to pad_token
            self.mask_token_id = self._find_mask_token()
            self._gen_mode = "diffusion"

        for p in self.lm.parameters():
            p.requires_grad = False

        self.lm_hidden = self.lm.config.hidden_size
        print(f"  hidden={self.lm_hidden}  vocab={self.lm.config.vocab_size}  gen={self._gen_mode}")

        # ── 3. Projection ─────────────────────────────────────────────────
        print(f"\n[3/3] Projection {self._vision_dim} → {self.lm_hidden}")
        if self._vision_dim != self.lm_hidden:
            self.proj = nn.Linear(self._vision_dim, self.lm_hidden,
                                  bias=False).to(device).to(self.dtype)
            nn.init.normal_(self.proj.weight, std=0.02)
            for p in self.proj.parameters():
                p.requires_grad = False
            print(f"  Created linear projection (frozen zero-shot baseline)")
        else:
            self.proj = None
            print(f"  No projection needed")

        print("\n" + "=" * 60)
        print(f"VisionLM [{backend.upper()}] ready!")
        print("=" * 60 + "\n")

    # ------------------------------------------------------------------ #
    # Setup helpers                                                         #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def _probe_vision_dim(self) -> int:
        dummy = Image.new("RGB", (56, 56), (128, 128, 128))
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "x"}]}]
        inp = self.qwen_processor(
            text=[self.qwen_processor.apply_chat_template(msgs, add_generation_prompt=True)],
            images=[dummy], return_tensors="pt",
        )
        pv   = inp["pixel_values"].to(self.device).type(self.visual.get_dtype())
        grid = inp["image_grid_thw"].to(self.device)
        return self.visual(pv, grid_thw=grid).shape[-1]

    def _find_mask_token(self) -> int:
        """Discover the mask token ID for LLaDA-style models."""
        candidates = ["[MASK]", "<mask>", "<|mask|>", "<MASK>"]
        for tok in candidates:
            tid = self.tokenizer.convert_tokens_to_ids(tok)
            if tid is not None and tid != self.tokenizer.unk_token_id:
                print(f"  mask_token={tok!r}  id={tid}")
                return tid
        # Fall back: use the pad_token_id (common for LLaDA variants)
        pid = self.tokenizer.pad_token_id or self.lm.config.pad_token_id or 156892
        print(f"  mask_token not found; using pad_token_id={pid} as mask")
        return pid

    # ------------------------------------------------------------------ #
    # Vision encoding                                                       #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def encode_vision(self, pixel_values: torch.Tensor,
                      image_grid_thw: torch.Tensor) -> torch.Tensor:
        """[N_vis, lm_hidden_size]"""
        pv  = pixel_values.type(self.visual.get_dtype())
        out = self.visual(pv, grid_thw=image_grid_thw)
        if self.proj is not None:
            out = self.proj(out.to(self.dtype))
        return out.to(self.dtype)

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

    def _preprocess_dream(self, text: str, images) -> dict:
        """Qwen processor — input_ids includes <|image_pad|> tokens."""
        if images:
            images = self._resize(list(images))
            content = [{"type": "image"} for _ in images] + [{"type": "text", "text": text}]
            msgs = [{"role": "user", "content": content}]
            return self.qwen_processor(
                text=[self.qwen_processor.apply_chat_template(msgs, add_generation_prompt=True)],
                images=images, return_tensors="pt", padding=True,
            )
        msgs = [{"role": "user", "content": text}]
        return self.qwen_processor(
            text=[self.qwen_processor.apply_chat_template(msgs, add_generation_prompt=True)],
            return_tensors="pt", padding=True,
        )

    def _preprocess_prepend(self, text: str, images) -> dict:
        """
        Use the backend's own tokenizer for text.
        Images are preprocessed via Qwen only for pixel extraction.
        Vision tokens will be prepended in the embedding step.
        """
        msgs = [{"role": "user", "content": text}]
        prompt = self.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
        )
        tok = self.tokenizer(prompt, return_tensors="pt")

        if images:
            images = self._resize(list(images))
            q_content = [{"type": "image"} for _ in images] + [{"type": "text", "text": "x"}]
            q_msgs = [{"role": "user", "content": q_content}]
            qinp = self.qwen_processor(
                text=[self.qwen_processor.apply_chat_template(q_msgs, add_generation_prompt=True)],
                images=images, return_tensors="pt",
            )
            tok["pixel_values"]   = qinp["pixel_values"]
            tok["image_grid_thw"] = qinp["image_grid_thw"]

        return tok

    # ------------------------------------------------------------------ #
    # Embed + inject helpers                                                #
    # ------------------------------------------------------------------ #

    def _get_embed_fn(self):
        """Return (embed_tokens, lm_model) for the current backend."""
        if self.backend == "dream":
            return self.lm.model.embed_tokens, self.lm
        elif self.backend == "nbdiff":
            return self.lm.model.embed_tokens, self.lm
        else:  # llada — model IS the backbone (no .model wrapper)
            return self.lm.get_input_embeddings(), self.lm

    def _embed_inject(self, input_ids: torch.Tensor,
                      vision_embeds: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Dream  : replace <|image_pad|> positions with vision features.
        Others : prepend vision tokens before text embeddings.
        """
        embed_fn, _ = self._get_embed_fn()
        text_embeds = embed_fn(input_ids)   # [B, L, H]

        if vision_embeds is None:
            return text_embeds

        if self.backend == "dream":
            mask = (input_ids == self.image_pad_id)
            n_pads = mask.sum().item()
            if n_pads != vision_embeds.shape[0]:
                raise ValueError(
                    f"image_pad count ({n_pads}) != vision_embeds rows ({vision_embeds.shape[0]})"
                )
            embeds = text_embeds.clone()
            embeds[mask] = vision_embeds
            return embeds
        else:
            # prepend: [vis_tokens | text_tokens]
            ve = vision_embeds.unsqueeze(0)             # [1, N, H]
            return torch.cat([ve, text_embeds], dim=1)  # [B, N+L, H]

    # ------------------------------------------------------------------ #
    # Dream forward (used inside diffusion loop)                           #
    # ------------------------------------------------------------------ #

    def _forward_dream(self, input_ids, vision_embeds=None, attention_mask=None, **kw):
        embeds = self._embed_inject(input_ids, vision_embeds)
        out = self.lm(inputs_embeds=embeds, attention_mask=attention_mask, **kw)
        return out.logits if hasattr(out, "logits") else out

    # ------------------------------------------------------------------ #
    # generate()                                                            #
    # ------------------------------------------------------------------ #

    def generate(
        self,
        text: str,
        images: Optional[List[Image.Image]] = None,
        max_new_tokens: int = 128,
        steps: int = 128,
        temperature: float = 0.3,
        top_p: float = 0.95,
        top_k: Optional[int] = None,
        alg: str = "entropy",
        verbose: bool = True,
    ) -> str:
        if self.backend == "dream":
            return self._gen_dream(text, images, max_new_tokens,
                                   steps, temperature, top_p, top_k, alg, verbose)
        elif self.backend == "nbdiff":
            return self._gen_causal(text, images, max_new_tokens,
                                    temperature, top_p, verbose)
        else:
            return self._gen_llada(text, images, max_new_tokens,
                                   steps, temperature, top_p, top_k, alg, verbose)

    # ── Dream ─────────────────────────────────────────────────────────────

    def _gen_dream(self, text, images, max_new_tokens, steps,
                   temperature, top_p, top_k, alg, verbose):
        inputs = self._preprocess_dream(text, images)
        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                  for k, v in inputs.items()}

        vision_embeds = None
        if "pixel_values" in inputs:
            vision_embeds = self.encode_vision(
                inputs["pixel_values"], inputs["image_grid_thw"]
            )
            if verbose:
                n = vision_embeds.shape[0]
                pads = (inputs["input_ids"] == self.image_pad_id).sum().item()
                print(f"  [dream] vis={vision_embeds.shape}  pads={pads}  match={n==pads}")

        # Thin wrapper so _diffusion_generate can call model(input_ids=…)
        _self = self

        class _DreamProxy:
            mask_token_id = _self.mask_token_id

            def __call__(s, input_ids, vision_embeds=None, attention_mask=None, **kw):
                return _self._forward_dream(input_ids, vision_embeds, attention_mask, **kw)

            def get_embedding_layer(s):
                return _self.lm.model.embed_tokens

            @property
            def config(s):
                return _self.lm.config

            @property
            def model(s):
                return _self.lm.model

        output = _diffusion_generate(
            model=_DreamProxy(),
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
        L = inputs["input_ids"].shape[1]
        return self.tokenizer.decode(
            output["sequences"][0, L:], skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        ).strip()

    # ── NBDiff (causal, manual autoregressive loop) ────────────────────────

    def _gen_causal(self, text, images, max_new_tokens, temperature, top_p, verbose):
        inputs = self._preprocess_prepend(text, images)
        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                  for k, v in inputs.items()}

        input_ids = inputs["input_ids"]
        attn_mask = inputs.get("attention_mask")

        vision_embeds = None
        n_vis = 0
        if "pixel_values" in inputs:
            vision_embeds = self.encode_vision(
                inputs["pixel_values"], inputs["image_grid_thw"]
            )
            n_vis = vision_embeds.shape[0]
            if verbose:
                print(f"  [nbdiff] vis={vision_embeds.shape}  prepend")

        # Build starting inputs_embeds (vision prepended)
        inputs_embeds = self._embed_inject(input_ids, vision_embeds)  # [1, N+L, H]
        if attn_mask is not None and n_vis > 0:
            vis_mask = torch.ones((1, n_vis), dtype=attn_mask.dtype, device=self.device)
            attn_mask = torch.cat([vis_mask, attn_mask], dim=1)

        if verbose:
            print(f"  [nbdiff] embeds shape: {inputs_embeds.shape}")

        # NBDiff's SDPA requires bool mask (tokenizer returns int64)
        if attn_mask is not None:
            attn_mask = attn_mask.bool()

        generated = self._autoregressive_loop(
            inputs_embeds, attn_mask, max_new_tokens, temperature, top_p,
        )
        return self.tokenizer.decode(generated, skip_special_tokens=True,
                                     clean_up_tokenization_spaces=True).strip()

    def _autoregressive_loop(self, inputs_embeds, attention_mask,
                              max_new_tokens, temperature, top_p):
        """
        KV-cache autoregressive generation without model.generate().
        Works for any causal LM loaded via AutoModelForCausalLM.
        """
        generated = []
        past_kv   = None
        cur_embeds = inputs_embeds
        cur_mask   = attention_mask

        for _ in range(max_new_tokens):
            with torch.no_grad():
                out = self.lm(
                    inputs_embeds=cur_embeds,
                    attention_mask=cur_mask,
                    use_cache=True,
                    past_key_values=past_kv,
                )
            logits  = out.logits[:, -1, :]   # [1, vocab]
            past_kv = getattr(out, "past_key_values", None)

            # Nucleus sampling / greedy
            if temperature > 0:
                probs = F.softmax(logits / temperature, dim=-1)
                if top_p < 1.0:
                    s_probs, s_idx = torch.sort(probs, descending=True)
                    cum = s_probs.cumsum(-1)
                    remove = cum > top_p
                    remove[:, 1:] = remove[:, :-1].clone()
                    remove[:, 0]  = False
                    s_probs[remove] = 0.0
                    s_probs /= s_probs.sum(-1, keepdim=True).clamp_min(1e-9)
                    next_tok = torch.multinomial(s_probs, 1)
                    next_tok = s_idx.gather(-1, next_tok)
                else:
                    next_tok = torch.multinomial(probs, 1)
            else:
                next_tok = logits.argmax(-1, keepdim=True)

            tok_id = next_tok.item()
            if tok_id == getattr(self, "eos_id", -1):
                break
            generated.append(tok_id)

            # Embed next token, carry on with KV cache
            embed_fn, _ = self._get_embed_fn()
            cur_embeds = embed_fn(next_tok)             # [1, 1, H]
            if cur_mask is not None:
                cur_mask = torch.cat(
                    [cur_mask, torch.ones((1, 1), dtype=cur_mask.dtype, device=self.device)],
                    dim=1,
                )

        return generated

    # ── LLaDA (masked diffusion, prepend) ─────────────────────────────────

    def _gen_llada(self, text, images, max_new_tokens, steps,
                   temperature, top_p, top_k, alg, verbose):
        inputs = self._preprocess_prepend(text, images)
        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                  for k, v in inputs.items()}

        input_ids = inputs["input_ids"]
        attn_mask = inputs.get("attention_mask")

        vision_embeds = None
        n_vis = 0
        if "pixel_values" in inputs:
            vision_embeds = self.encode_vision(
                inputs["pixel_values"], inputs["image_grid_thw"]
            )
            n_vis = vision_embeds.shape[0]
            if verbose:
                print(f"  [llada] vis={vision_embeds.shape}  prepend")

        # prompt_ids = [vision_prefix (virtual) | text tokens]
        # We handle the vision prefix by adding it to inputs_embeds in forward,
        # but the diffusion loop works on input_ids — we extend them with
        # placeholder tokens (BOS) for the vision positions so lengths align.
        bos_id = self.tokenizer.bos_token_id or 1
        vis_placeholder = torch.full(
            (1, n_vis), bos_id, dtype=torch.long, device=self.device
        )
        full_prompt_ids = torch.cat([vis_placeholder, input_ids], dim=1) if n_vis > 0 else input_ids

        if attn_mask is not None and n_vis > 0:
            vis_mask = torch.ones((1, n_vis), dtype=attn_mask.dtype, device=self.device)
            attn_mask = torch.cat([vis_mask, attn_mask], dim=1)

        _self = self

        class _LLaDAProxy:
            mask_token_id = _self.mask_token_id

            def __call__(s, input_ids, vision_embeds=None, attention_mask=None, **kw):
                # Re-embed: replace the leading n_vis placeholder rows with vision features
                embed_fn = _self.lm.get_input_embeddings()
                embeds = embed_fn(input_ids)                # [B, L, H]
                if vision_embeds is not None and n_vis > 0:
                    embeds = embeds.clone()
                    embeds[:, :n_vis, :] = vision_embeds.unsqueeze(0)
                # LLaDA requires a 4D float block attention mask (B, 1, L, L).
                # Additive convention: 0.0 = attend, large neg = mask.
                # All-zeros → full bidirectional attention (correct for diffusion).
                B, L = embeds.shape[0], embeds.shape[1]
                attention_mask = torch.zeros(
                    (B, 1, L, L), dtype=embeds.dtype, device=embeds.device
                )
                out = _self.lm(inputs_embeds=embeds,
                               attention_mask=attention_mask, **kw)
                return out.logits if hasattr(out, "logits") else out

            def get_embedding_layer(s):
                return _self.lm.get_input_embeddings()

            @property
            def config(s):
                return _self.lm.config

            @property
            def model(s):
                return _self.lm.model if hasattr(_self.lm, "model") else _self.lm

        if verbose:
            print(f"  [llada] full_prompt shape: {full_prompt_ids.shape}")

        output = _diffusion_generate(
            model=_LLaDAProxy(),
            tokenizer=self.tokenizer,
            input_ids=full_prompt_ids,
            attention_mask=attn_mask,
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
        L = full_prompt_ids.shape[1]
        return self.tokenizer.decode(
            output["sequences"][0, L:], skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        ).strip()

    # ------------------------------------------------------------------ #

    @property
    def config(self):
        return self.lm.config

    def __repr__(self) -> str:
        nd = sum(p.numel() for p in self.lm.parameters()) / 1e9
        nv = sum(p.numel() for p in self.visual.parameters()) / 1e9
        np_ = sum(p.numel() for p in self.proj.parameters()) / 1e6 if self.proj else 0
        return (
            f"VisionLM(backend={self.backend!r}  hidden={self.lm_hidden}  "
            f"gen={self._gen_mode}\n"
            f"  lm={nd:.2f}B  vis={nv:.2f}B  proj={np_:.1f}M params\n"
            f"  device={self.device}  dtype={self.dtype})"
        )
