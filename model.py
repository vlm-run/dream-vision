"""
VisionDream — Qwen2-VL visual encoder + Dream-v0 diffusion LM

Architecture:
    image → Qwen2-VL ViT → vision_embeds [N, H]
    text  → Qwen processor → input_ids (with <|image_pad|> placeholders)
    inject vision_embeds at image_pad positions → Dream transformer → masked diffusion
"""

import gc
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModel, AutoTokenizer, AutoProcessor, Qwen2VLForConditionalGeneration


# NaN-safe multinomial (diffusion sampling can produce degenerate distributions)
def _patch_multinomial():
    def _sanitize(p):
        p = torch.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
        s = p.sum(dim=-1, keepdim=True) if p.ndim > 1 else p.sum()
        zero = s <= 0
        if (zero if p.ndim == 1 else zero.any()):
            p = torch.where(
                zero.expand_as(p) if p.ndim > 1 else zero,
                torch.full_like(p, 1.0 / p.shape[-1]), p,
            )
            s = p.sum(dim=-1, keepdim=True) if p.ndim > 1 else p.sum()
        return p / s

    _orig = torch.multinomial

    def _safe(p, n, replacement=False, *, generator=None, out=None):
        if p.ndim > 2:
            sh = p.shape
            return _orig(_sanitize(p.contiguous().view(-1, sh[-1])), n, replacement,
                         generator=generator, out=out).view(*sh[:-1], n)
        return _orig(_sanitize(p), n, replacement, generator=generator, out=out)

    if not getattr(torch.multinomial, "_vd_patched", False):
        _safe._vd_patched = True
        torch.multinomial = _safe


_patch_multinomial()


class VisionDream(nn.Module):
    DREAM = "Dream-org/Dream-v0-Instruct-7B"
    QWEN  = "Qwen/Qwen2-VL-7B-Instruct"

    def __init__(
        self,
        dream_model: str = DREAM,
        qwen_model: str = QWEN,
        device: str = "cuda",
        dtype: Optional[torch.dtype] = None,
        max_pixels: int = 1024 * 1024,
    ):
        super().__init__()
        self.device = device
        self.max_pixels = max_pixels
        self.dtype = dtype or (
            torch.bfloat16 if device == "cuda" else
            torch.float16  if device == "mps"  else
            torch.float32
        )

        print(f"Loading Dream ({dream_model})...")
        self.dream = AutoModel.from_pretrained(
            dream_model, dtype=self.dtype, trust_remote_code=True,
        ).to(device).eval()
        for p in self.dream.parameters():
            p.requires_grad = False

        print(f"Loading Qwen2-VL vision encoder ({qwen_model})...")
        _full = Qwen2VLForConditionalGeneration.from_pretrained(
            qwen_model, dtype=self.dtype, device_map=device,
        )
        self.visual = _full.visual.eval()
        for p in self.visual.parameters():
            p.requires_grad = False
        del _full.model, _full.lm_head, _full
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

        self.processor = AutoProcessor.from_pretrained(qwen_model)
        self.tokenizer = AutoTokenizer.from_pretrained(dream_model, trust_remote_code=True)

        self.image_pad_id  = self.processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        self.mask_token_id = self.tokenizer.mask_token_id or 151666
        print(f"Ready. image_pad={self.image_pad_id}  mask={self.mask_token_id}  "
              f"device={device}  dtype={self.dtype}")

    # ------------------------------------------------------------------

    def _resize(self, images):
        if not self.max_pixels:
            return images
        out = []
        for img in images:
            w, h = img.size
            if w * h > self.max_pixels:
                s = (self.max_pixels / (w * h)) ** 0.5
                img = img.resize((max(1, int(w * s)), max(1, int(h * s))), Image.LANCZOS)
            out.append(img)
        return out

    def preprocess(self, text: str, images: Optional[List[Image.Image]] = None) -> dict:
        if images:
            images = self._resize(list(images))
            content = [{"type": "image"} for _ in images] + [{"type": "text", "text": text}]
            msgs = [{"role": "user", "content": content}]
            return self.processor(
                text=[self.processor.apply_chat_template(msgs, add_generation_prompt=True)],
                images=images, return_tensors="pt", padding=True,
            )
        msgs = [{"role": "user", "content": text}]
        return self.processor(
            text=[self.processor.apply_chat_template(msgs, add_generation_prompt=True)],
            return_tensors="pt", padding=True,
        )

    @torch.no_grad()
    def encode_vision(self, pixel_values, image_grid_thw):
        pv = pixel_values.type(self.visual.get_dtype())
        return self.visual(pv, grid_thw=image_grid_thw).to(self.dtype)

    def _embed(self, input_ids, vision_embeds=None):
        embeds = self.dream.model.embed_tokens(input_ids)
        if vision_embeds is not None:
            mask = input_ids == self.image_pad_id
            n = mask.sum().item()
            if n != vision_embeds.shape[0]:
                raise ValueError(f"{n} image_pad tokens but {vision_embeds.shape[0]} vision rows")
            if n > 0:
                embeds = embeds.clone()
                embeds[mask] = vision_embeds
        return embeds

    def forward(self, input_ids, vision_embeds=None, attention_mask=None, **kw):
        inputs_embeds = self._embed(input_ids, vision_embeds)
        out = self.dream(inputs_embeds=inputs_embeds, attention_mask=attention_mask, **kw)
        return out.logits if hasattr(out, "logits") else out

    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        text: str,
        images: Optional[List[Image.Image]] = None,
        max_new_tokens: int = 128,
        steps: int = 64,
        temperature: float = 0.3,
        top_p: float = 0.95,
        alg: str = "entropy",
    ) -> str:
        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                  for k, v in self.preprocess(text, images).items()}

        vision_embeds = None
        if "pixel_values" in inputs:
            vision_embeds = self.encode_vision(inputs["pixel_values"], inputs["image_grid_thw"])

        out = _generate(self, inputs["input_ids"], inputs.get("attention_mask"),
                        vision_embeds, max_new_tokens, steps, temperature, top_p, alg)

        prompt_len = inputs["input_ids"].shape[1]
        return self.tokenizer.decode(
            out[0, prompt_len:], skip_special_tokens=True, clean_up_tokenization_spaces=True,
        ).strip()

    @property
    def config(self):
        return self.dream.config

    @property
    def model(self):
        return self.dream.model


# ------------------------------------------------------------------
# Masked diffusion generation
# ------------------------------------------------------------------

def _sample(logits, temperature, top_p, margin=False, neg_entropy=False):
    if temperature > 0:
        logits = logits / temperature
    if top_p < 1.0:
        sorted_l, sorted_i = torch.sort(logits, descending=True)
        cum = torch.cumsum(F.softmax(sorted_l, dim=-1), dim=-1)
        remove = cum > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        logits = logits.masked_fill(
            torch.zeros_like(logits, dtype=torch.bool).scatter_(-1, sorted_i, remove),
            torch.finfo(logits.dtype).min,
        )

    probs = F.softmax(logits, dim=-1)
    if temperature > 0:
        try:
            x0 = torch.multinomial(probs, 1).squeeze(-1)
            conf = torch.gather(probs, -1, x0.unsqueeze(-1)).squeeze(-1)
        except Exception:
            conf, x0 = probs.max(-1)
    else:
        conf, x0 = probs.max(-1)

    if margin:
        sp, _ = torch.sort(probs, -1, descending=True)
        conf = sp[:, 0] - sp[:, 1]
    if neg_entropy:
        conf = (probs * torch.log(probs + 1e-10)).sum(-1)
    return conf, x0


def _generate(
    model: VisionDream,
    input_ids: torch.Tensor,
    attention_mask,
    vision_embeds,
    max_new_tokens: int,
    steps: int,
    temperature: float,
    top_p: float,
    alg: str,
    eps: float = 1e-3,
) -> torch.Tensor:
    device = model.device
    B, L = input_ids.shape
    mask_id = model.mask_token_id

    x = torch.cat([input_ids, torch.full((B, max_new_tokens), mask_id, dtype=torch.long, device=device)], dim=1)

    has_pad = attention_mask is not None and attention_mask.eq(0).any()
    if has_pad:
        gm = torch.ones(B, max_new_tokens, dtype=attention_mask.dtype, device=device)
        full_mask = torch.cat([attention_mask, gm], dim=1)
        tok_idx = full_mask.long().cumsum(-1) - 1
        tok_idx.masked_fill_(full_mask == 0, 1)
        attn = torch.logical_and(full_mask[:, None, None, :], full_mask[:, None, :, None])
        pos = tok_idx
    else:
        attn, pos = None, None

    timesteps = torch.linspace(1.0, eps, steps + 1, device=device)

    for i in range(steps):
        mask_idx = x == mask_id
        logits = model(input_ids=x, vision_embeds=vision_embeds, attention_mask=attn, position_ids=pos)
        logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)  # Dream's one-step shift

        gen_logits = logits[:, L:]
        gen_mask   = mask_idx[:, L:]
        masked_logits = gen_logits[gen_mask]

        t, s = timesteps[i], timesteps[i + 1]

        conf, x0 = _sample(
            masked_logits, temperature, top_p,
            margin=(alg == "topk_margin"), neg_entropy=(alg == "entropy"),
        )

        num_mask = gen_mask.sum() / B
        n_reveal = int(num_mask * (1 - s / t)) if i < steps - 1 else int(num_mask)

        if n_reveal > 0:
            full_conf = torch.full((B, max_new_tokens), float("-inf"), dtype=logits.dtype, device=device)
            full_conf[gen_mask] = conf
            _, reveal_idx = torch.topk(full_conf, n_reveal)
            x0_full = torch.full((B, max_new_tokens), mask_id, dtype=torch.long, device=device)
            x0_full[gen_mask] = x0
            rows = torch.arange(B, device=device).unsqueeze(1).expand_as(reveal_idx)
            x[:, L:][rows, reveal_idx] = x0_full[rows, reveal_idx]

    return x
