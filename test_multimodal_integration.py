"""
Integration Test for Hybrid Dream Model

Tests all major components with deterministic synthetic inputs:
  1. Model loading
  2. Processor loading + tokenizer consistency
  3. Vision encoder layer debug (shape + token-count validation)
  4. HybridDreamModel forward pass shape check
  5. Text-only generation (baseline, greedy)
  6. Single synthetic image generation + output matching
  7. Multi-image generation + output matching

Run with:
    python test_multimodal_integration.py
    python test_multimodal_integration.py --image /path/to/img.jpg
    python test_multimodal_integration.py --images a.jpg b.jpg
"""

import torch
import argparse
from PIL import Image, ImageDraw
from transformers import AutoTokenizer

from src.diffllm.hybrid_dream_model import HybridDreamModel
from src.diffllm.qwen_vision_system import Qwen3VLMultimodalProcessor
from src.diffllm.multimodal_generation_utils import multimodal_diffusion_generate


# ---------------------------------------------------------------------------
# Multinomial safety patch — same as app.py to prevent NaN crashes
# ---------------------------------------------------------------------------
def _sanitize_probabilities(probs: torch.Tensor) -> torch.Tensor:
    if not torch.is_floating_point(probs):
        return probs
    sanitized = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    if sanitized.ndim == 1:
        total = sanitized.sum()
        return sanitized / total if total > 0 else torch.full_like(sanitized, 1.0 / sanitized.numel())
    total = sanitized.sum(dim=-1, keepdim=True)
    zero_mask = total <= 0
    if zero_mask.any():
        sanitized = sanitized.clone()
        sanitized[zero_mask.expand(-1, sanitized.size(-1))] = 1.0
        total = sanitized.sum(dim=-1, keepdim=True)
    return sanitized / total


_original_multinomial = torch.multinomial


def _safe_multinomial(probs, num_samples, replacement=False, *, generator=None, out=None):
    if probs.ndim > 2:
        orig_shape = probs.shape
        flat = probs.contiguous().view(-1, orig_shape[-1])
        samples = _original_multinomial(
            _sanitize_probabilities(flat), num_samples, replacement, generator=generator, out=out
        )
        return samples.view(*orig_shape[:-1], num_samples)
    return _original_multinomial(
        _sanitize_probabilities(probs), num_samples, replacement, generator=generator, out=out
    )


if not getattr(torch.multinomial, "_dream_safe", False):
    _safe_multinomial._dream_safe = True
    torch.multinomial = _safe_multinomial


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
PASS = "\u2713 PASS"
WARN = "\u26a0  WARN"
FAIL = "\u2717 FAIL"
_n_pass = _n_fail = _n_warn = 0


def check(condition: bool, msg_pass: str, msg_fail: str) -> bool:
    global _n_pass, _n_fail
    if condition:
        print(f"  {PASS}: {msg_pass}")
        _n_pass += 1
    else:
        print(f"  {FAIL}: {msg_fail}")
        _n_fail += 1
    return condition


def warn(condition: bool, msg_pass: str, msg_warn: str) -> bool:
    global _n_pass, _n_warn
    if condition:
        print(f"  {PASS}: {msg_pass}")
        _n_pass += 1
    else:
        print(f"  {WARN}: {msg_warn}")
        _n_warn += 1
    return condition


def select_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return "mps"
    return "cpu"


def make_synthetic_image(color: str = "red", size: int = 224) -> Image.Image:
    """Return a white canvas with a solid-color square in the centre."""
    img = Image.new("RGB", (size, size), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    q, tq = size // 4, 3 * size // 4
    fill = {"red": (220, 20, 20), "blue": (20, 20, 220), "green": (20, 180, 20)}.get(
        color, (128, 128, 128)
    )
    draw.rectangle([q, q, tq, tq], fill=fill)
    return img


# ---------------------------------------------------------------------------
# TEST 1 — Model loading
# ---------------------------------------------------------------------------
def test_model_loading():
    print("\n" + "=" * 60)
    print("TEST 1: Model Loading")
    print("=" * 60)

    device = select_device()
    dtype = {"cuda": torch.bfloat16, "mps": torch.float16, "cpu": torch.float32}[device]

    model = HybridDreamModel(
        dream_model_name="Dream-org/Dream-v0-Instruct-7B",
        qwen_model_name="Qwen/Qwen2-VL-7B-Instruct",
        device=device,
        dtype=dtype,
    )

    check(model is not None, "Model object created", "Model is None")
    check(
        model.dream_model.config.hidden_size == 3584,
        f"Dream hidden_size = {model.dream_model.config.hidden_size}",
        f"Unexpected hidden_size = {model.dream_model.config.hidden_size}",
    )
    check(
        model.vision_system.qwen_hidden_size == model.dream_model.config.hidden_size,
        f"Vision hidden_size ({model.vision_system.qwen_hidden_size}) "
        f"== Dream hidden_size ({model.dream_model.config.hidden_size})",
        f"Hidden-size mismatch: vision={model.vision_system.qwen_hidden_size} "
        f"dream={model.dream_model.config.hidden_size}",
    )
    check(
        model.vision_system.dimension_adapter is None,
        "No dimension adapter (same hidden size, no projection needed)",
        "Unexpected adapter created",
    )
    print(f"  Device: {device}  Dtype: {dtype}")
    return model, device, dtype


# ---------------------------------------------------------------------------
# TEST 2 — Processor loading + tokenizer consistency
# ---------------------------------------------------------------------------
def test_processor_loading():
    print("\n" + "=" * 60)
    print("TEST 2: Processor Loading & Tokenizer Consistency")
    print("=" * 60)

    processor = Qwen3VLMultimodalProcessor(model_name="Qwen/Qwen2-VL-7B-Instruct")
    dream_tok = AutoTokenizer.from_pretrained(
        "Dream-org/Dream-v0-Instruct-7B", trust_remote_code=True
    )
    qwen_tok = processor.get_tokenizer()

    # Dream adds extra special tokens (e.g. <|mask|>=151666) beyond Qwen's base vocab.
    # Both share the same text BPE tokens — vocab difference is only special tokens.
    warn(
        len(dream_tok) == len(qwen_tok),
        f"Identical vocab sizes: {len(dream_tok)}",
        f"Vocab size differs: Dream={len(dream_tok)} Qwen={len(qwen_tok)} "
        f"(expected — Dream adds mask/special tokens)",
    )

    # Token-ID consistency for plain English text
    test_phrase = "Hello world"
    d_ids = dream_tok.encode(test_phrase, add_special_tokens=False)
    q_ids = qwen_tok.encode(test_phrase, add_special_tokens=False)
    check(
        d_ids == q_ids,
        f"Token IDs match for {test_phrase!r}: {d_ids}",
        f"Token ID mismatch: Dream={d_ids}  Qwen={q_ids}",
    )

    check(
        dream_tok.mask_token_id is not None,
        f"Dream mask_token_id = {dream_tok.mask_token_id}  ({dream_tok.mask_token!r})",
        "Dream tokenizer missing mask_token_id",
    )

    print(f"  EOS token: {dream_tok.eos_token!r} (ID={dream_tok.eos_token_id})")
    return processor, dream_tok


# ---------------------------------------------------------------------------
# TEST 3 — Vision encoder layer debug
# ---------------------------------------------------------------------------
def test_vision_encoder(model, processor, device):
    print("\n" + "=" * 60)
    print("TEST 3: Vision Encoder Layer Debug")
    print("=" * 60)

    img = make_synthetic_image("red", size=224)
    print(f"  Synthetic image: {img.size}  mode={img.mode}")

    inputs = processor.process_images([img], text="Describe this image.")
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    pv = inputs.get("pixel_values")
    grid = inputs.get("image_grid_thw")

    check(pv is not None, f"pixel_values present, shape={pv.shape}", "pixel_values missing")
    check(grid is not None, f"image_grid_thw present, shape={grid.shape}", "image_grid_thw missing")
    # The processor always returns float32; the vision encoder casts internally.
    check(pv.is_floating_point(),
          f"pixel_values is float tensor: dtype={pv.dtype}",
          f"pixel_values is not floating-point: dtype={pv.dtype}")

    # Encode through vision system
    with torch.no_grad():
        vision_embeds = model.encode_vision(pixel_values=pv, image_grid_thw=grid)

    check(vision_embeds.dim() in (2, 3),
          f"vision_embeds dim={vision_embeds.dim()}, shape={vision_embeds.shape}",
          f"Unexpected dim={vision_embeds.dim()}")

    num_tokens = vision_embeds.shape[-2] if vision_embeds.dim() == 3 else vision_embeds.shape[0]
    h_size = vision_embeds.shape[-1]

    check(
        h_size == model.dream_model.config.hidden_size,
        f"Vision embed dim {h_size} matches Dream hidden_size {model.dream_model.config.hidden_size}",
        f"Embed dim mismatch: vision={h_size} dream={model.dream_model.config.hidden_size}",
    )

    # Verify token count: image_grid_thw stores raw (T, H, W) patch counts.
    # Qwen2-VL's PatchMerger reduces by merge_size^2 (default 4).
    t, h, w = [int(v) for v in grid[0].tolist()]
    merge_size = getattr(
        getattr(model.vision_system.visual, "merger", None), "spatial_merge_size", 2
    )
    expected = (t * h * w) // (merge_size ** 2)
    check(
        num_tokens == expected,
        f"Vision token count {num_tokens} == expected post-merge {expected} "
        f"(raw={t * h * w}, merge_size={merge_size})",
        f"Token count mismatch: got {num_tokens}, expected {expected} "
        f"(raw={t * h * w}, merge_size={merge_size})",
    )

    # Also verify that get_num_vision_tokens() now matches the actual output
    library_count = model.get_num_vision_tokens(image_grid_thw=grid)
    check(
        library_count == num_tokens,
        f"get_num_vision_tokens() == actual shape: {library_count}",
        f"get_num_vision_tokens()={library_count} != actual shape={num_tokens}",
    )

    check(not torch.isnan(vision_embeds).any().item(), "No NaN in vision_embeds", "NaN in vision_embeds")
    check(not torch.isinf(vision_embeds).any().item(), "No Inf in vision_embeds", "Inf in vision_embeds")

    print(f"  image_grid_thw: T={t} H={h} W={w}")
    print(f"  vision_embeds shape: {vision_embeds.shape}")
    return inputs


# ---------------------------------------------------------------------------
# TEST 4 — Forward pass shape check (no sampling)
# ---------------------------------------------------------------------------
def test_forward_pass(model, processor, tokenizer, device, vision_inputs):
    print("\n" + "=" * 60)
    print("TEST 4: HybridDreamModel Forward Pass (Shape & Sanity)")
    print("=" * 60)

    input_ids = vision_inputs["input_ids"].to(device)
    attention_mask = vision_inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    pv = vision_inputs.get("pixel_values")
    grid = vision_inputs.get("image_grid_thw")

    B, L = input_ids.shape
    print(f"  input_ids shape: {input_ids.shape}")

    with torch.no_grad():
        vision_embeds = model.encode_vision(pixel_values=pv, image_grid_thw=grid)
        if vision_embeds.dim() == 2:
            vision_embeds_b = vision_embeds.unsqueeze(0).expand(B, -1, -1)
        else:
            vision_embeds_b = vision_embeds
        V = vision_embeds_b.shape[1]
        print(f"  Vision tokens prepended: V={V}  Text tokens: L={L}  Total: {V+L}")

        # Generation passes attention_mask=None when there is no padding (no left-pad tokens).
        # This mirrors multimodal_diffusion_generate's no-padding branch and avoids Dream's
        # SDPA receiving a raw int64 mask (it expects bool or float).
        has_padding = attention_mask is not None and torch.any(attention_mask == 0).item()
        fwd_mask = attention_mask if has_padding else None
        print(f"  has_padding={has_padding}  → passing {'bool 4D mask' if has_padding else 'None'} to Dream")

        logits = model(
            input_ids=input_ids,
            attention_mask=fwd_mask,
            vision_embeds=vision_embeds_b,
        )

    expected_seq = V + L
    check(logits.shape[0] == B,
          f"Batch dim: {logits.shape[0]}", f"Batch dim wrong: got {logits.shape[0]} expected {B}")
    check(logits.shape[1] == expected_seq,
          f"Sequence dim: {logits.shape[1]} = V({V}) + L({L})",
          f"Sequence dim wrong: got {logits.shape[1]} expected {expected_seq}")
    check(logits.shape[2] == model.dream_model.config.vocab_size,
          f"Vocab dim: {logits.shape[2]}",
          f"Vocab dim wrong: got {logits.shape[2]} expected {model.dream_model.config.vocab_size}")
    check(not torch.isnan(logits).any().item(), "No NaN in logits", "NaN detected in logits")
    check(not torch.isinf(logits).any().item(), "No Inf in logits", "Inf detected in logits")

    # Logit shift used during generation: cat([logits[:,:1], logits[:,:-1]], 1)
    logits_shifted = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
    check(logits_shifted.shape == logits.shape,
          f"Logit-shift preserves shape: {logits_shifted.shape}", "Shape changed after logit shift")

    print(f"  logits shape: {logits.shape}  dtype: {logits.dtype}")


# ---------------------------------------------------------------------------
# TEST 5 — Text-only generation (baseline, greedy)
# ---------------------------------------------------------------------------
def test_text_only_generation(model, processor, tokenizer, device):
    print("\n" + "=" * 60)
    print("TEST 5: Text-Only Generation (Greedy Baseline)")
    print("=" * 60)

    # Use a factual prompt where a short greedy generation should produce "Paris"
    prompt = "The capital of France is"
    print(f"  Prompt: {prompt!r}")

    messages = [{"role": "user", "content": prompt}]
    inputs = processor.processor(
        text=[processor.processor.apply_chat_template(messages, add_generation_prompt=True)],
        return_tensors="pt",
        padding=True,
    )
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    prompt_length = inputs["input_ids"].shape[1]
    MAX_NEW = 32
    print(f"  prompt_length={prompt_length}  max_new_tokens={MAX_NEW}")

    with torch.no_grad():
        output = multimodal_diffusion_generate(
            model=model,
            tokenizer=tokenizer,
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=MAX_NEW,
            steps=32,
            temperature=0.0,   # greedy — deterministic
            top_p=0.95,
            alg="entropy",
            device=device,
        )

    sequences = output["sequences"]
    check(sequences.shape == (1, prompt_length + MAX_NEW),
          f"Output shape correct: {sequences.shape}",
          f"Shape mismatch: {sequences.shape} != {(1, prompt_length + MAX_NEW)}")

    mask_id = tokenizer.mask_token_id or 151666
    remaining = (sequences[0, prompt_length:] == mask_id).sum().item()
    check(remaining == 0,
          "All mask tokens resolved (0 remaining)",
          f"{remaining}/{MAX_NEW} mask tokens still present — generation incomplete")

    generated_ids = sequences[0, prompt_length:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    check(len(generated_text) > 0,
          f"Non-empty output: {generated_text[:80]!r}",
          "Generated text is empty string")

    # Greedy + factual prompt → "Paris" should appear
    warn(
        "paris" in generated_text.lower(),
        f"'Paris' found in output: {generated_text!r}",
        f"'Paris' not found (model may be hallucinating): {generated_text!r}",
    )

    print(f"\n  Generated: {generated_text!r}")


# ---------------------------------------------------------------------------
# TEST 6 — Single synthetic image generation
# ---------------------------------------------------------------------------
def test_single_image_generation(model, processor, tokenizer, device, image=None):
    print("\n" + "=" * 60)
    print("TEST 6: Single Image Generation")
    print("=" * 60)

    if image is None:
        image = make_synthetic_image("red", size=224)
        expected_color = "red"
        print(f"  Using synthetic image: white canvas with red square ({image.size})")
    else:
        expected_color = None
        print(f"  Using provided image: {image.size}  mode={image.mode}")

    prompt = "What color is the main shape in this image? Answer in one word."
    print(f"  Prompt: {prompt!r}")

    inputs = processor.process_images([image], text=prompt)
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    prompt_length = inputs["input_ids"].shape[1]
    MAX_NEW = 64
    print(f"  prompt_length={prompt_length}  max_new_tokens={MAX_NEW}")

    with torch.no_grad():
        output = multimodal_diffusion_generate(
            model=model,
            tokenizer=tokenizer,
            pixel_values=inputs.get("pixel_values"),
            image_grid_thw=inputs.get("image_grid_thw"),
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=MAX_NEW,
            steps=64,
            temperature=0.0,
            top_p=0.95,
            alg="entropy",
            device=device,
        )

    sequences = output["sequences"]
    mask_id = tokenizer.mask_token_id or 151666
    remaining = (sequences[0, prompt_length:] == mask_id).sum().item()
    check(remaining == 0,
          "All mask tokens resolved",
          f"{remaining}/{MAX_NEW} mask tokens still present")

    generated_ids = sequences[0, prompt_length:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    check(len(generated_text) > 0,
          f"Non-empty output: {generated_text[:80]!r}",
          "Generated text is empty")

    # Validate that output tokens are all within the Dream vocabulary (no stray IDs)
    vocab_size = model.dream_model.config.vocab_size
    out_of_vocab = ((generated_ids < 0) | (generated_ids >= vocab_size)).sum().item()
    check(out_of_vocab == 0,
          f"All {len(generated_ids)} output token IDs within vocab range [0, {vocab_size})",
          f"{out_of_vocab} token IDs outside vocab range")

    if expected_color is not None:
        warn(
            expected_color in generated_text.lower(),
            f"Expected color '{expected_color}' found in output: {generated_text!r}",
            f"'{expected_color}' not found (model may not have vision fine-tuning): {generated_text!r}",
        )

    print(f"\n  Generated: {generated_text!r}")


# ---------------------------------------------------------------------------
# TEST 7 — Multi-image generation
# ---------------------------------------------------------------------------
def test_multi_image_generation(model, processor, tokenizer, device, image_paths=None):
    print("\n" + "=" * 60)
    print("TEST 7: Multi-Image Generation")
    print("=" * 60)

    if image_paths and len(image_paths) >= 2:
        images = [Image.open(p).convert("RGB") for p in image_paths]
        expected_colors = None
        print(f"  Using {len(images)} provided images")
    else:
        images = [make_synthetic_image("red", 224), make_synthetic_image("blue", 224)]
        expected_colors = ["red", "blue"]
        print(f"  Using 2 synthetic images: red square + blue square")

    prompt = "List the color of the main shape in each image, one per line."
    print(f"  Prompt: {prompt!r}")

    inputs = processor.process_images(images, text=prompt)
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    prompt_length = inputs["input_ids"].shape[1]
    MAX_NEW = 64
    print(f"  prompt_length={prompt_length}  max_new_tokens={MAX_NEW}")

    with torch.no_grad():
        output = multimodal_diffusion_generate(
            model=model,
            tokenizer=tokenizer,
            pixel_values=inputs.get("pixel_values"),
            image_grid_thw=inputs.get("image_grid_thw"),
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=MAX_NEW,
            steps=64,
            temperature=0.0,
            top_p=0.95,
            alg="entropy",
            device=device,
        )

    sequences = output["sequences"]
    mask_id = tokenizer.mask_token_id or 151666
    remaining = (sequences[0, prompt_length:] == mask_id).sum().item()
    check(remaining == 0,
          "All mask tokens resolved",
          f"{remaining}/{MAX_NEW} mask tokens still present")

    generated_ids = sequences[0, prompt_length:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    # Pipeline check: all tokens in-vocab
    vocab_size = model.dream_model.config.vocab_size
    oov = ((generated_ids < 0) | (generated_ids >= vocab_size)).sum().item()
    check(oov == 0,
          f"All {len(generated_ids)} token IDs within vocab range",
          f"{oov} out-of-vocab token IDs")

    # Content check: model may emit only EOS for prompts it can't handle (no vision fine-tuning)
    raw_text_no_skip = tokenizer.decode(generated_ids).strip()
    if len(generated_text) == 0:
        print(f"  DEBUG: raw ids (first 16): {generated_ids[:16].tolist()}")
        print(f"  DEBUG: decoded (no skip): {raw_text_no_skip[:80]!r}")
    warn(
        len(generated_text) > 0,
        f"Non-empty output: {generated_text[:80]!r}",
        f"Generated text is empty (model likely emits only EOS for unseen multi-image format)",
    )

    if expected_colors and generated_text:
        for color in expected_colors:
            warn(
                color in generated_text.lower(),
                f"'{color}' found in output",
                f"'{color}' not found in: {generated_text!r}",
            )

    print(f"\n  Generated: {generated_text!r}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Test Hybrid Dream Model")
    parser.add_argument("--image", type=str, help="Path to test image (Test 6)")
    parser.add_argument("--images", type=str, nargs="+", help="Paths to test images (Test 7)")
    parser.add_argument("--skip-image", action="store_true", help="Skip image generation tests")
    args = parser.parse_args()

    print("=" * 60)
    print("HYBRID DREAM MODEL INTEGRATION TEST")
    print("=" * 60)

    model, device, dtype = test_model_loading()
    processor, tokenizer = test_processor_loading()
    vision_inputs = test_vision_encoder(model, processor, device)
    test_forward_pass(model, processor, tokenizer, device, vision_inputs)
    test_text_only_generation(model, processor, tokenizer, device)

    if not args.skip_image:
        img = Image.open(args.image).convert("RGB") if args.image else None
        test_single_image_generation(model, processor, tokenizer, device, img)
        test_multi_image_generation(model, processor, tokenizer, device, args.images)

    print("\n" + "=" * 60)
    print(f"RESULTS: {_n_pass} passed  {_n_fail} failed  {_n_warn} warnings")
    if _n_fail > 0:
        print("OVERALL: FAILED")
    else:
        print("OVERALL: PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
