"""
OCR Integration Test for VisionDreamModel

Creates an image with simple text, asks the model to reproduce it.
This validates that vision features flow correctly through injection
into Dream's transformer and produce coherent text output.

Usage:
    python test_ocr.py
    python test_ocr.py --text "OPEN AI"
    python test_ocr.py --save-image /tmp/test_ocr.png
"""

import argparse
import sys
import torch
from PIL import Image, ImageDraw, ImageFont

from src.diffllm.vision_dream_model import VisionDreamModel


# ---------------------------------------------------------------------------
# Image generation helpers
# ---------------------------------------------------------------------------
FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    "/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
]


def _get_font(size: int = 72):
    for path in FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, size)
        except (IOError, OSError):
            pass
    # Fallback to PIL default (tiny but always available)
    return ImageFont.load_default()


def make_text_image(text: str, img_size: int = 448, font_size: int = 72) -> Image.Image:
    """
    Create a white image with black bold text centred on it.
    Uses a large canvas (448px) so the text is legible after processing.
    """
    img  = Image.new("RGB", (img_size, img_size), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    font = _get_font(font_size)

    # Centre the text
    bbox   = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x = (img_size - text_w) // 2 - bbox[0]
    y = (img_size - text_h) // 2 - bbox[1]
    draw.text((x, y), text, fill=(0, 0, 0), font=font)
    return img


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------
PASS = "\u2713 PASS"
WARN = "\u26a0  WARN"
FAIL = "\u2717 FAIL"
_results = {"pass": 0, "fail": 0, "warn": 0}


def check(cond: bool, ok: str, fail: str):
    if cond:
        print(f"  {PASS}: {ok}")
        _results["pass"] += 1
    else:
        print(f"  {FAIL}: {fail}")
        _results["fail"] += 1
    return cond


def warn(cond: bool, ok: str, w: str):
    if cond:
        print(f"  {PASS}: {ok}")
        _results["pass"] += 1
    else:
        print(f"  {WARN}: {w}")
        _results["warn"] += 1
    return cond


# ---------------------------------------------------------------------------
# Test 1: pipeline sanity — image_pad injection matches vision_embeds shape
# ---------------------------------------------------------------------------
def test_injection_alignment(model: VisionDreamModel, text_img: Image.Image):
    print("\n" + "=" * 60)
    print("TEST 1: Vision Injection Alignment")
    print("=" * 60)
    print("  Checks that #image_pad tokens == #vision_embeds rows.")

    inputs = model.preprocess("What text is in this image?", images=[text_img])
    inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v
              for k, v in inputs.items()}

    n_pads = (inputs["input_ids"] == model.image_pad_id).sum().item()
    print(f"  input_ids shape : {inputs['input_ids'].shape}")
    print(f"  image_pad tokens: {n_pads}")

    with torch.no_grad():
        ve = model.encode_vision(inputs["pixel_values"], inputs["image_grid_thw"])

    n_vis = ve.shape[0]
    print(f"  vision_embeds   : {ve.shape}")

    check(n_pads == n_vis,
          f"image_pad count ({n_pads}) == vision_embeds rows ({n_vis})",
          f"MISMATCH: image_pad={n_pads}  vision_embeds={n_vis}  — injection would crash")
    check(not torch.isnan(ve).any().item(), "No NaN in vision_embeds", "NaN in vision_embeds")
    check(not torch.isinf(ve).any().item(), "No Inf in vision_embeds", "Inf in vision_embeds")

    return inputs, ve


# ---------------------------------------------------------------------------
# Test 2: forward pass — shape and value sanity after injection
# ---------------------------------------------------------------------------
def test_forward_pass(model: VisionDreamModel, inputs: dict, ve: torch.Tensor):
    print("\n" + "=" * 60)
    print("TEST 2: Forward Pass With Vision Injection")
    print("=" * 60)

    input_ids = inputs["input_ids"]
    B, L = input_ids.shape
    V = ve.shape[0]

    with torch.no_grad():
        logits = model(
            input_ids=input_ids,
            vision_embeds=ve,
            attention_mask=None,   # no padding → no mask (matches generation behaviour)
        )

    # After injection the sequence length is still L (no prepend!)
    check(logits.shape == (B, L, model.dream.config.vocab_size),
          f"logits shape correct: {logits.shape}",
          f"logits shape wrong: {logits.shape}  expected ({B}, {L}, {model.dream.config.vocab_size})")

    check(not torch.isnan(logits).any().item(), "No NaN in logits", "NaN in logits")
    check(not torch.isinf(logits).any().item(), "No Inf in logits", "Inf in logits")

    # Logits at image_pad positions should differ from those at regular text positions
    pad_mask  = (input_ids == model.image_pad_id)[0]   # [L]
    text_mask = ~pad_mask
    if pad_mask.any() and text_mask.any():
        pad_var  = logits[0, pad_mask].float().var().item()
        text_var = logits[0, text_mask].float().var().item()
        print(f"  logit variance — vision positions: {pad_var:.4f}   text positions: {text_var:.4f}")
        check(pad_var > 0,
              "Vision-injected positions have non-zero logit variance",
              "Vision positions have zero variance — injection may be silent")

    print(f"  logits: {logits.shape}  dtype={logits.dtype}")


# ---------------------------------------------------------------------------
# Test 3: OCR generation
# ---------------------------------------------------------------------------
def test_ocr_generation(
    model: VisionDreamModel,
    target_text: str,
    save_image: str = None,
    max_new_tokens: int = 32,
    steps: int = 64,
):
    print("\n" + "=" * 60)
    print(f"TEST 3: OCR Generation  (target = {target_text!r})")
    print("=" * 60)

    img = make_text_image(target_text, img_size=448, font_size=72)
    if save_image:
        img.save(save_image)
        print(f"  Saved test image → {save_image}")
    print(f"  Image size: {img.size}  mode={img.mode}")

    prompt = "What text is written in this image? Reply with only the text."
    print(f"  Prompt: {prompt!r}")

    generated = model.generate(
        text=prompt,
        images=[img],
        max_new_tokens=max_new_tokens,
        steps=steps,
        temperature=0.0,
        alg="entropy",
        verbose=True,
    )

    print(f"\n  Target   : {target_text!r}")
    print(f"  Generated: {generated!r}")

    # Pipeline checks (always required)
    check(len(generated) > 0,
          "Generated text is non-empty",
          "Generated text is empty (model produced only special tokens)")

    # Content check (may fail before any vision fine-tuning)
    warn(
        target_text.lower() in generated.lower(),
        f"Target text found in output: {generated!r}",
        f"Target text NOT found — model may need vision projection training. "
        f"Generated: {generated!r}",
    )

    return generated


# ---------------------------------------------------------------------------
# Test 4: text-only baseline (no images — sanity check)
# ---------------------------------------------------------------------------
def test_text_baseline(model: VisionDreamModel):
    print("\n" + "=" * 60)
    print("TEST 4: Text-Only Baseline")
    print("=" * 60)

    prompt = "The capital of France is"
    print(f"  Prompt: {prompt!r}")

    generated = model.generate(
        text=prompt,
        images=None,
        max_new_tokens=16,
        steps=16,
        temperature=0.0,
        verbose=True,
    )

    print(f"  Generated: {generated!r}")
    check(len(generated) > 0, "Non-empty output", "Empty output")
    warn("paris" in generated.lower(),
         f"'Paris' found: {generated!r}",
         f"'Paris' not found: {generated!r}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text",       default="HELLO",  help="Text to put in the image")
    parser.add_argument("--save-image", default=None,     help="Save the test image to this path")
    parser.add_argument("--steps",      type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load the unified model
    model = VisionDreamModel(device=device, use_proj=False)
    print(model)

    # Run tests
    img = make_text_image(args.text, img_size=448, font_size=72)
    inputs, ve = test_injection_alignment(model, img)
    test_forward_pass(model, inputs, ve)
    test_ocr_generation(model, args.text,
                        save_image=args.save_image,
                        max_new_tokens=args.max_new_tokens,
                        steps=args.steps)
    test_text_baseline(model)

    print("\n" + "=" * 60)
    p, f, w = _results["pass"], _results["fail"], _results["warn"]
    print(f"RESULTS: {p} passed  {f} failed  {w} warnings")
    print("OVERALL:", "PASSED" if f == 0 else "FAILED")
    print("=" * 60)


if __name__ == "__main__":
    main()
