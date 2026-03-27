"""
bench_vision.py — 3-way benchmark: Dream vs NBDiff vs LLaDA

Usage:
    python bench_vision.py                           # all three backends
    python bench_vision.py --backend dream           # single backend
    python bench_vision.py --backend dream nbdiff    # two backends
    python bench_vision.py --image photo.jpg --text "Describe this"
"""

import argparse
import os
import time
import torch
from PIL import Image, ImageDraw, ImageFont

from src.diffllm.vision_lm import VisionLM


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
]

def _get_font(size=72):
    for p in FONT_CANDIDATES:
        try:
            return ImageFont.truetype(p, size)
        except (IOError, OSError):
            pass
    return ImageFont.load_default()

def make_text_image(text: str, size: int = 448) -> Image.Image:
    img  = Image.new("RGB", (size, size), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    font = _get_font(72)
    bbox   = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((size - tw)//2 - bbox[0], (size - th)//2 - bbox[1]),
              text, fill=(0, 0, 0), font=font)
    return img


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

class BenchSuite:
    def __init__(self, models: dict):
        """models = {"dream": VisionLM, "nbdiff": VisionLM, "llada": VisionLM}"""
        self.models = models

    def run(self, label: str, text: str, images, expected: str,
            max_new_tokens=32, steps=128, temperature=0.3):
        print(f"\n{'─'*70}")
        print(f"[{label}]")
        print(f"  Prompt  : {text!r}")
        if expected:
            print(f"  Expected: {expected!r}")
        if images:
            print(f"  Image   : {images[0].size}")

        for name, model in self.models.items():
            t0 = time.perf_counter()
            try:
                result = model.generate(
                    text=text, images=images,
                    max_new_tokens=max_new_tokens,
                    steps=steps, temperature=temperature,
                    verbose=False,
                )
            except Exception as e:
                result = f"ERROR: {e}"
            dt = time.perf_counter() - t0

            ok = ""
            if expected:
                ok = "✓" if expected.lower() in result.lower() else "✗"
            print(f"  [{name.upper():6s}] {ok}  {result!r}  ({dt:.1f}s)")

    def run_all(self, custom_image=None, custom_text=None):
        print("\n" + "=" * 70)
        print("VISION LM BENCHMARK  —  Dream vs NBDiff vs LLaDA")
        print("=" * 70)

        # 1. Text-only baseline
        self.run(
            "Text-only: capital of France",
            text="The capital of France is",
            images=None, expected="paris",
            max_new_tokens=16, steps=16, temperature=0.3,
        )

        # 2. OCR: simple bold text
        self.run(
            "OCR: HELLO",
            text="What text is written in this image? Reply with the text only.",
            images=[make_text_image("HELLO", 448)],
            expected="hello",
            max_new_tokens=16, steps=64, temperature=0.3,
        )

        # 3. OCR: two-word phrase
        self.run(
            "OCR: OPEN AI",
            text="What text is written in this image? Reply with the text only.",
            images=[make_text_image("OPEN AI", 448)],
            expected="open ai",
            max_new_tokens=16, steps=64, temperature=0.3,
        )

        # 4. Real image (lightbulb)
        real = "/home/vlmrun/Downloads/text-image-intersect.png"
        if os.path.exists(real):
            self.run(
                "Real image: lightbulb text",
                text="What text is written in this image? Reply with only the text.",
                images=[Image.open(real).convert("RGB")],
                expected="creative ideas",
                max_new_tokens=32, steps=128, temperature=0.3,
            )

        # 5. Custom image/prompt from CLI
        if custom_image and custom_text:
            self.run(
                "Custom",
                text=custom_text,
                images=[custom_image],
                expected="",
                max_new_tokens=64, steps=128, temperature=0.3,
            )

        print("\n" + "=" * 70)
        print("BENCHMARK COMPLETE")
        print("=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", nargs="+",
                        choices=["dream", "nbdiff", "llada"],
                        default=["dream", "nbdiff", "llada"])
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--text",  type=str, default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    models = {}
    for b in args.backend:
        print(f"\n{'='*60}\nLoading backend: {b.upper()}\n{'='*60}")
        models[b] = VisionLM(backend=b, device=device, max_pixels=None)
        print(models[b])

    custom_img = Image.open(args.image).convert("RGB") if args.image else None
    BenchSuite(models).run_all(custom_image=custom_img, custom_text=args.text)


if __name__ == "__main__":
    main()
