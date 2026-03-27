"""
infer.py — CLI inference for VisionDreamModel

Usage examples:
    # Text only
    python infer.py --text "What is the capital of France?"

    # Single image
    python infer.py --image photo.jpg --text "Describe this image."

    # Multiple images
    python infer.py --images a.jpg b.jpg --text "Compare these two images."

    # Tune generation
    python infer.py --image photo.jpg --text "What is this?" \
        --steps 64 --max-new-tokens 128 --temperature 0.3
"""

import argparse
import torch
from PIL import Image

from src.diffllm.vision_dream_model import VisionDreamModel


def select_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    parser = argparse.ArgumentParser(description="VisionDreamModel inference")
    parser.add_argument("--text",           type=str, required=True, help="Prompt text")
    parser.add_argument("--image",          type=str, default=None,  help="Single image path")
    parser.add_argument("--images",         type=str, nargs="+",     help="Multiple image paths")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--steps",          type=int, default=128)
    parser.add_argument("--temperature",    type=float, default=0.0)
    parser.add_argument("--top-p",          type=float, default=0.95)
    parser.add_argument("--top-k",          type=int, default=None)
    parser.add_argument("--alg",            type=str,  default="entropy",
                        choices=["entropy", "topk_margin", "maskgit_plus", "origin"])
    args = parser.parse_args()

    device = select_device()
    print(f"Device: {device}")

    model = VisionDreamModel(device=device)
    print(model)

    # Collect images
    images = None
    if args.images:
        images = [Image.open(p).convert("RGB") for p in args.images]
        print(f"Loaded {len(images)} image(s): {args.images}")
    elif args.image:
        images = [Image.open(args.image).convert("RGB")]
        print(f"Loaded image: {args.image}  size={images[0].size}")

    print(f"\nPrompt: {args.text!r}")
    print("-" * 60)

    response = model.generate(
        text=args.text,
        images=images,
        max_new_tokens=args.max_new_tokens,
        steps=args.steps,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        alg=args.alg,
    )

    print(f"Response:\n{response}")


if __name__ == "__main__":
    main()
