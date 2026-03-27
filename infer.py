"""
python infer.py --text "What is in this image?" --image photo.jpg
python infer.py --text "The capital of France is"
"""

import argparse
import torch
from PIL import Image
from model import VisionDream


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True)
    parser.add_argument("--image", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--alg", default="entropy", choices=["entropy", "origin", "topk_margin"])
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model = VisionDream(device=device)

    images = [Image.open(args.image).convert("RGB")] if args.image else None

    print(model.generate(
        text=args.text,
        images=images,
        max_new_tokens=args.max_new_tokens,
        steps=args.steps,
        temperature=args.temperature,
        top_p=args.top_p,
        alg=args.alg,
    ))


if __name__ == "__main__":
    main()
