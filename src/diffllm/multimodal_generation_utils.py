"""
Multimodal Diffusion Generation Utilities

Adapts Dream's diffusion-based generation for vision-conditioned text generation.
Key difference: Vision embeddings are fixed (not masked), only text tokens are denoised.

Generation algorithm mirrors Dream's internal _sample() method:
  - Logit shift:  logits = cat([logits[:,:1], logits[:,:-1]], dim=1)
  - Timestep schedule: linspace(1, eps, steps+1) with proportional unmasking p = 1 - s/t
  - Attention mask: None (full) when no padding; 4D bool mask when padding present
"""

import torch
import torch.nn.functional as F
from typing import Optional, List


# Re-export from Dream's own generation_utils so callers get the correct implementation
try:
    from transformers_modules.Dream_hyphen_org.Dream_hyphen_v0_hyphen_Instruct_hyphen_7B.\
        modeling_dream import sample_tokens as _dream_sample_tokens  # noqa: F401
    _sample_tokens = _dream_sample_tokens
except Exception:
    from .gen_utils import sample_tokens as _sample_tokens


def multimodal_diffusion_generate(
    model,
    tokenizer,
    pixel_values: Optional[torch.Tensor] = None,
    pixel_values_videos: Optional[torch.Tensor] = None,
    image_grid_thw: Optional[torch.Tensor] = None,
    video_grid_thw: Optional[torch.Tensor] = None,
    input_ids: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    max_new_tokens: int = 256,
    steps: int = 128,
    eps: float = 1e-3,
    temperature: float = 0.2,
    top_p: float = 0.95,
    top_k: Optional[int] = None,
    alg: str = "entropy",
    alg_temp: Optional[float] = None,
    output_history: bool = False,
    generation_tokens_hook_func=None,
    device: str = "cuda",
) -> dict:
    """
    Generate text conditioned on vision inputs using Dream's masked diffusion.

    Mirrors Dream's _sample() algorithm exactly, extended to support prepended
    vision embeddings (which are never masked and shift the logit extraction window).

    Returns:
        dict with 'sequences' [B, prompt_len + max_new_tokens] and optionally 'history'
    """

    batch_size = input_ids.shape[0]
    prompt_length = input_ids.shape[1]

    # ------------------------------------------------------------------ #
    # 1. Encode vision inputs (fixed – never masked)                       #
    # ------------------------------------------------------------------ #
    vision_embeds = None
    num_vision_tokens = 0

    if pixel_values is not None or pixel_values_videos is not None:
        print("Encoding vision inputs...")
        vision_embeds = model.encode_vision(
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
        )
        # Qwen visual returns [num_tokens, H] (no batch dim) → [B, num_tokens, H]
        if vision_embeds.dim() == 2:
            vision_embeds = vision_embeds.unsqueeze(0).expand(batch_size, -1, -1)
        num_vision_tokens = vision_embeds.shape[1]
        print(f"  Vision tokens: {num_vision_tokens}")

    # ------------------------------------------------------------------ #
    # 2. Initialise generation sequence with mask tokens                   #
    # ------------------------------------------------------------------ #
    mask_token_id = tokenizer.mask_token_id
    if mask_token_id is None:
        mask_token_id = 151666  # Dream <|mask|>

    # x = [prompt | mask * max_new_tokens]
    generation_ids = torch.full(
        (batch_size, max_new_tokens),
        mask_token_id,
        dtype=torch.long,
        device=device,
    )
    x = torch.cat([input_ids, generation_ids], dim=1)  # [B, L+G]

    # ------------------------------------------------------------------ #
    # 3. Build attention mask (matches Dream's _sample logic)              #
    # ------------------------------------------------------------------ #
    has_padding = (attention_mask is not None) and torch.any(attention_mask == 0).item()

    if has_padding:
        # Extend prompt attention mask to cover the full generation sequence
        gen_mask = torch.ones((batch_size, max_new_tokens), dtype=attention_mask.dtype, device=device)
        full_mask_1d = torch.cat([attention_mask, gen_mask], dim=1)  # [B, L+G]

        # Compute position ids (handles left-padding)
        tok_idx = full_mask_1d.long().cumsum(-1) - 1
        tok_idx.masked_fill_(full_mask_1d == 0, 1)

        # Expand to 4D [B, 1, L+G, L+G] as Dream does
        attn_mask_4d = torch.logical_and(
            full_mask_1d.unsqueeze(1).unsqueeze(-2),
            full_mask_1d.unsqueeze(1).unsqueeze(-1),
        )

        # If vision is prepended, extend the 4D mask to cover vision tokens
        if num_vision_tokens > 0:
            B, _, N, _ = attn_mask_4d.shape
            V = num_vision_tokens
            full_4d = torch.ones((B, 1, V + N, V + N), dtype=torch.bool, device=device)
            full_4d[:, :, V:, V:] = attn_mask_4d
            attn_mask_4d = full_4d
            # Extend tok_idx for vision positions
            vis_tok_idx = torch.arange(V, device=device).unsqueeze(0).expand(batch_size, -1)
            tok_idx = torch.cat([vis_tok_idx, tok_idx + V], dim=1)

        dream_attention_mask = attn_mask_4d
        dream_position_ids = tok_idx
    else:
        # No padding → Dream passes attention_mask="full" (i.e. None)
        dream_attention_mask = None
        dream_position_ids = None
        if num_vision_tokens > 0:
            # Provide explicit position_ids so vision and text positions are correct
            total_len = num_vision_tokens + prompt_length + max_new_tokens
            dream_position_ids = torch.arange(total_len, device=device).unsqueeze(0).expand(batch_size, -1)

    # ------------------------------------------------------------------ #
    # 4. Diffusion timestep schedule (mirrors Dream's linspace schedule)   #
    # ------------------------------------------------------------------ #
    timesteps = torch.linspace(1, eps, steps + 1, device=device)
    histories = [] if output_history else None

    print(f"Starting diffusion generation: {steps} steps, {max_new_tokens} tokens")

    # ------------------------------------------------------------------ #
    # 5. Diffusion denoising loop                                          #
    # ------------------------------------------------------------------ #
    for i in range(steps):
        mask_index = (x == mask_token_id)  # [B, L+G]

        # -- Forward pass with optional vision conditioning --
        text_embeds = model.get_embedding_layer()(x)  # [B, L+G, H]

        logits = model(
            inputs_embeds=text_embeds,
            attention_mask=dream_attention_mask,
            position_ids=dream_position_ids,
            vision_embeds=vision_embeds,
        )
        # logits shape: [B, V+L+G, vocab]

        # -- Critical: Dream's one-position logit shift --
        # Applied to the full sequence (vision + text + generation)
        logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)

        # -- Extract logits for generation positions only --
        # Vision tokens are prepended, so generation logits start at V+L
        gen_start = num_vision_tokens + prompt_length
        gen_logits = logits[:, gen_start:gen_start + max_new_tokens, :]  # [B, G, vocab]

        # -- Sample from masked positions only --
        mask_logits = gen_logits[mask_index[:, prompt_length:]]  # flatten masked positions

        t = timesteps[i]
        s = timesteps[i + 1]

        if alg == "origin":
            p_transfer = (1 - s / t) if i < steps - 1 else 1.0
            x0 = torch.full_like(x[:, prompt_length:][mask_index[:, prompt_length:]], mask_token_id)
            transfer = torch.rand(x0.shape, device=device) < p_transfer
            if transfer.any():
                _, sampled = _sample_tokens(
                    mask_logits[transfer], temperature=temperature, top_p=top_p, top_k=top_k
                )
                x0[transfer] = sampled
            x[:, prompt_length:][mask_index[:, prompt_length:]] = x0

        else:
            # maskgit_plus / topk_margin / entropy
            margin = (alg == "topk_margin")
            neg_ent = (alg == "entropy")
            confidence, x0 = _sample_tokens(
                mask_logits, temperature=temperature, top_p=top_p, top_k=top_k,
                margin_confidence=margin, neg_entropy=neg_ent,
            )

            num_mask = mask_index[:, prompt_length:].sum() / batch_size
            n_transfer = int(num_mask * (1 - s / t)) if i < steps - 1 else int(num_mask)

            if n_transfer > 0:
                # Build full-size confidence tensor for topk selection
                full_conf = torch.full(
                    (batch_size, max_new_tokens), float("-inf"), dtype=logits.dtype, device=device
                )
                full_conf[mask_index[:, prompt_length:]] = confidence

                if alg_temp is None or alg_temp == 0:
                    _, transfer_idx = torch.topk(full_conf, n_transfer)
                else:
                    full_conf = F.softmax(full_conf / alg_temp, dim=-1)
                    transfer_idx = torch.multinomial(full_conf, num_samples=n_transfer)

                x0_full = torch.full(
                    (batch_size, max_new_tokens), mask_token_id, dtype=torch.long, device=device
                )
                x0_full[mask_index[:, prompt_length:]] = x0

                row_idx = torch.arange(batch_size, device=device).unsqueeze(1).expand_as(transfer_idx)
                x[:, prompt_length:][row_idx, transfer_idx] = x0_full[row_idx, transfer_idx]

        # Fire the visualization hook (same signature as Dream's hook)
        if generation_tokens_hook_func is not None:
            x = generation_tokens_hook_func(i, x, logits)

        if output_history:
            histories.append(x.clone())

        if (i + 1) % max(1, steps // 10) == 0 or i == steps - 1:
            remaining = (x[:, prompt_length:] == mask_token_id).sum().item()
            print(f"  Step {i+1}/{steps}: {remaining} mask tokens remaining")

    print("Generation complete!")
    result = {"sequences": x}
    if output_history:
        result["history"] = histories
    return result


def multimodal_batch_generate(
    model,
    tokenizer,
    prompts: List[str],
    images=None,
    videos=None,
    processor=None,
    max_new_tokens: int = 256,
    steps: int = 128,
    temperature: float = 0.2,
    top_p: float = 0.95,
    alg: str = "entropy",
    device: str = "cuda",
) -> List[str]:
    """Generate text for a list of prompts with optional vision inputs."""

    generations = []
    for i, prompt in enumerate(prompts):
        print(f"\n--- Prompt {i+1}/{len(prompts)} ---")

        image = images[i] if images and i < len(images) else None
        video = videos[i] if videos and i < len(videos) else None

        if processor is not None:
            if image is not None:
                inputs = processor.process_images([image], text=prompt)
            elif video is not None:
                inputs = processor.process_video(video, text=prompt)
            else:
                messages = [{"role": "user", "content": prompt}]
                inputs = processor.processor(
                    text=[processor.processor.apply_chat_template(messages, add_generation_prompt=True)],
                    return_tensors="pt",
                    padding=True,
                )
        else:
            inputs = tokenizer(prompt, return_tensors="pt", padding=True)

        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        output = multimodal_diffusion_generate(
            model=model,
            tokenizer=tokenizer,
            pixel_values=inputs.get("pixel_values"),
            pixel_values_videos=inputs.get("pixel_values_videos"),
            image_grid_thw=inputs.get("image_grid_thw"),
            video_grid_thw=inputs.get("video_grid_thw"),
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            steps=steps,
            temperature=temperature,
            top_p=top_p,
            alg=alg,
            device=device,
        )

        prompt_len = inputs["input_ids"].shape[1]
        generated_text = tokenizer.decode(output["sequences"][0, prompt_len:], skip_special_tokens=True)
        generations.append(generated_text)
        print(f"Generated: {generated_text[:100]}...")

    return generations
