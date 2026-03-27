import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import time
import copy
import threading
import traceback

import torch
import gradio as gr

from model import VisionDream, _generate


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
device = (
    "cuda" if torch.cuda.is_available() else
    "mps"  if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else
    "cpu"
)
print(f"Device: {device}")

try:
    model = VisionDream(device=device)
    tokenizer = model.tokenizer
    mask_token_id  = model.mask_token_id
    mask_token_str = "[MASK]"
except Exception:
    print(traceback.format_exc())
    raise


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _to_messages(history):
    msgs = []
    for user_msg, assistant_msg in (history or []):
        msgs.append({"role": "user", "content": user_msg})
        if assistant_msg is not None:
            msgs.append({"role": "assistant", "content": str(assistant_msg)})
    return msgs


def _add_user(history, message):
    return (history or []) + [[message, None]]


def _highlight(msg, color="#CC6666"):
    return [(msg, color)]


# ---------------------------------------------------------------------------
# Generation with per-step visualization
# ---------------------------------------------------------------------------
def _dream_generate_vis(image, history, max_new_tokens, steps, temperature, top_p, top_k, delay, alg, alg_temp):
    messages = _to_messages(history)

    # Preprocess
    try:
        images = [image] if image is not None else None
        user_text = messages[-1]["content"] if messages else ""
        # Use model.preprocess to get input_ids (same path as model.generate)
        inputs = model.preprocess(user_text, images)
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        input_ids     = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")
        prompt_length = input_ids.shape[1]
    except Exception as e:
        err = f"Input error: {e}"
        h = copy.deepcopy(history); h[-1][1] = err
        yield _to_messages(h), _highlight(err), history
        return

    # Encode vision once
    vision_embeds = None
    if "pixel_values" in inputs:
        try:
            vision_embeds = model.encode_vision(inputs["pixel_values"], inputs["image_grid_thw"])
        except Exception as e:
            err = f"Vision encode error: {e}"
            h = copy.deepcopy(history); h[-1][1] = err
            yield _to_messages(h), _highlight(err), history
            return

    # Per-step hook
    states = []
    def hook(step, x, logits):
        states.append(x[0].clone().cpu())
        return x

    effective_top_k = top_k if top_k > 0 else None
    container = {}

    def gen_func():
        try:
            # Use Dream's native diffusion_generate for text-only (preserves hook support),
            # fall back to our _generate for vision inputs
            if vision_embeds is None:
                out = model.dream.diffusion_generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    output_history=False,
                    return_dict_in_generate=True,
                    steps=steps,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=effective_top_k,
                    alg=alg,
                    alg_temp=alg_temp,
                    generation_tokens_hook_func=hook,
                )
                container["sequences"] = out.sequences
            else:
                # Vision path: wrap hook into our generation loop
                def _hook_wrapper(i, x, logits):
                    states.append(x[0].clone().cpu())
                    return x
                from model import _generate as _gen
                import types
                # Monkey-patch hook into _generate via a custom call
                container["sequences"] = _generate_with_hook(
                    model, input_ids, attention_mask, vision_embeds,
                    max_new_tokens, steps, temperature, top_p, alg,
                    hook_func=_hook_wrapper,
                )
        except Exception as e:
            container["error"] = e

    gen_thread = threading.Thread(target=gen_func)
    gen_thread.start()

    while len(states) == 0:
        time.sleep(0.01)

    gen_length = states[0].shape[0] - prompt_length
    prev_tokens = [mask_token_id] * gen_length
    last_yielded = 0
    inter_history = copy.deepcopy(history)

    while gen_thread.is_alive() or last_yielded < len(states):
        cur_len = len(states)
        while last_yielded < cur_len:
            current = states[last_yielded][prompt_length:].tolist()
            colored = []
            for idx, tid in enumerate(current):
                if tid == mask_token_id:
                    colored.append((mask_token_str, "#444444"))
                elif prev_tokens[idx] == mask_token_id:
                    colored.append((tokenizer.decode([tid], skip_special_tokens=True), "#66CC66"))
                else:
                    colored.append((tokenizer.decode([tid], skip_special_tokens=True), "#6699CC"))
            prev_tokens = current
            inter_history[-1][1] = f"⏳ Step {last_yielded + 1}/{steps}"
            yield _to_messages(inter_history), colored, history
            last_yielded += 1
        time.sleep(delay)

    gen_thread.join()

    if "error" in container:
        err = f"Generation error: {container['error']}"
        h = copy.deepcopy(history); h[-1][1] = err
        yield _to_messages(h), _highlight(err), history
        return

    seqs = container["sequences"]
    final_ids = seqs[0][prompt_length:].tolist()
    colored_final = [
        (mask_token_str, "#444444") if tid == mask_token_id
        else (tokenizer.decode([tid], skip_special_tokens=True), "#6699CC")
        for tid in final_ids
    ]
    final_text = tokenizer.decode(final_ids, skip_special_tokens=True,
                                  clean_up_tokenization_spaces=True).strip()
    history[-1][1] = final_text
    yield _to_messages(history), colored_final, history


def _generate_with_hook(model, input_ids, attention_mask, vision_embeds,
                        max_new_tokens, steps, temperature, top_p, alg,
                        hook_func=None, eps=1e-3):
    """_generate() with an optional per-step hook for visualization."""
    import torch.nn.functional as F
    device = model.device
    B, L = input_ids.shape
    mask_id = model.mask_token_id

    x = torch.cat([input_ids,
                   torch.full((B, max_new_tokens), mask_id, dtype=torch.long, device=device)], dim=1)

    has_pad = attention_mask is not None and attention_mask.eq(0).any()
    if has_pad:
        gm = torch.ones(B, max_new_tokens, dtype=attention_mask.dtype, device=device)
        full_mask = torch.cat([attention_mask, gm], dim=1)
        tok_idx = full_mask.long().cumsum(-1) - 1
        tok_idx.masked_fill_(full_mask == 0, 1)
        attn = torch.logical_and(full_mask[:, None, None, :], full_mask[:, None, :, None])
        pos = tok_idx
    else:
        attn = pos = None

    from model import _sample
    timesteps = torch.linspace(1.0, eps, steps + 1, device=device)

    for i in range(steps):
        mask_idx = x == mask_id
        logits = model(input_ids=x, vision_embeds=vision_embeds, attention_mask=attn, position_ids=pos)
        logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)

        gen_logits = logits[:, L:]
        gen_mask   = mask_idx[:, L:]
        masked_logits = gen_logits[gen_mask]

        t, s = timesteps[i], timesteps[i + 1]
        conf, x0 = _sample(masked_logits, temperature, top_p,
                           margin=(alg == "topk_margin"), neg_entropy=(alg == "entropy"))

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

        if hook_func is not None:
            x = hook_func(i, x, logits)

    return x


# ---------------------------------------------------------------------------
# Gradio wrappers
# ---------------------------------------------------------------------------
def bot_response(image, history, max_new_tokens, steps, temperature, top_p, top_k, delay, alg, alg_temp):
    if not history or history[-1][1] is not None:
        yield _to_messages(history), [], history
        return
    yield from _dream_generate_vis(
        image, history, max_new_tokens, steps, temperature, top_p, top_k, delay, alg, alg_temp
    )


def user_submitted(message, history):
    if not message or not message.strip():
        return history, _to_messages(history), ""
    h = _add_user(history, message)
    return h, _to_messages(h), ""


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
css = """
.gradio-container .prose ::selection { background-color: #ACE6FF; }
.gradio-container .prose ::-moz-selection { background-color: #ACE6FF; }
"""

with gr.Blocks(css=css, theme=gr.themes.Soft(), title="VisionDream") as demo:
    gr.Markdown("# VisionDream — Diffusion LM with Vision")
    gr.Markdown(
        "Chat with **Dream-v0-Instruct-7B** enhanced with **Qwen2-VL** vision. "
        "Upload an image to ask about it, or chat text-only. "
        "Watch tokens emerge through the diffusion process in real time."
    )

    chat_history = gr.State([])

    with gr.Row():
        with gr.Column(scale=3):
            image_input = gr.Image(label="Image (optional)", type="pil")
            chatbot = gr.Chatbot(label="Chat", height=500, type="messages")
            with gr.Group():
                with gr.Row():
                    textbox = gr.Textbox(
                        placeholder="Ask about the image or just chat...",
                        scale=4, show_label=False, container=False,
                    )
                    send_btn = gr.Button("Send", scale=1, variant="primary")

        with gr.Column(scale=2):
            vis = gr.HighlightedText(
                label="Diffusion Visualization",
                show_legend=True, combine_adjacent=False,
                color_map={"[MASK]": "#444444"},
            )
            gr.Markdown(
                "**Legend:** "
                "<span style='color:#66CC66'>■</span> Newly revealed &nbsp;"
                "<span style='color:#6699CC'>■</span> Stable &nbsp;"
                "<span style='background:#444444;color:white;padding:0 4px'>[MASK]</span> Pending"
            )

    with gr.Accordion("Generation Parameters", open=False):
        with gr.Row():
            max_new_tokens = gr.Slider(16, 512, value=128, step=16, label="Max New Tokens")
            steps          = gr.Slider(8, 512, value=64,  step=8,  label="Diffusion Steps")
        with gr.Row():
            temperature = gr.Slider(0.0, 1.0, value=0.3,  step=0.05, label="Temperature (0.2–0.3 best for vision)")
            top_p       = gr.Slider(0.0, 1.0, value=0.95, step=0.05, label="Top-p")
        with gr.Row():
            top_k = gr.Slider(0, 100, value=0,    step=1,    label="Top-k (0=off)")
            delay = gr.Slider(0.0, 0.5, value=0.02, step=0.01, label="Visualization Delay (s)")
        with gr.Row():
            alg      = gr.Dropdown(["origin", "maskgit_plus", "topk_margin", "entropy"], value="entropy", label="Algorithm")
            alg_temp = gr.Slider(0.0, 1.0, value=0.1, step=0.01, label="Algorithm Temperature")

    clear_btn = gr.Button("Clear")
    clear_btn.click(fn=lambda: ([], [], "", []), inputs=[],
                    outputs=[chat_history, chatbot, textbox, vis], queue=False)

    gen_params = [max_new_tokens, steps, temperature, top_p, top_k, delay, alg, alg_temp]
    submit_kw  = dict(fn=user_submitted, inputs=[textbox, chat_history],
                      outputs=[chat_history, chatbot, textbox])
    bot_kw     = dict(fn=bot_response, inputs=[image_input, chat_history] + gen_params,
                      outputs=[chatbot, vis, chat_history])

    for evt in [textbox.submit(**submit_kw), send_btn.click(**submit_kw)]:
        evt.then(lambda: [], outputs=[vis])
        evt.then(**bot_kw)


if __name__ == "__main__":
    demo.queue(max_size=10, default_concurrency_limit=1).launch(
        server_name="0.0.0.0", server_port=7860, share=False, debug=True,
    )
