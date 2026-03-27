import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import torch
import time
import copy
import threading
import traceback
import gradio as gr
from PIL import Image
from transformers import AutoTokenizer

from src.diffllm.hybrid_dream_model import HybridDreamModel
from src.diffllm.qwen_vision_system import Qwen3VLMultimodalProcessor
from src.diffllm.multimodal_generation_utils import multimodal_diffusion_generate


# ---------------------------------------------------------------------------
# Multinomial safety patch (ported from app.py)
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
        samples = _original_multinomial(_sanitize_probabilities(flat), num_samples, replacement, generator=generator, out=out)
        return samples.view(*orig_shape[:-1], num_samples)
    return _original_multinomial(_sanitize_probabilities(probs), num_samples, replacement, generator=generator, out=out)

if not getattr(torch.multinomial, "_dream_safe", False):
    _safe_multinomial._dream_safe = True
    torch.multinomial = _safe_multinomial


# ---------------------------------------------------------------------------
# Device / dtype selection
# ---------------------------------------------------------------------------
def select_device():
    if torch.cuda.is_available():
        return "cuda"
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return "mps"
    return "cpu"


device = select_device()
dtype = {"cuda": torch.bfloat16, "mps": torch.float16, "cpu": torch.float32}[device]
print(f"Using device: {device}  dtype: {dtype}")


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
try:
    model = HybridDreamModel(
        dream_model_name="Dream-org/Dream-v0-Instruct-7B",
        qwen_model_name="Qwen/Qwen2-VL-7B-Instruct",
        device=device,
        dtype=dtype,
    )
except Exception:
    print(traceback.format_exc())
    raise

try:
    processor = Qwen3VLMultimodalProcessor(model_name="Qwen/Qwen2-VL-7B-Instruct")
    # Use Dream's tokenizer for generation (has <|mask|> token)
    tokenizer = AutoTokenizer.from_pretrained("Dream-org/Dream-v0-Instruct-7B", trust_remote_code=True)
    mask_token_id = tokenizer.mask_token_id if tokenizer.mask_token_id is not None else 151666
    mask_token_str = "[MASK]"
    print(f"Tokenizer loaded. mask_token_id={mask_token_id}")
except Exception:
    print(traceback.format_exc())
    raise


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def format_history(history):
    """Convert [[user, assistant], ...] to OpenAI-style messages."""
    msgs = []
    for user_msg, assistant_msg in (history or []):
        msgs.append({"role": "user", "content": user_msg})
        if assistant_msg is not None:
            msgs.append({"role": "assistant", "content": str(assistant_msg)})
    return msgs


def add_user_msg(history, message):
    return (history or []) + [[message, None]]


def highlight(message, color="#CC6666"):
    return [(message, color)]


# ---------------------------------------------------------------------------
# Core: threaded generation with per-step visualization
# ---------------------------------------------------------------------------
def multimodal_generate_with_visualization(
    image, history,
    max_new_tokens, steps, temperature, top_p, top_k, delay, alg, alg_temp,
):
    """
    Generator that yields (chatbot_messages, highlighted_tokens, history) at
    every diffusion step, mirroring app.py's dream_generate_with_visualization.
    """
    print(f"\n--- multimodal_generate_with_visualization ---")
    print(f"  image={'yes' if image is not None else 'no'}  steps={steps}  alg={alg}")

    messages_for_model = format_history(history)

    # ---- Prepare inputs ----
    try:
        if image is not None:
            orig_w, orig_h = image.size
            # Build the last user message with image content using Qwen processor.
            # process_images() auto-resizes to processor.max_pixels (1 MP default).
            user_text = messages_for_model[-1]["content"] if messages_for_model else ""
            image_inputs = processor.process_images([image], text=user_text)
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                      for k, v in image_inputs.items()}
            pv = inputs.get("pixel_values")
            grid = inputs.get("image_grid_thw")
            n_vis = int(grid[0].prod().item()) // 4 if grid is not None else "?"
            print(f"  image orig={orig_w}×{orig_h}  vision_tokens≈{n_vis}  "
                  f"pixel_patches={pv.shape[0] if pv is not None else '?'}")
        else:
            inputs_raw = processor.processor(
                text=[processor.processor.apply_chat_template(messages_for_model, add_generation_prompt=True)],
                return_tensors="pt",
                padding=True,
            )
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                      for k, v in inputs_raw.items()}

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        prompt_length = input_ids.shape[1]
        print(f"  prompt_length={prompt_length}")

    except Exception as e:
        err = f"Input processing error: {e}"
        errhist = copy.deepcopy(history)
        errhist[-1][1] = f"Error: {err}"
        yield format_history(errhist), highlight(err), history
        return

    # ---- Hook: capture token state at every step ----
    visualization_token_states = []

    def hook(step, x, logits):
        visualization_token_states.append(x[0].clone().cpu())
        return x

    effective_top_k = top_k if top_k > 0 else None
    output_container = {}

    def generation_func():
        try:
            with torch.no_grad():
                out = multimodal_diffusion_generate(
                    model=model,
                    tokenizer=tokenizer,
                    pixel_values=inputs.get("pixel_values"),
                    image_grid_thw=inputs.get("image_grid_thw"),
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    steps=steps,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=effective_top_k,
                    alg=alg,
                    alg_temp=alg_temp if alg_temp > 0 else None,
                    generation_tokens_hook_func=hook,
                    device=device,
                )
            output_container["output"] = out
        except Exception as e:
            output_container["error"] = e

    gen_thread = threading.Thread(target=generation_func)
    gen_thread.start()

    # Wait for the first step before starting to yield
    while len(visualization_token_states) == 0:
        time.sleep(0.01)

    first_state = visualization_token_states[0]
    gen_length = first_state.shape[0] - prompt_length
    previous_tokens = [mask_token_id] * gen_length
    last_yielded = 0
    intermediate_history = copy.deepcopy(history)

    # ---- Stream per-step visualization ----
    while gen_thread.is_alive() or last_yielded < len(visualization_token_states):
        current_length = len(visualization_token_states)
        while last_yielded < current_length:
            state_tensor = visualization_token_states[last_yielded]
            current_tokens = state_tensor[prompt_length:].tolist()

            colored = []
            for idx, token_id in enumerate(current_tokens):
                if token_id == mask_token_id:
                    colored.append((mask_token_str, "#444444"))
                elif previous_tokens[idx] == mask_token_id:
                    # Newly revealed this step → green
                    colored.append((tokenizer.decode([token_id], skip_special_tokens=True), "#66CC66"))
                else:
                    # Previously revealed → blue
                    colored.append((tokenizer.decode([token_id], skip_special_tokens=True), "#6699CC"))

            previous_tokens = current_tokens
            intermediate_history[-1][1] = f"⏳ Step {last_yielded + 1}/{steps}"
            yield format_history(intermediate_history), colored, history
            last_yielded += 1
        time.sleep(delay)

    gen_thread.join()

    if "error" in output_container:
        err = f"Generation error: {output_container['error']}"
        errhist = copy.deepcopy(history)
        errhist[-1][1] = f"Error: {err}"
        yield format_history(errhist), highlight(err), history
        return

    # ---- Final result ----
    output = output_container["output"]
    final_ids = output["sequences"][0][prompt_length:].tolist()

    colored_final = []
    for token_id in final_ids:
        if token_id == mask_token_id:
            colored_final.append((mask_token_str, "#444444"))
        else:
            colored_final.append((tokenizer.decode([token_id], skip_special_tokens=True), "#6699CC"))

    final_text = tokenizer.decode(final_ids, skip_special_tokens=True,
                                  clean_up_tokenization_spaces=True).strip()
    history[-1][1] = final_text
    yield format_history(history), colored_final, history


# ---- Wrappers ----
def bot_response_generator(image, history, max_new_tokens, steps, temperature, top_p, top_k, delay, alg, alg_temp):
    if not history or history[-1][1] is not None:
        yield format_history(history), [], history
        return
    yield from multimodal_generate_with_visualization(
        image, history, max_new_tokens, steps, temperature, top_p, top_k, delay, alg, alg_temp
    )


def user_message_submitted(message, history):
    if not message or not message.strip():
        return history, format_history(history), ""
    new_history = add_user_msg(history, message)
    return new_history, format_history(new_history), ""


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
css = """
.gradio-container .prose ::selection { background-color: #ACE6FF; }
.gradio-container .prose ::-moz-selection { background-color: #ACE6FF; }
"""

with gr.Blocks(title="Hybrid Dream: Vision + Diffusion Chat") as demo:
    gr.Markdown("# Hybrid Dream — Vision + Diffusion Chat")
    gr.Markdown(
        "Upload an image and chat with the model. Watch the **diffusion denoising** process unfold in real time.\n\n"
        "Models: [Dream-org/Dream-v0-Instruct-7B](https://huggingface.co/Dream-org/Dream-v0-Instruct-7B) "
        "+ [Qwen/Qwen2-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct)"
    )

    chat_history_state = gr.State([])

    with gr.Row():
        # Left column: image + chat
        with gr.Column(scale=3):
            image_input = gr.Image(
                label="Image (optional — upload once, ask multiple questions)",
                type="pil",
            )
            chatbot_display = gr.Chatbot(label="Chat", height=500)
            with gr.Group():
                with gr.Row():
                    user_input_textbox = gr.Textbox(
                        label="Message",
                        placeholder="Ask about the image, or just chat...",
                        scale=4,
                        show_label=False,
                        container=False,
                    )
                    send_button = gr.Button("Send", scale=1, variant="primary")

        # Right column: diffusion visualization
        with gr.Column(scale=2):
            vis_display = gr.HighlightedText(
                label="Diffusion Process Visualization",
                show_legend=True,
                combine_adjacent=False,
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
            max_new_tokens_slider = gr.Slider(16, 512, value=128, step=16, label="Max New Tokens")
            steps_slider = gr.Slider(8, 512, value=64, step=8, label="Diffusion Steps (64 ≈ fast, 128 = quality)")
        with gr.Row():
            temperature_slider = gr.Slider(0.0, 1.0, value=0.3, step=0.05, label="Temperature (0=greedy; 0.2–0.3 best for vision)")
            top_p_slider = gr.Slider(0.0, 1.0, value=0.95, step=0.05, label="Top-p")
        with gr.Row():
            top_k_slider = gr.Slider(0, 100, value=0, step=1, label="Top-k (0 = disabled)")
            delay_slider = gr.Slider(0.0, 0.5, value=0.02, step=0.01, label="Visualization Delay (s)")
        with gr.Row():
            alg_dropdown = gr.Dropdown(
                choices=["origin", "maskgit_plus", "topk_margin", "entropy"],
                value="entropy",
                label="Algorithm",
            )
            alg_temp_slider = gr.Slider(0.0, 1.0, value=0.1, step=0.01, label="Algorithm Temperature")

    clear_button = gr.Button("Clear Chat")

    def clear_all():
        return [], [], "", []

    clear_button.click(fn=clear_all, inputs=[], outputs=[chat_history_state, chatbot_display, user_input_textbox, vis_display], queue=False)

    gen_params = [max_new_tokens_slider, steps_slider, temperature_slider, top_p_slider, top_k_slider, delay_slider, alg_dropdown, alg_temp_slider]

    submit_args = dict(fn=user_message_submitted, inputs=[user_input_textbox, chat_history_state], outputs=[chat_history_state, chatbot_display, user_input_textbox])
    bot_args = dict(fn=bot_response_generator, inputs=[image_input, chat_history_state] + gen_params, outputs=[chatbot_display, vis_display, chat_history_state])

    submit_evt = user_input_textbox.submit(**submit_args)
    submit_evt.then(lambda: [], inputs=None, outputs=[vis_display])
    submit_evt.then(**bot_args)

    send_evt = send_button.click(**submit_args)
    send_evt.then(lambda: [], inputs=None, outputs=[vis_display])
    send_evt.then(**bot_args)


if __name__ == "__main__":
    demo.queue(max_size=10, default_concurrency_limit=1).launch(
        server_name="0.0.0.0", server_port=7860, share=False, debug=True,
        theme=gr.themes.Soft(), css=css,
    )
