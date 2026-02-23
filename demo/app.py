"""Gradio demo for speculative decoding side-by-side comparison.

Launch:
    python3 demo/app.py

Requires: pip install gradio>=4.0
"""

import time

import torch

_target_backend = None
_draft_backend = None
_tokenizer = None
_loaded_models = (None, None)


def _ensure_models(target_name: str, draft_name: str):
    """Lazy-load models on first inference (or when selection changes)."""
    global _target_backend, _draft_backend, _tokenizer, _loaded_models

    if _loaded_models == (target_name, draft_name):
        return

    from transformers import AutoTokenizer

    from src.speculative.backends import create_backend

    print(f"Loading target: {target_name}")
    _target_backend = create_backend(target_name, dtype="float16", device="auto")

    print(f"Loading draft: {draft_name}")
    _draft_backend = create_backend(draft_name, dtype="float16", device="auto")

    _tokenizer = AutoTokenizer.from_pretrained(target_name)
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token

    _loaded_models = (target_name, draft_name)
    print("Models loaded.")


def run_comparison(
    prompt: str,
    target_model: str,
    draft_model: str,
    temperature: float,
    max_tokens: int,
    speculation_k: int,
):
    """Run both standard and speculative decoding, return results."""
    if not prompt.strip():
        return "Please enter a prompt.", "", "", "", "", ""

    _ensure_models(target_model, draft_model)

    from src.speculative.decoding import speculative_decode, standard_decode

    device = next(_target_backend.model.parameters()).device
    input_ids = _tokenizer.encode(prompt, return_tensors="pt").to(device)

    # Standard decoding
    std_output, std_metrics = standard_decode(
        model=_target_backend,
        input_ids=input_ids,
        max_new_tokens=max_tokens,
        temperature=temperature,
    )
    std_text = _tokenizer.decode(std_output[0, input_ids.shape[1]:], skip_special_tokens=True)

    # Speculative decoding
    spec_output, spec_metrics = speculative_decode(
        target_model=_target_backend,
        draft_model=_draft_backend,
        input_ids=input_ids,
        max_new_tokens=max_tokens,
        speculation_length=speculation_k,
        temperature=temperature,
        profile=True,
    )
    spec_text = _tokenizer.decode(spec_output[0, input_ids.shape[1]:], skip_special_tokens=True)

    # Metrics
    std_info = (
        f"Tokens: {std_metrics.total_tokens}\n"
        f"Latency: {std_metrics.latency_seconds:.3f}s\n"
        f"Tokens/sec: {std_metrics.tokens_per_second:.1f}"
    )

    spec_info = (
        f"Tokens: {spec_metrics.total_tokens}\n"
        f"Latency: {spec_metrics.latency_seconds:.3f}s\n"
        f"Tokens/sec: {spec_metrics.tokens_per_second:.1f}\n"
        f"Acceptance rate: {spec_metrics.acceptance_rate:.1%}"
    )

    # Speedup summary
    speedup = (
        spec_metrics.tokens_per_second / std_metrics.tokens_per_second
        if std_metrics.tokens_per_second > 0
        else 0
    )
    summary = (
        f"Speedup: {speedup:.2f}x\n"
        f"Acceptance rate: {spec_metrics.acceptance_rate:.1%}\n"
        f"Standard: {std_metrics.tokens_per_second:.1f} tok/s\n"
        f"Speculative: {spec_metrics.tokens_per_second:.1f} tok/s"
    )

    return std_text, spec_text, std_info, spec_info, summary


def create_demo():
    """Build and return the Gradio Blocks interface."""
    import gradio as gr

    with gr.Blocks(title="SpecDecode - Speculative Decoding Demo") as demo:
        gr.Markdown("# SpecDecode: Speculative Decoding Demo")
        gr.Markdown("Compare standard autoregressive decoding with speculative decoding side-by-side.")

        with gr.Row():
            with gr.Column(scale=2):
                prompt = gr.Textbox(
                    label="Prompt",
                    placeholder="Enter your prompt here...",
                    lines=3,
                    value="Write a Python function to compute the fibonacci sequence:",
                )
            with gr.Column(scale=1):
                target_model = gr.Dropdown(
                    label="Target Model",
                    choices=[
                        "Qwen/Qwen2.5-7B",
                        "Qwen/Qwen2.5-3B",
                        "gpt2-medium",
                        "gpt2-large",
                        "gpt2-xl",
                        "gpt2",
                    ],
                    value="gpt2-medium",
                    allow_custom_value=True,
                )
                draft_model = gr.Dropdown(
                    label="Draft Model",
                    choices=[
                        "Qwen/Qwen2.5-0.5B",
                        "distilgpt2",
                        "gpt2",
                    ],
                    value="distilgpt2",
                    allow_custom_value=True,
                )
                gr.Markdown(
                    "*Target and draft must share the same tokenizer "
                    "(e.g., GPT-2 family together, Qwen family together).*"
                )

        with gr.Row():
            temperature = gr.Slider(
                minimum=0.0, maximum=2.0, value=1.0, step=0.1,
                label="Temperature",
            )
            max_tokens = gr.Slider(
                minimum=16, maximum=512, value=128, step=16,
                label="Max Tokens",
            )
            speculation_k = gr.Slider(
                minimum=1, maximum=10, value=5, step=1,
                label="Speculation Length (K)",
            )

        run_btn = gr.Button("Generate", variant="primary")

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Standard Decoding")
                std_output = gr.Textbox(label="Output", lines=10)
                std_metrics = gr.Textbox(label="Metrics", lines=3)
            with gr.Column():
                gr.Markdown("### Speculative Decoding")
                spec_output = gr.Textbox(label="Output", lines=10)
                spec_metrics = gr.Textbox(label="Metrics", lines=4)

        with gr.Row():
            summary = gr.Textbox(label="Comparison Summary", lines=4)

        run_btn.click(
            fn=run_comparison,
            inputs=[prompt, target_model, draft_model, temperature, max_tokens, speculation_k],
            outputs=[std_output, spec_output, std_metrics, spec_metrics, summary],
        )

    return demo


if __name__ == "__main__":
    demo = create_demo()
    demo.launch(server_name="0.0.0.0", server_port=7860)
