"""Gradio Web UI for télos 85M Model Ratio Study Parallel Comparison.

Renders an interactive 5-way side-by-side comparative interface for télos 85M
Masked Diffusion models trained at dataset-to-parameter ratios of 1:1, 1:3, 1:5, 1:10, and 1:17.
"""

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import gradio as gr
from telos.hub.inference import TelosModel

# Checkpoint file dictionary for the 5 ratio study models
RATIO_CHECKPOINTS = {
    "1:1 Ratio (85M tokens)": "checkpoints/phase_b_85m_tpu_ratio_study/checkpoint_ratio_1_1_step_162.pt",
    "1:3 Ratio (255M tokens)": "checkpoints/phase_b_85m_tpu_ratio_study/checkpoint_ratio_1_3_step_486.pt",
    "1:5 Ratio (425M tokens)": "checkpoints/phase_b_85m_tpu_ratio_study/checkpoint_ratio_1_5_step_811.pt",
    "1:10 Ratio (850M tokens)": "checkpoints/phase_b_85m_tpu_ratio_study/checkpoint_ratio_1_10_step_1621.pt",
    "1:17 Ratio (1.44B tokens)": "checkpoints/phase_b_85m_tpu_ratio_study/checkpoint_ratio_1_17_step_2741.pt",
}

# Cache loaded model instances in memory for instantaneous multi-model inference
LOADED_MODELS: dict[str, TelosModel] = {}


def load_all_models() -> dict[str, TelosModel]:
    """Loads and caches all 5 ratio study checkpoints into memory."""
    for label, path_str in RATIO_CHECKPOINTS.items():
        if label not in LOADED_MODELS:
            ckpt_path = Path(path_str)
            if ckpt_path.exists():
                try:
                    print(f"Pre-warming ratio model: {label} ({path_str})...")
                    LOADED_MODELS[label] = TelosModel.from_pretrained(ckpt_path)
                except Exception as e:
                    print(f"Error loading {label} from {path_str}: {e}")
            else:
                print(f"Warning: Checkpoint file for {label} not found at {path_str}")
    return LOADED_MODELS


def load_model_cached(label: str, path_str: str) -> TelosModel | None:
    """Loads and caches a model checkpoint into memory safely."""
    if label in LOADED_MODELS:
        return LOADED_MODELS[label]

    ckpt_path = Path(path_str)
    if not ckpt_path.exists():
        print(f"Warning: Checkpoint file for {label} not found at {path_str}")
        return None

    try:
        print(f"Loading ratio model: {label} ({path_str})...")
        model = TelosModel.from_pretrained(ckpt_path)
        LOADED_MODELS[label] = model
        return model
    except Exception as e:
        print(f"Error loading {label} from {path_str}: {e}")
        return None


def generate_single_completion(
    model: TelosModel,
    prompt: str,
    max_tokens: int,
    num_steps: int,
    temperature: float,
    repetition_penalty: float
) -> str:
    """Helper function to generate code completion for a single model instance."""
    try:
        completion = model.complete(
            prompt=prompt,
            max_tokens=int(max_tokens),
            num_steps=int(num_steps),
            temperature=float(temperature),
            repetition_penalty=float(repetition_penalty)
        )
        return completion
    except Exception as e:
        return f"# Error during generation: {e}"


def predict_all_ratios_stream(
    prompt: str,
    max_tokens: int,
    num_steps: int,
    temperature: float,
    repetition_penalty: float
):
    """Executes thread-safe sequential generation across all 5 ratio study models, streaming progress to UI."""
    labels = list(RATIO_CHECKPOINTS.keys())
    current_outputs = ["⏳ Waiting in queue...", "⏳ Waiting in queue...", "⏳ Waiting in queue...", "⏳ Waiting in queue...", "⏳ Waiting in queue..."]

    for idx, label in enumerate(labels):
        current_outputs[idx] = "⏳ Model loading / generating..."
        yield tuple(current_outputs)

        path_str = RATIO_CHECKPOINTS[label]
        model = load_model_cached(label, path_str)
        if model is None:
            current_outputs[idx] = f"# Error: Checkpoint file not found at '{path_str}'"
        else:
            completion = generate_single_completion(
                model, prompt, max_tokens, num_steps, temperature, repetition_penalty
            )
            current_outputs[idx] = completion

        yield tuple(current_outputs)


def build_ratio_comparison_app():
    """Constructs the Gradio 5-Way Ratio Study Comparison Web App."""

    # Custom dark glassmorphism CSS styling
    custom_css = """
    .main-title { text-align: center; margin-bottom: 0.5rem; }
    .subtitle { text-align: center; color: #888; margin-bottom: 1.5rem; }
    .ratio-card { border: 1px solid #333; border-radius: 8px; padding: 10px; background-color: #1a1a24; }
    """

    with gr.Blocks(title="τέλος — 85M Ratio Study Comparison") as demo:
        gr.Markdown(
            """
            # télos (τέλος) — 85M Model Ratio Study Comparison
            ### *Simultaneous 5-Way Evaluation across Dataset-to-Parameter Ratios (1:1, 1:3, 1:5, 1:10, 1:17)*
            *Trained on Google Cloud TPU v6e-1 using Discrete Masked Diffusion with Full Bidirectional Self-Attention.*
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                prompt_input = gr.Code(
                    value="def is_prime(n: int) -> bool:\n    \"\"\"Check if an integer is a prime number.\"\"\"\n",
                    language="python",
                    label="Input Code Prompt (Function Signature + Docstring)",
                    lines=6
                )
                with gr.Row():
                    max_tokens_slider = gr.Slider(
                        minimum=32, maximum=512, value=128, step=32,
                        label="Max Target Tokens"
                    )
                    num_steps_slider = gr.Slider(
                        minimum=16, maximum=128, value=64, step=8,
                        label="Denoising Steps"
                    )
                with gr.Row():
                    temp_slider = gr.Slider(
                        minimum=0.0, maximum=1.5, value=0.3, step=0.05,
                        label="Sampling Temperature"
                    )
                    rep_penalty_slider = gr.Slider(
                        minimum=1.0, maximum=2.0, value=1.2, step=0.05,
                        label="Repetition Penalty"
                    )

                submit_btn = gr.Button("🚀 Generate 5-Model Comparison", variant="primary", size="lg")

        gr.Markdown("---")
        gr.Markdown("### 📊 Live Model Outputs Across Ratio Milestones")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("**1:1 Ratio** *(85M Tokens, Step 162, CE 6.71)*")
                out_1_1 = gr.Code(language="python", label="1:1 Ratio (Step 162)", lines=14)

            with gr.Column(scale=1):
                gr.Markdown("**1:3 Ratio** *(255M Tokens, Step 486, CE 6.27)*")
                out_1_3 = gr.Code(language="python", label="1:3 Ratio (Step 486)", lines=14)

            with gr.Column(scale=1):
                gr.Markdown("**1:5 Ratio** *(425M Tokens, Step 811, CE 6.28)*")
                out_1_5 = gr.Code(language="python", label="1:5 Ratio (Step 811)", lines=14)

            with gr.Column(scale=1):
                gr.Markdown("**1:10 Ratio** *(850M Tokens, Step 1621, CE 5.43)*")
                out_1_10 = gr.Code(language="python", label="1:10 Ratio (Step 1621)", lines=14)

            with gr.Column(scale=1):
                gr.Markdown("**1:17 Ratio** *(1.44B Tokens, Step 2741, CE 4.76)*")
                out_1_17 = gr.Code(language="python", label="1:17 Ratio (Step 2741)", lines=14)

        submit_btn.click(
            fn=predict_all_ratios_stream,
            inputs=[prompt_input, max_tokens_slider, num_steps_slider, temp_slider, rep_penalty_slider],
            outputs=[out_1_1, out_1_3, out_1_5, out_1_10, out_1_17]
        )

    return demo


if __name__ == "__main__":
    app = build_ratio_comparison_app()
    app.launch()
