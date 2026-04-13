# Copyright (C) 2026 NVIDIA Corporation.  All rights reserved.
# Gradio web UI for LoRWeB inference.
# Usage: python app.py -w /workspace/lorweb_model -c /workspace/lorweb_model/config.yaml

import argparse
import os
import sys
import tempfile
import types

# pkg_resources is part of setuptools but not always registered in venvs.
# clip-anytorch needs it only for `packaging`, which is available standalone.
if "pkg_resources" not in sys.modules:
    try:
        import pkg_resources  # noqa: F401
    except ModuleNotFoundError:
        import packaging as _packaging
        _pr = types.ModuleType("pkg_resources")
        _pr.packaging = _packaging
        sys.modules["pkg_resources"] = _pr

import gradio as gr
import torch
import yaml
from PIL import Image
from transformers import (
    CLIPImageProcessor,
    CLIPVisionModelWithProjection,
    Siglip2ImageProcessor,
    Siglip2VisionModel,
)

from extensions_built_in.flux_kontext import (
    CustomFluxKontextPipeline,
    CustomFluxTransformer2DModel,
)


# ── Model loading ─────────────────────────────────────────────────────────────

def load_pipeline(lora_weight_file: str, config_file: str, device: str):
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)["config"]

    net_kwargs = config["process"][0]["network"]["network_kwargs"]
    heads = net_kwargs["lora_heads"]
    lora_softmax = net_kwargs["lora_softmax"]
    mixing_coeffs_type = net_kwargs["mixing_coeffs_type"]
    query_pooling = net_kwargs["pooling_type"]
    query_projection_type = net_kwargs["query_projection_type"]
    query_mode = net_kwargs["query_mode"]
    external_query_model = net_kwargs["external_query_model"]

    image_encoder = None
    feature_extractor = None

    if external_query_model == "openai/clip-vit-large-patch14":
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            "openai/clip-vit-large-patch14", ignore_mismatched_sizes=True
        ).to(device, dtype=torch.bfloat16)
        feature_extractor = CLIPImageProcessor.from_pretrained(
            "openai/clip-vit-large-patch14",
            size=image_encoder.config.image_size,
            crop_size=image_encoder.config.image_size,
        )
    elif external_query_model == "google/siglip2-base-patch16-224":
        image_encoder = Siglip2VisionModel.from_pretrained(
            "google/siglip2-base-patch16-224", ignore_mismatched_sizes=True
        ).to(device, dtype=torch.bfloat16)
        feature_extractor = Siglip2ImageProcessor.from_pretrained(
            "google/siglip2-base-patch16-224"
        )

    transformer = CustomFluxTransformer2DModel.from_pretrained(
        "black-forest-labs/FLUX.1-Kontext-dev",
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
    )
    pipe = CustomFluxKontextPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-Kontext-dev",
        torch_dtype=torch.bfloat16,
        transformer=transformer,
        image_encoder=image_encoder,
        feature_extractor=feature_extractor,
        external_query=True,
        query_mode=query_mode,
    )

    pipe.load_lora_weights(
        lora_weight_file,
        lora_softmax=lora_softmax,
        mixing_coeffs_type=mixing_coeffs_type,
        query_mode=query_mode,
        query_projection_type=query_projection_type,
        query_pooling=query_pooling,
        external_query=True,
        heads=heads,
    )

    pipe.to(device)
    return pipe


# ── Inference ─────────────────────────────────────────────────────────────────

def build_control_image(a: Image.Image, atag: Image.Image, b: Image.Image) -> Image.Image:
    w, h = a.size
    grid = Image.new("RGB", (2 * w, 2 * h))
    grid.paste(a.resize((w, h), Image.BICUBIC),    (0, 0))
    grid.paste(atag.resize((w, h), Image.BICUBIC), (w, 0))
    grid.paste(b.resize((w, h), Image.BICUBIC),    (0, h))
    grid.paste(b.resize((w, h), Image.BICUBIC),    (w, h))  # masked target slot
    return grid


def run_inference(pipe, a_img, atag_img, b_img, prompt, guidance_scale, seed, device):
    a    = Image.fromarray(a_img).convert("RGB")
    atag = Image.fromarray(atag_img).convert("RGB")
    b    = Image.fromarray(b_img).convert("RGB")

    control = build_control_image(a, atag, b)
    generator = torch.Generator(device).manual_seed(seed)

    out = pipe(
        image=control,
        prompt=prompt,
        guidance_scale=guidance_scale,
        generator=generator,
        height=control.height,
        width=control.width,
    ).images[0]

    # Crop bottom-right quadrant (the generated b')
    result = out.crop((out.width // 2, out.height // 2, out.width, out.height))
    return result


# ── Gradio UI ─────────────────────────────────────────────────────────────────

def make_ui(pipe, device):
    def infer(a_img, atag_img, b_img, prompt, guidance_scale, seed):
        if a_img is None or atag_img is None or b_img is None:
            raise gr.Error("Please upload all three images (a, a', b).")
        if not prompt.strip():
            raise gr.Error("Please enter a transformation prompt.")
        result = run_inference(pipe, a_img, atag_img, b_img, prompt, guidance_scale, int(seed), device)
        return result

    with gr.Blocks(title="LoRWeB — Visual Analogy") as demo:
        gr.Markdown(
            """
# LoRWeB — Visual Analogy Image Editing
Upload an analogy triplet **a → a'** and a new image **b**.
LoRWeB will generate **b'** by applying the same transformation to b.

> **a** : original &nbsp;|&nbsp; **a'** : transformed &nbsp;|&nbsp; **b** : new image to transform
            """
        )
        gr.Image(
            value="https://research.nvidia.com/labs/par/lorweb/static/images/teaser.jpg",
            label="How it works",
            interactive=False,
        )

        with gr.Row():
            a_input    = gr.Image(label="a  (original)", type="numpy")
            atag_input = gr.Image(label="a' (transformed)", type="numpy")
            b_input    = gr.Image(label="b  (image to transform)", type="numpy")

        prompt_input = gr.Textbox(
            label="Transformation prompt",
            placeholder="e.g. Give this animal a fantastical set of armor",
            lines=2,
        )

        with gr.Accordion("Advanced settings", open=False):
            guidance = gr.Slider(1.0, 10.0, value=2.5, step=0.5, label="Guidance scale")
            seed     = gr.Number(value=50, label="Seed", precision=0)

        run_btn = gr.Button("Generate b'", variant="primary")
        output  = gr.Image(label="Result (b')", type="pil")

        run_btn.click(
            fn=infer,
            inputs=[a_input, atag_input, b_input, prompt_input, guidance, seed],
            outputs=output,
        )

        gr.Examples(
            examples=[
                [None, None, None, "Give this animal a fantastical set of armor", 2.5, 50],
            ],
            inputs=[a_input, atag_input, b_input, prompt_input, guidance, seed],
            label="Examples (replace None with your images)",
        )

    return demo


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--lora_weight_file", required=True)
    parser.add_argument("-c", "--config_file", required=True)
    parser.add_argument("-d", "--device", default="cuda:0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="Create a public Gradio link")
    args = parser.parse_args()

    print("Loading pipeline (this takes ~1–2 min on first run)...")
    pipe = load_pipeline(args.lora_weight_file, args.config_file, args.device)
    print("Pipeline ready.")

    demo = make_ui(pipe, args.device)
    demo.launch(server_name="0.0.0.0", server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
