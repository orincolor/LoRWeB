# Copyright (C) 2026 NVIDIA Corporation.  All rights reserved.
#
# This work is licensed under the LICENSE file
# located at the root directory.

import re
import argparse
import os
import yaml
from PIL import Image
import torch
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor, Siglip2ImageProcessor, Siglip2VisionModel
from extensions_built_in.flux_kontext import CustomFluxKontextPipeline, CustomFluxTransformer2DModel


def process_control_image_with_replace(tl_path, tr_path, bl_path, max_size=None):
    tl_img = Image.open(tl_path)
    tl_img = tl_img.convert("RGB")
    tr_img = Image.open(tr_path)
    tr_img = tr_img.convert("RGB")
    bl_img = Image.open(bl_path)
    bl_img = bl_img.convert("RGB")

    w, h = (tl_img.size[0], tl_img.size[1])
    # Create 2x2 grid with target part masked out (black)
    # Top row: Control part 1, Control part 2
    top_row = Image.new('RGB', (2 * w, h))
    top_row.paste(tl_img.resize((w, h), Image.BICUBIC), (0, 0))
    top_row.paste(tr_img.resize((w, h), Image.BICUBIC), (w, 0))

    # Bottom row: Control part 3, Black (masked target)
    bottom_row = Image.new('RGB', (2 * w, h))
    bottom_row.paste(bl_img.resize((w, h), Image.BICUBIC), (0, 0))
    bottom_row.paste(bl_img.resize((w, h), Image.BICUBIC), (w, 0))

    # Combine rows
    control_img = Image.new('RGB', (2 * w, 2 * h))
    control_img.paste(top_row, (0, 0))
    control_img.paste(bottom_row, (0, h))

    if max_size is not None and (2*w > max_size or 2*h > max_size):
        print('need to resize')
        #long edge resize to fit max_size
        if w > h:
            print('resizing to ', (max_size, int(h * max_size / w)))
            control_img = control_img.resize((max_size, int(h * max_size / w)), Image.BICUBIC)
        else:
            print('resizing to ', (int(w * max_size / h), max_size))
            control_img = control_img.resize((int(w * max_size / h), max_size), Image.BICUBIC)

        print(control_img.size)
    return control_img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--lora_weight_file", type=str, required=True)
    parser.add_argument("-c", "--config_file", type=str, required=True)
    parser.add_argument("-a","--a_path", type=str, required=True)
    parser.add_argument("-t","--atag_path", type=str, required=True)
    parser.add_argument("-b","--b_path", type=str, required=True)
    parser.add_argument("-p", "--prompt", type=str, required=True)
    parser.add_argument("-o","--output_path", type=str, required=True)

    parser.add_argument("-d", "--device", type=str, default="cuda:0")
    parser.add_argument("-m", "--max_size", type=int, default=None)
    parser.add_argument("-g", "--guidance_scale", type=float, default=2.5)
    parser.add_argument("-s", "--seed", type=int, default=50)
    args = parser.parse_args()

    print(args.lora_weight_file)

    # load the params from the yaml file
    with open(args.config_file, 'r') as f:
        config = yaml.safe_load(f)['config']

    external_query = True
    heads = config['process'][0]['network']['network_kwargs']['lora_heads']
    lora_softmax = config['process'][0]['network']['network_kwargs']['lora_softmax']
    mixing_coeffs_type = config['process'][0]['network']['network_kwargs']['mixing_coeffs_type']
    query_pooling = config['process'][0]['network']['network_kwargs']['pooling_type']
    query_projection_type = config['process'][0]['network']['network_kwargs']['query_projection_type']
    query_mode = config['process'][0]['network']['network_kwargs']['query_mode']

    image_encoder = None
    feature_extractor = None
    # load the external query model
    if config['process'][0]['network']['network_kwargs']['external_query_model'] == 'openai/clip-vit-large-patch14':
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            "openai/clip-vit-large-patch14", ignore_mismatched_sizes=True).to(args.device, dtype=torch.bfloat16)
        feature_extractor = CLIPImageProcessor.from_pretrained(
            "openai/clip-vit-large-patch14",
            size=image_encoder.config.image_size,
            crop_size=image_encoder.config.image_size
        )
    elif config['process'][0]['network']['network_kwargs']['external_query_model'] == 'google/siglip2-base-patch16-224':
        image_encoder = Siglip2VisionModel.from_pretrained("google/siglip2-base-patch16-224", ignore_mismatched_sizes=True).to(args.device, dtype=torch.bfloat16)
        feature_extractor = Siglip2ImageProcessor.from_pretrained("google/siglip2-base-patch16-224")

    transformer = CustomFluxTransformer2DModel.from_pretrained("black-forest-labs/FLUX.1-Kontext-dev", subfolder='transformer', torch_dtype=torch.bfloat16)
    pipe = CustomFluxKontextPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-Kontext-dev",
        torch_dtype=torch.bfloat16,
        transformer=transformer,
        image_encoder=image_encoder,
        feature_extractor=feature_extractor,
        external_query=external_query,
        query_mode=query_mode)

    if os.path.isdir(args.lora_weight_file):
        proposals = [os.path.join(args.lora_weight_file, x) for x in os.listdir(args.lora_weight_file) if x.endswith('.safetensors')]
        # if theres xxxx.safetensors in the options, use it. Else check for xxxx_00000yyyy.safetensors and use the highest yyyy.
        best_prop = None
        best_checkpoint_num = None
        for prop in proposals:
            res = re.search(r'_(000\d+)\.safetensors', prop)
            if res is not None:
                checkpoint_num = int(res.group(1))
                if best_prop is None or checkpoint_num > best_checkpoint_num:
                    best_prop = prop
                    best_checkpoint_num = checkpoint_num
            else: # this is the last checkpoint
                best_prop = prop
                break
        args.lora_weight_file = best_prop
        print(f"Using checkpoint {best_prop}")
        if best_checkpoint_num is not None:
            output_path = output_path+f"_{best_checkpoint_num}"

    pipe.load_lora_weights(args.lora_weight_file,
                           lora_softmax=lora_softmax,
                           mixing_coeffs_type=mixing_coeffs_type,
                           query_mode=query_mode,
                           query_projection_type=query_projection_type,
                           query_pooling=query_pooling,
                           external_query=external_query,
                           heads=heads)
    pipe.to(args.device)

    output_dir = os.path.dirname(args.output_path)
    os.makedirs(output_dir, exist_ok=True)

    input_image = process_control_image_with_replace(
        tl_path=args.a_path, tr_path=args.atag_path, bl_path=args.b_path,
        max_size=args.max_size,
    )

    generator = torch.Generator(args.device).manual_seed(args.seed)
    image = pipe(
    image=input_image,
    prompt=args.prompt,
    guidance_scale=args.guidance_scale,
    generator=generator,
    height=input_image.height,
    width=input_image.width,
    ).images[0]

    # crop the bottom left quadrant according to the image sizes# crop the bottom left quadrant according to the image sizes
    image = image.crop((image.width // 2, image.height // 2, image.width, image.height))
    # Save the generated image
    image.save(args.output_path)

if __name__ == "__main__":
    main()
