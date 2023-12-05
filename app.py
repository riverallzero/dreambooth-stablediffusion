import gradio as gr
from PIL import Image
import json
import subprocess

import torch
from torch import autocast
from diffusers import StableDiffusionPipeline, DDIMScheduler
import shutil

import os


FOLDER_NAME = "suzy"
os.environ["GRADIO_TEMP_DIR"] = f"./tmp/{FOLDER_NAME}"

if os.path.exists(os.environ["GRADIO_TEMP_DIR"]):
    command = "rm -rf ./tmp/suzy/*"
    subprocess.run(command, shell=True)


def dreambooth_stablediffusion(img_path_list, seed, num_samples, height, width, prompt, negative_prompt):
    for i, img_path in enumerate(img_path_list):
        try:
            output_dir = os.path.join(os.environ["GRADIO_TEMP_DIR"], "input")
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)                
                    
            img = Image.open(img_path)
            img = img.resize((512, 512))
            img.save(os.path.join(output_dir, f'{i}.{img_path.split(".")[-1]}'))
            print(f"- Resized {img_path}")
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")

    output_dir = os.path.join(os.environ["GRADIO_TEMP_DIR"], "stable_diffusion_weights/zwx")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    concepts_list = [
        {
            "instance_prompt": f"photo of {FOLDER_NAME} person",
            "class_prompt": "photo of a person",
            "instance_data_dir": os.path.join(os.environ["GRADIO_TEMP_DIR"], "input"),
            "class_data_dir": "./data/person"
        },
    ]

    for c in concepts_list:
        os.makedirs(c["instance_data_dir"], exist_ok=True)

    json_dir = os.path.join(os.environ["GRADIO_TEMP_DIR"],"concepts_list.json")
    with open(json_dir, "w") as f:
        json.dump(concepts_list, f, indent=4)

    EPOCH = 800
    # Training 
    training_command = (
        "python train_dreambooth.py "
        "--pretrained_model_name_or_path=runwayml/stable-diffusion-v1-5 "
        "--pretrained_vae_name_or_path=stabilityai/sd-vae-ft-mse "
        f"--output_dir={output_dir} "
        '--revision="fp16" '
        "--with_prior_preservation "
        "--prior_loss_weight=1.0 "
        "--seed=1337 "
        "--resolution=512 "
        "--train_batch_size=2 "
        "--train_text_encoder "
        '--mixed_precision="fp16" '
        "--use_8bit_adam "
        "--gradient_accumulation_steps=2 "
        "--learning_rate=1e-6 "
        '--lr_scheduler="constant" '
        "--lr_warmup_steps=0 "
        "--num_class_images=50 "
        "--sample_batch_size=1 "
        f"--max_train_steps={EPOCH} "
        "--save_interval=10000 "
        '--save_sample_prompt="photo of zwx dog" '
        f'--concepts_list="{json_dir}"'
    )

    subprocess.run(training_command, shell=True)

    convert_command = (
        "python convert_diffusers_to_original_stable_diffusion.py "
        f'--model_path "{output_dir}/{EPOCH}" '
        f'--checkpoint_path "{output_dir}/{EPOCH}/model.ckpt" '
        "--half"
    )

    subprocess.run(convert_command, shell=True)

    pipe = StableDiffusionPipeline.from_pretrained(f"{output_dir}/{EPOCH}", safety_checker=None, torch_dtype=torch.float16).to("cuda")
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.enable_xformers_memory_efficient_attention()
    g_cuda = torch.Generator(device="cuda")
    seed = int(seed) 
    g_cuda.manual_seed(seed)

    prompt = f"{FOLDER_NAME} person as {prompt}"
    negative_prompt = negative_prompt
    num_samples = int(num_samples)
    guidance_scale = 7.5 
    num_inference_steps = 30
    height = int(height) 
    width = int(width) 

    with autocast("cuda"), torch.inference_mode():
        images = pipe(
            prompt,
            height=height,
            width=width,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_samples,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=g_cuda
        ).images

    output_dir = os.path.join(os.environ["GRADIO_TEMP_DIR"], "output")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    result_image_path = []
    for i, img in enumerate(images):
        filename = f"image_{i}.png"
        result_path = os.path.join(output_dir, filename)

        img.save(result_path)
        result_image_path.append(result_path)

        print(f"- 이미지 저장: {result_path}")

    return result_image_path

with gr.Blocks() as demo:
    with gr.Tab("Dreambooth - StableDiffusion"):
        with gr.Row():
            img_path_list = gr.File(file_types=["filepath"], file_count="multiple", type="filepath")
            with gr.Row():
                with gr.Column(1):
                    seed_input = gr.Textbox(label="seed", info="for generation")
                with gr.Column(2):
                    num_samples_input = gr.Textbox(label="num samples", info="generate image nums")
                with gr.Column(3):
                    height_input = gr.Radio(["256", "512", "720", "1024"], label="height")
                    width_input = gr.Radio(["256", "512", "720", "1024"], label="width")
            with gr.Row():
                with gr.Column(1):
                    prompt_input = gr.Textbox(label="prompt")
                with gr.Column(2):
                    negative_prompt_input = gr.Textbox(label="negative prompt")
                with gr.Column(3):
                    btn = gr.Button()
        with gr.Row():
            output_gallery = gr.Gallery(["./tmp/suzy/output"])

    btn.click(dreambooth_stablediffusion, inputs=[img_path_list, seed_input, num_samples_input, height_input, width_input, prompt_input, negative_prompt_input], outputs=[output_gallery])

demo.launch(debug=True, share=True)
