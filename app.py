import gradio as gr
import modin.pandas as pd
import torch
import numpy as np
from PIL import Image
from diffusers import DiffusionPipeline
from huggingface_hub import login
import os

login(token=os.environ.get('HF_KEY'))

device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16) if torch.cuda.is_available() else DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-refiner-1.0")
pipe = pipe.to(device)

def resize(value,img):
    img = Image.open(img)
    img = img.resize((value,value))
    return img

def infer(source_img, prompt, negative_prompt, guide, steps, seed, Strength):
    generator = torch.Generator(device).manual_seed(seed)     
    source_image = resize(768, source_img)
    source_image.save('source.png')
    image = pipe(prompt, negative_prompt=negative_prompt, image=source_image, strength=Strength, guidance_scale=guide, num_inference_steps=steps).images[0]
    return image

gr.Interface(fn=infer, inputs=[gr.Image(source="upload", type="filepath", label="Raw Image. Must Be .png"), gr.Textbox(label = 'Prompt Input Text. 77 Token (Keyword or Symbol) Maximum'), gr.Textbox(label='What you Do Not want the AI to generate.'),
    gr.Slider(2, 15, value = 7, label = 'Guidance Scale'),
    gr.Slider(1, 25, value = 10, step = 1, label = 'Number of Iterations'),
    gr.Slider(label = "Seed", minimum = 0, maximum = 987654321987654321, step = 1, randomize = True), 
    gr.Slider(label='Strength', minimum = 0, maximum = 1, step = .05, value = .5)], 
    outputs='image', title = "Stable Diffusion XL 1.0 Image to Image Pipeline CPU", description = "For more information on Stable Diffusion XL 1.0 see https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0 <br><br>Upload an Image (<b>MUST Be .PNG and 512x512 or 768x768</b>) enter a Prompt, or let it just do its Thing, then click submit. 10 Iterations takes about ~900-1200 seconds currently. For more informationon about Stable Diffusion or Suggestions for prompts, keywords, artists or styles see https://github.com/Maks-s/sd-akashic", article = "Code Monkey: <a href=\"https://huggingface.co/Manjushri\">Manjushri</a>").queue(max_size=5).launch()
