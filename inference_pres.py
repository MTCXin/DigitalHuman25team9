from diffusers import StableDiffusionPipeline
import torch
import os

method = "more_img"
model_id = "/home/jixiao/.cache/huggingface/dh/cat/" + method
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")


save_dir = "pres_cat"
prompt = 'a photo of cat'


if not os.path.exists(save_dir + "/" + method):
    os.makedirs(save_dir + "/" + method)

for i in range(100):
    image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]
    image.save(f"{save_dir}/{method}/{method}_{i}.png")
