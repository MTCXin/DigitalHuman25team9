from diffusers import StableDiffusionPipeline
import torch
import os

method = "baseline"
model_id = "/home/jixiao/.cache/huggingface/dh/cat/" + method
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")

unique_token = "sks"
class_token = "cat"
save_dir = "sks_cat"

prompt_list = [
    'a {0} {1} in the jungle'.format(unique_token, class_token),
    'a {0} {1} in the snow'.format(unique_token, class_token),
    'a {0} {1} on the beach'.format(unique_token, class_token),
    'a {0} {1} on a cobblestone street'.format(unique_token, class_token),
    'a {0} {1} on top of pink fabric'.format(unique_token, class_token),
    'a {0} {1} on top of a wooden floor'.format(unique_token, class_token),
    'a {0} {1} with a city in the background'.format(unique_token, class_token),
    'a {0} {1} with a mountain in the background'.format(unique_token, class_token),
    'a {0} {1} with a blue house in the background'.format(unique_token, class_token),
    'a {0} {1} on top of a purple rug in a forest'.format(unique_token, class_token),
    'a {0} {1} wearing a red hat'.format(unique_token, class_token),
    'a {0} {1} wearing a santa hat'.format(unique_token, class_token),
    'a {0} {1} wearing a rainbow scarf'.format(unique_token, class_token),
    'a {0} {1} wearing a black top hat and a monocle'.format(unique_token, class_token),
    'a {0} {1} in a chef outfit'.format(unique_token, class_token),
    'a {0} {1} in a firefighter outfit'.format(unique_token, class_token),
    'a {0} {1} in a police outfit'.format(unique_token, class_token),
    'a {0} {1} wearing pink glasses'.format(unique_token, class_token),
    'a {0} {1} wearing a yellow shirt'.format(unique_token, class_token),
    'a {0} {1} in a purple wizard outfit'.format(unique_token, class_token),
    'a red {0} {1}'.format(unique_token, class_token),
    'a purple {0} {1}'.format(unique_token, class_token),
    'a shiny {0} {1}'.format(unique_token, class_token),
    'a wet {0} {1}'.format(unique_token, class_token),
    'a cube shaped {0} {1}'.format(unique_token, class_token),
]

if not os.path.exists(save_dir + "/" + method):
    os.makedirs(save_dir + "/" + method)

for idx, prompt in enumerate(prompt_list):
    for i in range(4):
        image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]
        image.save(f"{save_dir}/{method}/{method}_prompt{idx}_{i}.png")
