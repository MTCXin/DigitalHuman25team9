import torch
from diffusers import AnimateDiffPipeline, DDIMScheduler, MotionAdapter
from diffusers.utils import export_to_gif
import os

# Load the motion adapter
adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-4", torch_dtype=torch.float16)
# load SD 1.4 based finetuned model
method = "baseline"
model_id = "/home/jixiao/.cache/huggingface/dh/cat/" + method
pipe = AnimateDiffPipeline.from_pretrained(model_id, motion_adapter=adapter, torch_dtype=torch.float16)
scheduler = DDIMScheduler.from_pretrained(
    model_id,
    subfolder="scheduler",
    clip_sample=False,
    timestep_spacing="linspace",
    beta_schedule="linear",
    steps_offset=1,
)
pipe.scheduler = scheduler

# enable memory savings
pipe.enable_vae_slicing()
pipe.enable_model_cpu_offload()

unique_token = "sks"
class_token = "cat"
save_dir = "sks_cat_animation"

prompt_list = [
    'a {0} {1} running in the jungle'.format(unique_token, class_token),
    'a {0} {1} walking on the beach'.format(unique_token, class_token),
    'a {0} {1} swimming in the pool'.format(unique_token, class_token),
    'a {0} {1} flying with a city in the background'.format(unique_token, class_token),
    'a {0} {1} flying on top of a purple rug in a forest'.format(unique_token, class_token),
    'a {0} {1} wearing a santa hat'.format(unique_token, class_token),
    'a {0} {1} riding a bike'.format(unique_token, class_token),
    'a {0} {1} wearing pink glasses'.format(unique_token, class_token),
    'a {0} {1} shaking head'.format(unique_token, class_token),
    'a round shaped {0} {1} rolling on the floor'.format(unique_token, class_token),
]

if not os.path.exists(save_dir + "/" + method):
    os.makedirs(save_dir + "/" + method)

for idx, prompt in enumerate(prompt_list):
    for i in range(3):
        output = pipe(
            prompt=(
                prompt
            ),
            negative_prompt="bad quality, worse quality",
            num_frames=16,
            guidance_scale=7.5,
            num_inference_steps=25,
            generator=torch.Generator("cuda").manual_seed(42 + i * 1000),
        )
        frames = output.frames[0]
        export_to_gif(frames, f"{save_dir}/{method}/{method}_prompt{idx}_{i}.gif")