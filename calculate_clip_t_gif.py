import os
import re
import torch
from PIL import Image, ImageSequence
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel

# Load CLIP model and processor
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

unique_token = "sks"
class_token = "cat"

# Load your prompt list
prompts = [
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

method = "less_img"
gif_folder = f"sks_cat_animation/{method}"
pattern = re.compile(r".*_prompt(\d+)_\d+\.gif")

similarities = []
for fname in tqdm(os.listdir(gif_folder)):
    if not fname.lower().endswith(".gif"):
        continue
    match = pattern.match(fname)
    if not match:
        continue
    idx = int(match.group(1))
    if idx >= len(prompts):
        continue
    prompt = prompts[idx]
    gif_path = os.path.join(gif_folder, fname)
    with Image.open(gif_path) as im:
        for frame in ImageSequence.Iterator(im):
            frame = frame.convert("RGB")
            image_inputs = processor(images=frame, return_tensors="pt").to(device)
            text_inputs = processor(text=[prompt], return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                image_emb = model.get_image_features(**image_inputs)
                text_emb = model.get_text_features(**text_inputs)
                image_emb = image_emb / image_emb.norm(dim=-1, keepdim=True)
                text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
                sim = (image_emb @ text_emb.T).item()
                similarities.append(sim)

clip_t = sum(similarities) / len(similarities) if similarities else 0.0
print(f"CLIP-T (average cosine similarity between prompt and GIF frames): {clip_t:.4f}") 