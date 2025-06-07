import os
import re
import torch
from PIL import Image
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

img_folder = "sks_cat/less_img"
pattern = re.compile(r".*_prompt(\d+)_\d+\.\w+")

similarities = []
for fname in tqdm(os.listdir(img_folder)):
    if not fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif")):
        continue
    match = pattern.match(fname)
    if not match:
        continue
    idx = int(match.group(1))
    if idx >= len(prompts):
        continue
    prompt = prompts[idx]
    img_path = os.path.join(img_folder, fname)
    image = Image.open(img_path).convert("RGB")
    # Get CLIP embeddings (image and text processed separately)
    image_inputs = processor(images=image, return_tensors="pt").to(device)
    text_inputs = processor(text=[prompt], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        image_emb = model.get_image_features(**image_inputs)
        text_emb = model.get_text_features(**text_inputs)
        image_emb = image_emb / image_emb.norm(dim=-1, keepdim=True)
        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
        sim = (image_emb @ text_emb.T).item()
        similarities.append(sim)

clip_t = sum(similarities) / len(similarities) if similarities else 0.0
print(f"CLIP-T (average cosine similarity between prompt and image): {clip_t:.4f}") 