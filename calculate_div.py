import os
import re
import torch
from PIL import Image
from tqdm import tqdm
import lpips
import itertools
import numpy as np

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
loss_fn = lpips.LPIPS(net='vgg').to(device)

img_folder = "sks_cat/more_img"
pattern = re.compile(r".*_prompt(\d+)_\d+\.\w+")

# Group images by prompt index
prompt_to_files = {}
for fname in os.listdir(img_folder):
    if not fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif")):
        continue
    match = pattern.match(fname)
    if not match:
        continue
    idx = int(match.group(1))
    prompt_to_files.setdefault(idx, []).append(os.path.join(img_folder, fname))

all_prompt_divs = []
for idx, files in tqdm(prompt_to_files.items(), desc="Prompts"):
    if len(files) < 2:
        continue  # Need at least two images to compute diversity
    pairwise_scores = []
    for f1, f2 in itertools.combinations(files, 2):
        img1 = Image.open(f1).convert("RGB").resize((256, 256))
        img2 = Image.open(f2).convert("RGB").resize((256, 256))
        t1 = torch.tensor(np.array(img1)).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1
        t2 = torch.tensor(np.array(img2)).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1
        t1 = t1.to(device)
        t2 = t2.to(device)
        with torch.no_grad():
            score = loss_fn(t1, t2).item()
        pairwise_scores.append(score)
    if pairwise_scores:
        all_prompt_divs.append(sum(pairwise_scores) / len(pairwise_scores))

div_metric = sum(all_prompt_divs) / len(all_prompt_divs) if all_prompt_divs else 0.0
print(f"DIV (average LPIPS distance between generated images with the same prompt): {div_metric:.4f}") 