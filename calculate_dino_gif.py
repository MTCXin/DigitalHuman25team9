import os
import torch
from PIL import Image, ImageSequence
from tqdm import tqdm
from transformers import ViTImageProcessor, ViTModel

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load DINO ViT-S/16 model from transformers
model = ViTModel.from_pretrained("facebook/dino-vits16").to(device)
processor = ViTImageProcessor.from_pretrained("facebook/dino-vits16")

def get_image_embeddings_from_folder(image_folder):
    embeddings = []
    files = [f for f in os.listdir(image_folder) if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif"))]
    for fname in tqdm(files, desc=f"Embedding {image_folder}"):
        img_path = os.path.join(image_folder, fname)
        image = Image.open(img_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            emb = outputs.last_hidden_state[:, 0]  # CLS token
            emb = emb / emb.norm(dim=-1, keepdim=True)
            embeddings.append(emb.cpu())
    if len(embeddings) == 0:
        return torch.empty(0, model.config.hidden_size)
    return torch.cat(embeddings, dim=0)

def get_gif_frame_embeddings(gif_folder):
    embeddings = []
    files = [f for f in os.listdir(gif_folder) if f.lower().endswith(".gif")]
    for fname in tqdm(files, desc=f"Embedding GIF frames in {gif_folder}"):
        gif_path = os.path.join(gif_folder, fname)
        with Image.open(gif_path) as im:
            for frame in ImageSequence.Iterator(im):
                frame = frame.convert("RGB")
                inputs = processor(images=frame, return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs = model(**inputs)
                    emb = outputs.last_hidden_state[:, 0]  # CLS token
                    emb = emb / emb.norm(dim=-1, keepdim=True)
                    embeddings.append(emb.cpu())
    if len(embeddings) == 0:
        return torch.empty(0, model.config.hidden_size)
    return torch.cat(embeddings, dim=0)

# Paths to your folders
method = "more_camera_distribute"
gen_gif_folder = "sks_cat_animation/" + method
real_img_folder = "my_cat/" + method

gen_embs = get_gif_frame_embeddings(gen_gif_folder)
real_embs = get_image_embeddings_from_folder(real_img_folder)

if gen_embs.shape[0] == 0 or real_embs.shape[0] == 0:
    print("No frames or images found in one or both folders.")
    exit(1)

# Compute average pairwise cosine similarity
similarities = []
for gen_emb in tqdm(gen_embs, desc="Computing similarities"):
    sim = (gen_emb @ real_embs.T).squeeze()  # Cosine similarity with all real images
    similarities.append(sim.mean().item())

dino_i = sum(similarities) / len(similarities)
print(f"DINO (average pairwise cosine similarity between GIF frames and real images): {dino_i:.4f}") 