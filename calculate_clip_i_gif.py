import os
import torch
from PIL import Image, ImageSequence
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def get_image_embeddings_from_folder(image_folder):
    embeddings = []
    files = [f for f in os.listdir(image_folder) if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif"))]
    for fname in tqdm(files, desc=f"Embedding {image_folder}"):
        img_path = os.path.join(image_folder, fname)
        image = Image.open(img_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            emb = model.get_image_features(**inputs)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            embeddings.append(emb.cpu())
    if len(embeddings) == 0:
        return torch.empty(0, model.config.projection_dim)
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
                    emb = model.get_image_features(**inputs)
                    emb = emb / emb.norm(dim=-1, keepdim=True)
                    embeddings.append(emb.cpu())
    if len(embeddings) == 0:
        return torch.empty(0, model.config.projection_dim)
    return torch.cat(embeddings, dim=0)

# Paths to your folders
method = "less_img"
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

clip_i = sum(similarities) / len(similarities)
print(f"CLIP-I (average pairwise cosine similarity between GIF frames and real images): {clip_i:.4f}")
