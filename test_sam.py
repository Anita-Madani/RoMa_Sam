from transformers import SamModel, SamProcessor
import torch
from PIL import Image
import requests
import time
import gc
import torch
torch.cuda.empty_cache()
gc.collect()

# === Load the model ===
start = time.time()
print("Loading SAM model from Hugging Face...")
model = SamModel.from_pretrained("facebook/sam-vit-huge")
print(f"Loaded model in {time.time() - start:.2f} seconds")
model.eval()

# === Check image encoder ===
image_encoder = model.vision_encoder

# === Create dummy input (or use a real one) ===
dummy_image = torch.randn(1, 3, 1024, 1024)

# Move to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_encoder = image_encoder.to(device)
dummy_image = dummy_image.to(device)

# === Forward pass ===
with torch.no_grad():
    features = image_encoder(dummy_image)

# === Print feature shape ===
print("Feature shape from SAM encoder:", features.last_hidden_state.shape)
