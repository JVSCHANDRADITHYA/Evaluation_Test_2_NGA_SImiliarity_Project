import os
import pickle
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from torchvision import models, transforms

# ----- CONFIG -----
DATA_DIR = r'D:\NGA_open_dataset'         # Folder with dataset images
FEATURES_PATH = "features.pkl"             # Saved features file
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----- Load Model -----
model = models.resnet50(pretrained=True)
model.fc = torch.nn.Identity()
model = model.to(DEVICE).eval()

# ----- Image Transform -----
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ----- Load Features -----
with open(FEATURES_PATH, "rb") as f:
    features_dict = pickle.load(f)

image_paths = [os.path.join(DATA_DIR, fname) for fname in features_dict.keys()]
all_feats = np.array(list(features_dict.values()))
print(f"✅ Loaded features for {len(image_paths)} images.")

# ----- Similarity Function -----
def find_similar_from_file(query_img_path, top_k=5):
    try:
        image = Image.open(query_img_path).convert("RGB")
        image = transform(image).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            query_feat = model(image).squeeze().cpu().numpy()

        sim_scores = cosine_similarity([query_feat], all_feats)[0]
        top_indices = sim_scores.argsort()[::-1][:top_k]

        plt.figure(figsize=(15, 5))
        plt.subplot(1, top_k+1, 1)
        plt.imshow(Image.open(query_img_path))
        plt.title("Query Image")
        plt.axis('off')

        for i, idx in enumerate(top_indices):
            sim_img_path = image_paths[idx]
            sim_name = os.path.basename(sim_img_path)[:20]  # 20 chars max
            plt.subplot(1, top_k+1, i+2)
            plt.imshow(Image.open(sim_img_path))
            plt.title(f"{sim_name}", fontsize=8)
            plt.axis('off')

        plt.show()

    except Exception as e:
        print(f"❌ Error: {e}")

# ----- Run Query -----
if __name__ == "__main__":
    query_path = "vase.jpg"  # << Change this
    find_similar_from_file(query_path, top_k=5)
