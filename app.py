import streamlit as st
import os
import numpy as np
import pickle
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from io import BytesIO

# ---------- Config ----------
st.set_page_config(
    page_title="Art Similarity Explorer",
    layout="wide",
    page_icon="🖼️"
)

bg_gradient = """
<style>
body {
    background: linear-gradient(to right, #ece9e6, #ffffff);
    color: #1a1a1a;
    font-family: 'Segoe UI', sans-serif;
}
</style>
"""
st.markdown(bg_gradient, unsafe_allow_html=True)

st.title("🎨 Art Similarity Explorer")
st.write("Upload an artwork or pick one from the dataset to find similar paintings.")

# ---------- Load Data ----------
with open("features.pkl", "rb") as f:
    features_dict = pickle.load(f)

image_paths = [os.path.join(r"D:\NGA_open_dataset", fname) for fname in features_dict.keys()]
features = np.array(list(features_dict.values()))
similarity_matrix = cosine_similarity(features)

# ---------- Functions ----------
def show_top_similar(query_feat, top_k=5):
    sims = cosine_similarity([query_feat], features).flatten()
    top_indices = sims.argsort()[::-1][1:top_k+1]
    return top_indices

def display_gallery(query_img, similar_indices):
    cols = st.columns(len(similar_indices)+1)
    with cols[0]:
        st.image(query_img, caption="🟡 Query Image", use_column_width=True)
    for i, idx in enumerate(similar_indices):
        with cols[i+1]:
            sim_img = Image.open(image_paths[idx])
            st.image(sim_img, caption=f"🔍 Similar #{i+1}", use_column_width=True)

# ---------- Upload or Pick ----------
upload_img = st.file_uploader("📤 Upload a painting image", type=["jpg", "jpeg", "png"])
or_choose = st.checkbox("📁 Or select from existing dataset")

query_feat = None
query_img = None

if upload_img:
    query_img = Image.open(upload_img).convert("RGB").resize((224, 224))
    # Load model just for inference
    import torch
    from torchvision import models, transforms

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet50(pretrained=True)
    model.fc = torch.nn.Identity()
    model.eval().to(device)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    with torch.no_grad():
        tensor = transform(query_img).unsqueeze(0).to(device)
        query_feat = model(tensor).squeeze().cpu().numpy()

elif or_choose:
    index = st.slider("Pick an index image:", 0, len(image_paths)-1, 0)
    query_img = Image.open(image_paths[index]).convert("RGB")
    query_feat = features[index]

# ---------- Show Results ----------
if query_feat is not None:
    top_similar = show_top_similar(query_feat)
    st.subheader("🧠 Most similar artworks:")
    display_gallery(query_img, top_similar)
