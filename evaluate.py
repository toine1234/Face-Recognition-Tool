import os
import torch
import numpy as np
from torchvision import datasets, transforms
from facenet_pytorch import InceptionResnetV1
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score
from itertools import combinations
import random

# ============================
# Thiết lập thiết bị
# ============================
device = "mps" if torch.backends.mps.is_available() else (
    "cuda" if torch.cuda.is_available() else "cpu"
)
print(f"[INFO] Evaluating on {device}")

# ============================
# Load mô hình đã fine-tune
# ============================
model = InceptionResnetV1(pretrained="vggface2", classify=False).to(device)
model.load_state_dict(torch.load("models/facenet_triplet.pth", map_location=device))
model.eval()

# ============================
# Dataset + transform (test set)
# ============================
transform = transforms.Compose([
    transforms.Resize((160,160)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

dataset = datasets.ImageFolder("dataset", transform=transform)
print(f"[INFO] Loaded dataset with {len(dataset)} images, {len(dataset.classes)} classes.")

# ============================
# Hàm trích embedding
# ============================
def get_embedding(img_tensor):
    with torch.no_grad():
        emb = model(img_tensor.unsqueeze(0).to(device)).cpu().numpy()[0]
    # chuẩn hóa L2
    n = np.linalg.norm(emb)
    if n > 0: emb = emb / n
    return emb

# ============================
# Sinh embeddings cho toàn bộ dataset
# ============================
embeddings, labels = [], []
for img, label in dataset:
    emb = get_embedding(img)
    embeddings.append(emb)
    labels.append(label)

embeddings = np.array(embeddings)
labels = np.array(labels)

print(f"[INFO] Generated {embeddings.shape[0]} embeddings of dim {embeddings.shape[1]}")

# ============================
# Đánh giá accuracy bằng pairwise
# ============================
pairs = list(combinations(range(len(dataset)), 2))  # tất cả cặp (i,j)
random.shuffle(pairs)
pairs = pairs[:2000]  # chọn ngẫu nhiên 2000 cặp để test nhanh

y_true, y_pred = [], []
threshold = 0.65  # ngưỡng cosine similarity

for i, j in pairs:
    sim = cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]
    same_person = (labels[i] == labels[j])
    y_true.append(1 if same_person else 0)
    y_pred.append(1 if sim > threshold else 0)

acc = accuracy_score(y_true, y_pred)
print(f"✅ Evaluation complete. Accuracy = {acc*100:.2f}% (threshold={threshold})")
