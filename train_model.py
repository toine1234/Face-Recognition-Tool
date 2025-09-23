import os, random
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from facenet_pytorch import InceptionResnetV1
from torch.utils.data import Dataset, DataLoader

# ============================
# Thiết lập thiết bị
# ============================
device = "mps" if torch.backends.mps.is_available() else (
    "cuda" if torch.cuda.is_available() else "cpu"
)
print(f"[INFO] Training on {device}")

# ============================
# Triplet Loss
# ============================
class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        pos_dist = (anchor - positive).pow(2).sum(1)   # khoảng cách anchor-positive
        neg_dist = (anchor - negative).pow(2).sum(1)   # khoảng cách anchor-negative
        losses = torch.relu(pos_dist - neg_dist + self.margin)
        return losses.mean()

# ============================
# Tạo Triplet Dataset từ ImageFolder
# ============================
class TripletDataset(Dataset):
    def __init__(self, dataset, num_triplets=1000):
        self.dataset = dataset
        self.num_triplets = num_triplets
        self.label_to_indices = {}

        for idx, (_, label) in enumerate(dataset.samples):
            self.label_to_indices.setdefault(label, []).append(idx)

        self.triplets = self._generate_triplets()

    def _generate_triplets(self):
        triplets = []
        labels = list(self.label_to_indices.keys())
        for _ in range(self.num_triplets):
            anchor_label = random.choice(labels)
            positive_label = anchor_label
            negative_label = random.choice(labels)
            while negative_label == anchor_label:
                negative_label = random.choice(labels)

            anchor_idx = random.choice(self.label_to_indices[anchor_label])
            positive_idx = random.choice(self.label_to_indices[positive_label])
            while positive_idx == anchor_idx:
                positive_idx = random.choice(self.label_to_indices[positive_label])

            negative_idx = random.choice(self.label_to_indices[negative_label])
            triplets.append((anchor_idx, positive_idx, negative_idx))
        return triplets

    def __getitem__(self, index):
        a_idx, p_idx, n_idx = self.triplets[index]
        a_img, _ = self.dataset[a_idx]
        p_img, _ = self.dataset[p_idx]
        n_img, _ = self.dataset[n_idx]
        return a_img, p_img, n_img

    def __len__(self):
        return len(self.triplets)

# ============================
# Data Augmentation
# ============================
transform = transforms.Compose([
    transforms.Resize((160,160)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # chuẩn hóa [-1,1]
])

train_dataset = datasets.ImageFolder("dataset", transform=transform)
triplet_dataset = TripletDataset(train_dataset, num_triplets=2000)
train_loader = DataLoader(triplet_dataset, batch_size=16, shuffle=True)

# ============================
# Mô hình FaceNet (fine-tune)
# ============================
model = InceptionResnetV1(pretrained="vggface2", classify=False).to(device)

# Freeze một số layer đầu, chỉ fine-tune phần cuối
for name, param in model.named_parameters():
    if "block8" not in name:   # chỉ fine-tune block cuối
        param.requires_grad = False

criterion = TripletLoss(margin=1.0)
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

# ============================
# Training Loop
# ============================
epochs = 5
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        a, p, n = [x.to(device) for x in batch]

        emb_a = model(a)
        emb_p = model(p)
        emb_n = model(n)

        loss = criterion(emb_a, emb_p, emb_n)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{epochs}, Loss={avg_loss:.4f}")

os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/facenet_triplet.pth")
print("✅ Training complete, model saved at models/facenet_triplet.pth")
