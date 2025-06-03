import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt

# === 1. 載入資料 ===
data = np.load("dreamer_psd_features.npz")
X = torch.tensor(data['X'], dtype=torch.float32).unsqueeze(1)  # shape: (N, 1, 14, 5)
Y = torch.tensor(data['Y'], dtype=torch.long)  # shape: (N, 2)

dataset = TensorDataset(X, Y)
train_size = int(0.8 * len(dataset))
train_ds, val_ds = random_split(dataset, [train_size, len(dataset) - train_size])
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32)

# === 2. 定義 CNN 模型 ===
class EEGCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 2))  # (B, 32, 4, 2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 4 * 2, 64), nn.ReLU(),
            nn.Linear(64, 10)  # 5 類 arousal + 5 類 valence
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x.view(-1, 2, 5)  # (B, 2, 5)

model = EEGCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# === 3. 開始訓練 ===
epochs = 5000
train_loss, val_acc = [], []

for ep in range(epochs):
    print(f"Epoch {ep+1}")
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out[:, 0], yb[:, 0]) + criterion(out[:, 1], yb[:, 1])
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    train_loss.append(total_loss / len(train_loader))

    # 驗證
    model.eval()
    correct = [0, 0]
    total = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            out = model(xb)
            pred = out.argmax(dim=2)
            correct[0] += (pred[:, 0] == yb[:, 0]).sum().item()
            correct[1] += (pred[:, 1] == yb[:, 1]).sum().item()
            total += yb.size(0)
    val_acc.append((correct[0] / total, correct[1] / total))
    print(f"Loss: {train_loss[-1]:.4f} | Arousal Acc: {val_acc[-1][0]:.2%} | Valence Acc: {val_acc[-1][1]:.2%}")

# === 4. 視覺化 ===
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(train_loss, label='Train Loss')
plt.title("CNN Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot([a for a, _ in val_acc], label='Arousal Acc')
plt.plot([v for _, v in val_acc], label='Valence Acc')
plt.title("Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("easy_cnn_training_from_npz_result.jpg", dpi=300)
plt.show()
