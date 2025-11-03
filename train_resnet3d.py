import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
import ast
import os
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# =========================================================
# 1. Dataset Definition
# =========================================================
class ObjectSequenceDataset(Dataset):
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        # Drop unnecessary columns
        self.df = self.df.drop(columns=['Unnamed: 0', 'video_id', 'frame'], errors='ignore')
        assert 'target' in self.df.columns, f"'target' column missing in {csv_path}"

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        obj_data = row.drop('target').values
        parsed = []
        for s in obj_data:
            coords = ast.literal_eval(s) if isinstance(s, str) else [0, 0, 0, 0]
            parsed.append(coords)
        tensor_data = torch.tensor(parsed, dtype=torch.float32).view(30, 10, 4)
        target = torch.tensor(row['target'], dtype=torch.float32)
        return tensor_data.unsqueeze(0), target.unsqueeze(0)  # (1, 30, 10, 4), (1,)



# =========================================================
# 3. Setup
# =========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load pretrained 3D ResNet-101 model (or skip pretrained)
model = generate_resnet101_3d_binary(pretrained_path="/content/r3d101_KM_200ep.pth").to(device)

# =========================================================
# 4. Data Loaders
# =========================================================
train_dataset = ObjectSequenceDataset("/content/train.csv")
val_dataset   = ObjectSequenceDataset("/content/val.csv")
test_dataset  = ObjectSequenceDataset("/content/test.csv")

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)
val_loader   = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=2)
test_loader  = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=2)

# =========================================================
# 5. Training Configuration
# =========================================================
criterion = nn.BCELoss()  # Model outputs sigmoid values already
optimizer = optim.Adam(model.parameters(), lr=1e-4)
num_epochs = 10
best_val_loss = float('inf')
save_path = "best_resnet3d_binary.pth"

# =========================================================
# 6. Training Loop
# =========================================================
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0

    for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * inputs.size(0)

    train_loss /= len(train_loader.dataset)

    # ----- Validation -----
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, targets in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item() * inputs.size(0)

    val_loss /= len(val_loader.dataset)
    print(f"Epoch [{epoch+1}/{num_epochs}] ➜ Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    # ----- Save Best Model -----
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), save_path)
        print(f"✅ Saved Best Model (Epoch {epoch+1})")

# =========================================================
# 7. Evaluation
# =========================================================
model.load_state_dict(torch.load(save_path, map_location=device))
model.eval()
correct, total = 0, 0
all_preds, all_targets = [], []

with torch.no_grad():
    for inputs, targets in tqdm(test_loader, desc="Testing"):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        preds = (outputs > 0.5).float()
        all_preds.extend(preds.cpu().numpy().flatten())
        all_targets.extend(targets.cpu().numpy().flatten())
        correct += (preds == targets).sum().item()
        total += targets.size(0)

accuracy = 100 * correct / total
print(f"✅ Test Accuracy: {accuracy:.2f}%")

# =========================================================
# 8. Classification Report & Confusion Matrix
# =========================================================
from sklearn.metrics import classification_report, confusion_matrix

print("\n📊 Classification Report:")
print(classification_report(all_targets, all_preds, digits=4))

cm = confusion_matrix(all_targets, all_preds)
print("\n🧩 Confusion Matrix:")
print(cm)

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Pred 0', 'Pred 1'],
            yticklabels=['True 0', 'True 1'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('ResNet3D Confusion Matrix')
plt.show()
