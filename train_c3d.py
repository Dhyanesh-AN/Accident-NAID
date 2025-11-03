import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
import ast
import os

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
        
        # Extract features (ignore target)
        obj_data = row.drop('target').values
        parsed = []
        for s in obj_data:
            coords = ast.literal_eval(s) if isinstance(s, str) else [0, 0, 0, 0]
            parsed.append(coords)

        # Shape: (30*10, 4) → (30, 10, 4)
        tensor_data = torch.tensor(parsed, dtype=torch.float32).view(30, 10, 4)
        
        # Binary target
        target = torch.tensor(row['target'], dtype=torch.float32)
        return tensor_data.unsqueeze(0), target.unsqueeze(0)  # (1, 30, 10, 4), (1,)

# =========================================================
# 2. Model Import
# ========================================================= 

# =========================================================
# 3. Setup
# =========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = C3D_Modified(num_classes=1, pretrained=True).to(device)

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
criterion = nn.BCEWithLogitsLoss()  # ✅ use BCEWithLogitsLoss (handles sigmoid internally)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
num_epochs = 10
best_val_loss = float('inf')
save_path = "best_c3d_binary.pth"

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

with torch.no_grad():
    for inputs, targets in tqdm(test_loader, desc="Testing"):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        preds = (torch.sigmoid(outputs) > 0.5).float()  # ✅ sigmoid for binary threshold
        correct += (preds == targets).sum().item()
        total += targets.size(0)

print(f"✅ Test Accuracy: {100 * correct / total:.2f}%")
