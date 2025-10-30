import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import os

# =========================================================
# 1. Local Imports (✅ works with your project structure)
# =========================================================
from models import get_3d_model
from datasets.object_sequence_dataset import ObjectSequenceDataset

# =========================================================
# 2. Setup
# =========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Choose model
model = get_3d_model(
    model_name="resnet3d",
    num_classes=1,
    in_channels=1,
    pretrained_path=None  # or path to pretrained weights
).to(device)

# =========================================================
# 3. Data Loaders
# =========================================================
train_dataset = ObjectSequenceDataset("data/train.csv")
val_dataset   = ObjectSequenceDataset("data/val.csv")
test_dataset  = ObjectSequenceDataset("data/test.csv")

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)
val_loader   = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=2)
test_loader  = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=2)

# =========================================================
# 4. Training Config
# =========================================================
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
num_epochs = 10
best_val_loss = float('inf')
save_path = "best_resnet3d_binary.pth"

# =========================================================
# 5. Training Loop
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
# 6. Evaluation
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
# 7. Classification Report & Confusion Matrix
# =========================================================
print("\n📊 Classification Report:")
print(classification_report(all_targets, all_preds, digits=4))

cm = confusion_matrix(all_targets, all_preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Pred 0', 'Pred 1'],
            yticklabels=['True 0', 'True 1'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('ResNet3D Confusion Matrix')
plt.show()

