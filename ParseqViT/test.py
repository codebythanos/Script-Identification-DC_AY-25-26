import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
import timm
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ── Config ────────────────────────────────────────────────────────────────────
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_CLASSES = 12
IMG_SIZE    = 224
BATCH_SIZE  = 32
WEIGHTS_PATH = 'vit_patch16_weights.pth'   # download from GitHub release
TEST_DIR     = '/kaggle/input/datasets/b24bb1040/12-language-dataset/12-way script classification dataset/test_478'

print(f"Device: {DEVICE}")

# ── Dataset ───────────────────────────────────────────────────────────────────
class ScriptDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        data_dir   = Path(data_dir)
        class_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
        self.classes    = [d.name for d in class_dirs]
        self.cls_to_idx = {n: i for i, n in enumerate(self.classes)}

        self.paths, self.labels = [], []
        for c in class_dirs:
            for img in c.glob('*'):
                if img.suffix.lower() in ['.jpg', '.png', '.jpeg']:
                    self.paths.append(str(img))
                    self.labels.append(self.cls_to_idx[c.name])

        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img   = Image.open(self.paths[idx]).convert('RGB')
        label = int(self.labels[idx])
        if self.transform:
            img = self.transform(img)
        return img, label


# ── Transform ─────────────────────────────────────────────────────────────────
test_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])

# ── Load data ─────────────────────────────────────────────────────────────────
test_ds     = ScriptDataset(TEST_DIR, test_tf)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE,
                         shuffle=False, num_workers=2, pin_memory=True)

print(f"Test samples : {len(test_ds)}")
print(f"Classes      : {test_ds.classes}")

# ── Build model & load weights ────────────────────────────────────────────────
model = timm.create_model(
    'vit_base_patch16_224',
    pretrained=False,
    num_classes=NUM_CLASSES,
    img_size=IMG_SIZE,
)
model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()
print(f"Weights loaded from: {WEIGHTS_PATH}")

# ── Evaluate ──────────────────────────────────────────────────────────────────
criterion = nn.CrossEntropyLoss()
total_loss, correct, total = 0.0, 0, 0
all_preds, all_labels = [], []

with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        outputs = model(imgs)
        loss    = criterion(outputs, labels)

        total_loss += loss.item() * imgs.size(0)
        preds       = outputs.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += imgs.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

test_acc  = correct / total
test_loss = total_loss / total
print(f"\nTest Loss : {test_loss:.4f}")
print(f"Test Acc  : {test_acc:.4f}  ({correct}/{total})\n")

# ── Classification report ─────────────────────────────────────────────────────
print(classification_report(all_labels, all_preds, target_names=test_ds.classes))

# ── Confusion matrix ──────────────────────────────────────────────────────────
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=test_ds.classes,
            yticklabels=test_ds.classes)
plt.title(f'Confusion Matrix — vit_base_patch16 | Acc: {test_acc:.4f}', fontsize=14)
plt.xlabel('Predicted'); plt.ylabel('True')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('confusion_matrix_patch16.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: confusion_matrix_patch16.png")
