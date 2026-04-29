# Generated from: Model3_parsec.ipynb
# Converted at: 2026-04-29T07:01:03.866Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

!pip install timm scikit-learn seaborn matplotlib pillow opencv-python -q

import os
import zipfile
import warnings
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
import timm

warnings.filterwarnings('ignore')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("PyTorch :", torch.__version__)
print("timm    :", timm.__version__)
print("Device  :", DEVICE)
if torch.cuda.is_available():
    print("GPU     :", torch.cuda.get_device_name(0))
    print("VRAM    :", round(torch.cuda.get_device_properties(0).total_memory/1e9,2), "GB")


train_dir = '/kaggle/input/datasets/b24bb1040/12-language-dataset/12-way script classification dataset/train_1800'
test_dir  = '/kaggle/input/datasets/b24bb1040/12-language-dataset/12-way script classification dataset/test_478'

print("Train classes:", sorted(os.listdir(train_dir)))
print("Test  classes:", sorted(os.listdir(test_dir)))

NUM_CLASSES = 12
SEED        = 42
EPOCHS      = 8

torch.manual_seed(SEED)
np.random.seed(SEED)

# KEY FIX:
# patch8  on 112×112 → (112/8)²  = 196 tokens  ← same as patch16, fits in GPU
# patch16 on 224×224 → (224/16)² = 196 tokens  ← your original baseline
# patch32 on 224×224 → (224/32)² =  49 tokens  ← fastest

PATCH_CONFIGS = {
    'patch8' : {
        'model_name' : 'vit_small_patch8_224',
        'patch_size' : 8,
        'img_size'   : 112,   # smaller image = fewer tokens = fits GPU
        'batch_size' : 32,
    },
    'patch16': {
        'model_name' : 'vit_base_patch16_224',
        'patch_size' : 16,
        'img_size'   : 224,
        'batch_size' : 32,
    },
    'patch32': {
        'model_name' : 'vit_base_patch32_224',
        'patch_size' : 32,
        'img_size'   : 224,
        'batch_size' : 32,
    },
}

for pk, cfg in PATCH_CONFIGS.items():
    n_tok = (cfg['img_size'] // cfg['patch_size']) ** 2
    print(f"  {pk:10s} | model={cfg['model_name']:28s} | "
          f"img={cfg['img_size']} | tokens={n_tok} | batch={cfg['batch_size']}")

class ScriptDataset(Dataset):
    def __init__(self, data_dir, transform=None, split='train',
                 val_split=0.1, seed=42):
        data_dir    = Path(data_dir)
        class_dirs  = sorted([d for d in data_dir.iterdir() if d.is_dir()])
        self.classes    = [d.name for d in class_dirs]
        self.cls_to_idx = {n: i for i, n in enumerate(self.classes)}

        all_paths, all_labels = [], []
        for c in class_dirs:
            for img in c.glob('*'):
                if img.suffix.lower() in ['.jpg', '.png', '.jpeg']:
                    all_paths.append(str(img))
                    all_labels.append(self.cls_to_idx[c.name])

        all_paths  = np.array(all_paths)
        all_labels = np.array(all_labels)
        rng = np.random.RandomState(seed)
        idx = rng.permutation(len(all_paths))
        all_paths, all_labels = all_paths[idx], all_labels[idx]

        n_val = int(len(all_paths) * val_split)
        if split == 'train':
            self.paths  = all_paths[n_val:]
            self.labels = all_labels[n_val:]
        elif split == 'val':
            self.paths  = all_paths[:n_val]
            self.labels = all_labels[:n_val]
        else:
            self.paths  = all_paths
            self.labels = all_labels

        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img   = Image.open(self.paths[idx]).convert('RGB')
        label = int(self.labels[idx])
        if self.transform:
            img = self.transform(img)
        return img, label


def get_transforms(img_size, augment=False):
    if augment:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),  # ← uses exact img_size
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),
        ])
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),      # ← uses exact img_size
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])


def build_loaders(patch_key):
    cfg   = PATCH_CONFIGS[patch_key]
    bs    = cfg['batch_size']
    sz    = cfg['img_size']          # ← patch8=112, patch16=224, patch32=224

    tr_tf = get_transforms(sz, augment=True)
    ev_tf = get_transforms(sz, augment=False)

    train_ds = ScriptDataset(train_dir, tr_tf, split='train')
    val_ds   = ScriptDataset(train_dir, ev_tf, split='val')
    test_ds  = ScriptDataset(test_dir,  ev_tf, split='test')

    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False,
                              num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=bs, shuffle=False,
                              num_workers=2, pin_memory=True)
    return train_loader, val_loader, test_loader, train_ds.classes


# Rebuild all loaders with correct sizes
loaders, class_names = {}, None
for pk in PATCH_CONFIGS:
    tr, va, te, cls = build_loaders(pk)
    loaders[pk]     = {'train': tr, 'val': va, 'test': te}
    class_names     = cls
    sz  = PATCH_CONFIGS[pk]['img_size']
    tok = (sz // PATCH_CONFIGS[pk]['patch_size']) ** 2
    print(f"  {pk}: img={sz}×{sz} | tokens={tok} | "
          f"train={len(tr.dataset)} | val={len(va.dataset)} | test={len(te.dataset)}")

print("\nClasses:", class_names)

def build_model(patch_key):
    cfg        = PATCH_CONFIGS[patch_key]
    model_name = cfg['model_name']
    img_size   = cfg['img_size']

    # Pass img_size to timm so positional embeddings are sized correctly
    model = timm.create_model(
        model_name,
        pretrained=True,
        num_classes=NUM_CLASSES,
        img_size=img_size        # ← KEY: tells timm the actual input size
    )
    model = model.to(DEVICE)

    n_tok    = (img_size // cfg['patch_size']) ** 2
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n✅ {patch_key} | {model_name} | img={img_size}")
    print(f"   Tokens : {n_tok} ({img_size//cfg['patch_size']}×"
          f"{img_size//cfg['patch_size']} grid)")
    print(f"   Params : {n_params:,}")
    return model

def train_one_epoch(model, loader, optimizer, criterion, scaler):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            outputs = model(imgs)
            loss    = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * imgs.size(0)
        preds       = outputs.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += imgs.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        with torch.cuda.amp.autocast():
            outputs = model(imgs)
            loss    = criterion(outputs, labels)

        total_loss += loss.item() * imgs.size(0)
        preds       = outputs.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += imgs.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return total_loss/total, correct/total, np.array(all_preds), np.array(all_labels)

results = {}
models  = {}

for pk in PATCH_CONFIGS:
    print(f"\n{'='*60}")
    print(f"  Training : {pk} | {PATCH_CONFIGS[pk]['model_name']}")
    print(f"{'='*60}")

    try:
        model     = build_model(pk)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
        scaler    = torch.cuda.amp.GradScaler()

        tr_loader = loaders[pk]['train']
        va_loader = loaders[pk]['val']
        te_loader = loaders[pk]['test']

        history = {'train_loss':[], 'train_acc':[],
                   'val_loss'  :[], 'val_acc'  :[]}

        best_val_acc = 0.0

        for epoch in range(1, EPOCHS + 1):
            tr_loss, tr_acc = train_one_epoch(model, tr_loader,
                                              optimizer, criterion, scaler)
            va_loss, va_acc, _, _ = evaluate(model, va_loader, criterion)
            scheduler.step()

            history['train_loss'].append(tr_loss)
            history['train_acc'].append(tr_acc)
            history['val_loss'].append(va_loss)
            history['val_acc'].append(va_acc)

            if va_acc > best_val_acc:
                best_val_acc = va_acc
                torch.save(model.state_dict(), f'best_{pk}.pth')

            print(f"  Epoch {epoch:2d}/{EPOCHS} | "
                  f"tr_loss={tr_loss:.4f} tr_acc={tr_acc:.4f} | "
                  f"val_loss={va_loss:.4f} val_acc={va_acc:.4f}")

        # Load best weights and test
        model.load_state_dict(torch.load(f'best_{pk}.pth'))
        te_loss, te_acc, y_pred, y_true = evaluate(model, te_loader, criterion)
        print(f"\n  [{pk}] Test Accuracy: {te_acc:.4f}")

        results[pk] = {
            'history'    : history,
            'test_acc'   : te_acc,
            'y_pred'     : y_pred,
            'y_true'     : y_true,
            'patch_size' : PATCH_CONFIGS[pk]['patch_size'],
            'n_tokens'   : (IMG_SIZE // PATCH_CONFIGS[pk]['patch_size']) ** 2,
        }
        models[pk] = model

        # Free GPU memory before next model
        if pk != list(PATCH_CONFIGS.keys())[-1]:
            torch.cuda.empty_cache()

    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"\n  ⚠️  Skipping {pk}: {e}")

print(f"\n✅ Trained: {list(models.keys())}")

patch_keys = list(models.keys())
colors     = {'patch8': 'royalblue', 'patch16': 'darkorange', 'patch32': 'green'}

fig, axes = plt.subplots(1, 2, figsize=(16, 5))
for pk in patch_keys:
    h = results[pk]['history']
    c = colors.get(pk, 'gray')
    axes[0].plot(h['train_acc'], label=f'{pk} train', color=c, linestyle='-')
    axes[0].plot(h['val_acc'],   label=f'{pk} val',   color=c, linestyle='--')
    axes[1].plot(h['train_loss'],label=f'{pk} train', color=c, linestyle='-')
    axes[1].plot(h['val_loss'],  label=f'{pk} val',   color=c, linestyle='--')

for ax, title in zip(axes, ['Accuracy', 'Loss']):
    ax.set_title(f'{title} — patch8 vs patch16 vs patch32', fontsize=13)
    ax.set_xlabel('Epoch'); ax.set_ylabel(title)
    ax.legend(); ax.grid(True)

plt.tight_layout()
plt.savefig('training_curves.png', dpi=150)
plt.show()

n_models = len(patch_keys)
fig, axes = plt.subplots(1, n_models, figsize=(13*n_models, 11))
if n_models == 1:
    axes = [axes]

for ax, pk in zip(axes, patch_keys):
    cm_mat = confusion_matrix(results[pk]['y_true'], results[pk]['y_pred'])
    sns.heatmap(cm_mat, annot=True, fmt='d', ax=ax,
                xticklabels=class_names, yticklabels=class_names, cmap='Blues')
    ax.set_title(f'{pk} | Acc: {results[pk]["test_acc"]:.4f}', fontsize=13)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.tick_params(axis='x', rotation=45)

plt.suptitle('Confusion Matrix — patch8 vs patch16 vs patch32',
             fontsize=16, y=1.01)
plt.tight_layout()
plt.savefig('confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.show()

def get_attention_heatmap(model, img_tensor, patch_size):
    """
    Extract attention from last transformer block CLS token.
    Uses timm's built-in forward_features with attention hooks.
    img_tensor: (1, 3, 224, 224) on DEVICE
    """
    attentions = []

    def hook_fn(module, input, output):
        # timm MHSA returns (x, attn) when attn_return enabled
        # We hook the softmax attention weights directly
        attentions.append(output.detach())

    # Register hook on last attention block's softmax
    # timm ViT structure: model.blocks[-1].attn.attn_drop
    hook = model.blocks[-1].attn.attn_drop.register_forward_hook(hook_fn)

    model.eval()
    with torch.no_grad():
        _ = model(img_tensor)

    hook.remove()

    if len(attentions) == 0:
        return None

    attn = attentions[0]           # (1, num_heads, T+1, T+1)
    # CLS token attention to all patches
    cls_attn = attn[0, :, 0, 1:]  # (num_heads, num_patches)
    cls_attn = cls_attn.mean(0)    # average over heads → (num_patches,)
    cls_attn = cls_attn.cpu().numpy()

    n_grid  = int(np.sqrt(len(cls_attn)))
    heatmap = cls_attn[:n_grid*n_grid].reshape(n_grid, n_grid)
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() + 1e-8)
    return heatmap


def overlay_heatmap(img_np, heatmap, alpha=0.5):
    """img_np: (H,W,3) float [0,1]"""
    h, w  = img_np.shape[:2]
    hmap  = cv2.resize(heatmap, (w, h))
    hmap  = np.uint8(255 * hmap)
    hmap  = cv2.applyColorMap(hmap, cv2.COLORMAP_JET)
    hmap  = cv2.cvtColor(hmap, cv2.COLOR_BGR2RGB)
    img8  = np.uint8(255 * img_np)
    out   = cv2.addWeighted(img8, 1-alpha, hmap, alpha, 0)
    return out


def load_img_display(path):
    """Load image as numpy (H,W,3) float32 [0,1] for display."""
    img = Image.open(path).convert('RGB').resize((IMG_SIZE, IMG_SIZE))
    return np.array(img).astype(np.float32) / 255.0


def load_img_tensor(path):
    """Load image as normalised tensor for model input."""
    img = eval_tf(Image.open(path).convert('RGB')).unsqueeze(0).to(DEVICE)
    return img


# ── Pick one sample per class ─────────────────────────────────────────────────
test_ds_ref = ScriptDataset(test_dir, eval_tf, split='test')
samples_per_class = {}
for path, label in zip(test_ds_ref.paths, test_ds_ref.labels):
    if label not in samples_per_class:
        samples_per_class[label] = path
    if len(samples_per_class) == NUM_CLASSES:
        break

# ── Plot ──────────────────────────────────────────────────────────────────────
n_cols = len(patch_keys) + 1
fig, axes = plt.subplots(NUM_CLASSES, n_cols,
                         figsize=(5*n_cols, 4*NUM_CLASSES))
if NUM_CLASSES == 1:
    axes = np.expand_dims(axes, 0)
if n_cols == 1:
    axes = np.expand_dims(axes, 1)

for row, cls_idx in enumerate(sorted(samples_per_class.keys())):
    path      = samples_per_class[cls_idx]
    img_disp  = load_img_display(path)
    img_tensor = load_img_tensor(path)

    # Column 0 — original
    axes[row, 0].imshow(img_disp)
    axes[row, 0].set_title(f'Original\n{class_names[cls_idx]}', fontsize=9)
    axes[row, 0].axis('off')

    for col, pk in enumerate(patch_keys):
        try:
            ps      = PATCH_CONFIGS[pk]['patch_size']
            heatmap = get_attention_heatmap(models[pk], img_tensor, ps)

            if heatmap is not None:
                overlay = overlay_heatmap(img_disp, heatmap)
                n_grid  = int(np.sqrt(heatmap.size))
                axes[row, col+1].imshow(overlay)
                axes[row, col+1].set_title(
                    f'{pk} | {n_grid}×{n_grid} grid\n'
                    f'Acc: {results[pk]["test_acc"]:.3f}', fontsize=9)
            else:
                axes[row, col+1].imshow(img_disp)
                axes[row, col+1].set_title(f'{pk}\n(no attn)', fontsize=9)
        except Exception as e:
            axes[row, col+1].text(0.5, 0.5, f'Error:\n{str(e)[:50]}',
                                  ha='center', va='center', fontsize=7,
                                  transform=axes[row, col+1].transAxes)
        axes[row, col+1].axis('off')

plt.suptitle('Attention Heatmaps — patch8 vs patch16 vs patch32\n'
             'CLS token attention averaged over all heads',
             fontsize=13, y=1.005)
plt.tight_layout()
plt.savefig('attention_heatmaps.png', dpi=130, bbox_inches='tight')
plt.show()
print("Saved: attention_heatmaps.png")

print("\n" + "="*70)
print(f"{'Patch':10s} {'Model':35s} {'Tokens':8s} {'Test Acc':10s} {'vs patch16'}")
print("="*70)

baseline = results.get('patch16', {}).get('test_acc', None)

for pk in patch_keys:
    r        = results[pk]
    acc      = r['test_acc']
    n_tok    = r['n_tokens']
    mname    = PATCH_CONFIGS[pk]['model_name']
    if baseline:
        diff     = acc - baseline
        diff_str = f"+{diff:.4f}" if diff >= 0 else f"{diff:.4f}"
        if pk == 'patch16': diff_str = "← baseline"
    else:
        diff_str = "N/A"
    print(f"{pk:10s} {mname:35s} {n_tok:<8d} {acc:<10.4f} {diff_str}")

print("="*70)

best_pk = max(results, key=lambda k: results[k]['test_acc'])
print(f"\n✅ Best: {best_pk}  (Acc: {results[best_pk]['test_acc']:.4f})\n")
print(classification_report(
    results[best_pk]['y_true'],
    results[best_pk]['y_pred'],
    target_names=class_names))