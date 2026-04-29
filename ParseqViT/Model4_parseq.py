# Generated from: Model4_parseq.ipynb
# Converted at: 2026-04-29T15:18:13.786Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

!pip install timm scikit-learn seaborn matplotlib pillow opencv-python -q

import os, zipfile, warnings
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import torch
import torch.nn as nn
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
    print("VRAM    :", round(torch.cuda.get_device_properties(0).total_memory/1e9, 2), "GB")


train_dir = '/kaggle/input/datasets/b24bb1040/12-language-dataset/12-way script classification dataset/train_1800'
test_dir  = '/kaggle/input/datasets/b24bb1040/12-language-dataset/12-way script classification dataset/test_478'

print("Train classes:", sorted(os.listdir(train_dir)))
print("Test  classes:", sorted(os.listdir(test_dir)))

NUM_CLASSES = 12
IMG_H       = 32    # height  — PARSeq paper uses 128×32
IMG_W       = 128   # width
SEED        = 42
EPOCHS      = 8

torch.manual_seed(SEED)
np.random.seed(SEED)

# patch8  on 128×32 → (128/8)×(32/8)   = 16×4  = 64  tokens
# patch16 on 128×32 → (128/16)×(32/16) = 8×2   = 16  tokens
# patch32 on 128×32 → (128/32)×(32/32) = 4×1   =  4  tokens
# All very lightweight — fits easily on Kaggle GPU

PATCH_CONFIGS = {
    'patch8' : {
        'model_name' : 'vit_small_patch8_224',
        'patch_size' : 8,
        'batch_size' : 64,
    },
    'patch16': {
        'model_name' : 'vit_base_patch16_224',
        'patch_size' : 16,
        'batch_size' : 64,
    },
    'patch32': {
        'model_name' : 'vit_base_patch32_224',
        'patch_size' : 32,
        'batch_size' : 64,
    },
}

for pk, cfg in PATCH_CONFIGS.items():
    h_tok = IMG_H // cfg['patch_size']
    w_tok = IMG_W // cfg['patch_size']
    print(f"  {pk:10s} | model={cfg['model_name']:28s} | "
          f"grid={w_tok}×{h_tok} | tokens={w_tok*h_tok}")

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


def get_transforms(augment=False):
    if augment:
        return transforms.Compose([
            transforms.Resize((IMG_H, IMG_W)),   # 32×128 rectangular
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),
        ])
    return transforms.Compose([
        transforms.Resize((IMG_H, IMG_W)),       # 32×128 rectangular
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])


def build_loaders(patch_key):
    bs    = PATCH_CONFIGS[patch_key]['batch_size']
    tr_tf = get_transforms(augment=True)
    ev_tf = get_transforms(augment=False)

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


loaders, class_names = {}, None
for pk in PATCH_CONFIGS:
    tr, va, te, cls = build_loaders(pk)
    loaders[pk]     = {'train': tr, 'val': va, 'test': te}
    class_names     = cls
    ps  = PATCH_CONFIGS[pk]['patch_size']
    tok = (IMG_W // ps) * (IMG_H // ps)
    print(f"  {pk}: img={IMG_W}×{IMG_H} | tokens={tok} | "
          f"train={len(tr.dataset)} | val={len(va.dataset)} | test={len(te.dataset)}")

print("\nClasses:", class_names)

def build_model(patch_key):
    cfg        = PATCH_CONFIGS[patch_key]
    model_name = cfg['model_name']
    ps         = cfg['patch_size']

    # timm accepts img_size as (H, W) tuple for rectangular images
    model = timm.create_model(
        model_name,
        pretrained=True,
        num_classes=NUM_CLASSES,
        img_size=(IMG_H, IMG_W)      # ← (32, 128) rectangular
    )
    model = model.to(DEVICE)

    h_tok    = IMG_H // ps
    w_tok    = IMG_W // ps
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n✅ {patch_key} | {model_name}")
    print(f"   Input   : {IMG_W}×{IMG_H} (W×H)")
    print(f"   Grid    : {w_tok}×{h_tok} = {w_tok*h_tok} tokens")
    print(f"   Params  : {n_params:,}")
    return model


# Sanity check all 3
for pk in PATCH_CONFIGS:
    m = build_model(pk)
    dummy = torch.randn(2, 3, IMG_H, IMG_W).to(DEVICE)
    out   = m(dummy)
    print(f"   Output shape: {out.shape} ✅")
    del m, dummy
    torch.cuda.empty_cache()

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
        correct    += (outputs.argmax(1) == labels).sum().item()
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
        correct    += (outputs.argmax(1) == labels).sum().item()
        total      += imgs.size(0)
        all_preds.extend(outputs.argmax(1).cpu().numpy())
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
        optimizer = torch.optim.AdamW(model.parameters(),
                                      lr=5e-5, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                        optimizer, T_max=EPOCHS)
        scaler    = torch.cuda.amp.GradScaler()

        tr_loader = loaders[pk]['train']
        va_loader = loaders[pk]['val']
        te_loader = loaders[pk]['test']

        history      = {'train_loss':[], 'train_acc':[],
                        'val_loss'  :[], 'val_acc'  :[]}
        best_val_acc = 0.0
        best_ckpt    = f'/kaggle/working/best_{pk}.pth'

        for epoch in range(1, EPOCHS + 1):
            tr_loss, tr_acc       = train_one_epoch(model, tr_loader,
                                                    optimizer, criterion, scaler)
            va_loss, va_acc, _, _ = evaluate(model, va_loader, criterion)
            scheduler.step()

            history['train_loss'].append(tr_loss)
            history['train_acc'].append(tr_acc)
            history['val_loss'].append(va_loss)
            history['val_acc'].append(va_acc)

            if va_acc > best_val_acc:
                best_val_acc = va_acc
                torch.save(model.state_dict(), best_ckpt)
                print(f"  💾 Saved best at epoch {epoch} "
                      f"(val_acc={va_acc:.4f})")

            print(f"  Epoch {epoch:2d}/{EPOCHS} | "
                  f"tr_loss={tr_loss:.4f} tr_acc={tr_acc:.4f} | "
                  f"val_loss={va_loss:.4f} val_acc={va_acc:.4f}")

        # Load best and test
        model.load_state_dict(torch.load(best_ckpt))
        te_loss, te_acc, y_pred, y_true = evaluate(model, te_loader, criterion)
        print(f"\n  [{pk}] Test Accuracy: {te_acc:.4f}")

        results[pk] = {
            'history'   : history,
            'test_acc'  : te_acc,
            'y_pred'    : y_pred,
            'y_true'    : y_true,
            'patch_size': PATCH_CONFIGS[pk]['patch_size'],
            'n_tokens'  : (IMG_W // PATCH_CONFIGS[pk]['patch_size']) *
                          (IMG_H // PATCH_CONFIGS[pk]['patch_size']),
            'ckpt_path' : best_ckpt,
        }
        models[pk] = model
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
    axes[0].plot(h['train_acc'],  label=f'{pk} train', color=c, linestyle='-')
    axes[0].plot(h['val_acc'],    label=f'{pk} val',   color=c, linestyle='--')
    axes[1].plot(h['train_loss'], label=f'{pk} train', color=c, linestyle='-')
    axes[1].plot(h['val_loss'],   label=f'{pk} val',   color=c, linestyle='--')

for ax, title in zip(axes, ['Accuracy', 'Loss']):
    ax.set_title(f'{title} — patch8 vs patch16 vs patch32 (128×32)',
                 fontsize=13)
    ax.set_xlabel('Epoch'); ax.set_ylabel(title)
    ax.legend(); ax.grid(True)

plt.tight_layout()
plt.savefig('/kaggle/working/training_curves.png', dpi=150)
plt.show()

n_models = len(patch_keys)
fig, axes = plt.subplots(1, n_models, figsize=(13*n_models, 11))
if n_models == 1:
    axes = [axes]

for ax, pk in zip(axes, patch_keys):
    cm_mat = confusion_matrix(results[pk]['y_true'], results[pk]['y_pred'])
    sns.heatmap(cm_mat, annot=True, fmt='d', ax=ax,
                xticklabels=class_names, yticklabels=class_names,
                cmap='Blues')
    ax.set_title(f'{pk} | tokens={results[pk]["n_tokens"]}'
                 f'\nAcc: {results[pk]["test_acc"]:.4f}', fontsize=13)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.tick_params(axis='x', rotation=45)

plt.suptitle('Confusion Matrix — patch8 vs patch16 vs patch32 (128×32 images)',
             fontsize=15, y=1.01)
plt.tight_layout()
plt.savefig('/kaggle/working/confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.show()

def get_attention_heatmap(model, img_tensor, patch_key):
    attentions = []

    def hook_fn(module, input, output):
        attentions.append(output.detach())

    hook = model.blocks[-1].attn.attn_drop.register_forward_hook(hook_fn)
    model.eval()
    with torch.no_grad():
        _ = model(img_tensor)
    hook.remove()

    if not attentions:
        return None

    attn     = attentions[0]           # (1, heads, T+1, T+1)
    cls_attn = attn[0, :, 0, 1:]      # (heads, tokens)
    cls_attn = cls_attn.mean(0).cpu().numpy()

    ps   = PATCH_CONFIGS[patch_key]['patch_size']
    h_g  = IMG_H // ps
    w_g  = IMG_W // ps
    heatmap = cls_attn[:h_g*w_g].reshape(h_g, w_g)
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() + 1e-8)
    return heatmap


def overlay_heatmap(img_np, heatmap, alpha=0.5):
    h, w = img_np.shape[:2]
    hmap = cv2.resize(heatmap, (w, h))
    hmap = cv2.applyColorMap(np.uint8(255*hmap), cv2.COLORMAP_JET)
    hmap = cv2.cvtColor(hmap, cv2.COLOR_BGR2RGB)
    out  = cv2.addWeighted(np.uint8(255*img_np), 1-alpha, hmap, alpha, 0)
    return out


def load_img_display(path):
    img = Image.open(path).convert('RGB').resize((IMG_W, IMG_H))
    return np.array(img).astype(np.float32) / 255.0


def load_img_tensor(path):
    tf_ = get_transforms(augment=False)
    return tf_(Image.open(path).convert('RGB')).unsqueeze(0).to(DEVICE)


# One sample per class
test_ref = ScriptDataset(test_dir, get_transforms(False), split='test')
samples  = {}
for path, label in zip(test_ref.paths, test_ref.labels):
    if label not in samples:
        samples[label] = path
    if len(samples) == NUM_CLASSES:
        break

n_cols = len(patch_keys) + 1
fig, axes = plt.subplots(NUM_CLASSES, n_cols,
                         figsize=(5*n_cols, 3*NUM_CLASSES))
if NUM_CLASSES == 1:
    axes = np.expand_dims(axes, 0)
if n_cols == 1:
    axes = np.expand_dims(axes, 1)

for row, cls_idx in enumerate(sorted(samples.keys())):
    path     = samples[cls_idx]
    img_disp = load_img_display(path)

    axes[row, 0].imshow(img_disp)
    axes[row, 0].set_title(f'Original\n{class_names[cls_idx]}', fontsize=8)
    axes[row, 0].axis('off')

    for col, pk in enumerate(patch_keys):
        try:
            img_tensor = load_img_tensor(path)
            heatmap    = get_attention_heatmap(models[pk], img_tensor, pk)
            if heatmap is not None:
                overlay = overlay_heatmap(img_disp, heatmap)
                ps      = PATCH_CONFIGS[pk]['patch_size']
                axes[row, col+1].imshow(overlay)
                axes[row, col+1].set_title(
                    f'{pk} | {IMG_W//ps}×{IMG_H//ps} grid\n'
                    f'Acc={results[pk]["test_acc"]:.3f}', fontsize=8)
            else:
                axes[row, col+1].imshow(img_disp)
                axes[row, col+1].set_title(f'{pk} no attn', fontsize=8)
        except Exception as e:
            axes[row, col+1].text(0.5, 0.5, f'Err:\n{str(e)[:40]}',
                                  ha='center', va='center', fontsize=7,
                                  transform=axes[row, col+1].transAxes)
        axes[row, col+1].axis('off')

plt.suptitle('Attention Heatmaps — patch8 vs patch16 vs patch32 | 128×32 images',
             fontsize=12, y=1.005)
plt.tight_layout()
plt.savefig('/kaggle/working/attention_heatmaps.png', dpi=130, bbox_inches='tight')
plt.show()

# ── Summary table ─────────────────────────────────────────────────────────────
print("\n" + "="*65)
print(f"{'Patch':10s} {'Tokens':8s} {'Test Acc':12s} {'vs patch16'}")
print("="*65)

baseline = results.get('patch16', {}).get('test_acc', None)
for pk in patch_keys:
    r        = results[pk]
    acc      = r['test_acc']
    n_tok    = r['n_tokens']
    diff_str = "N/A"
    if baseline:
        diff     = acc - baseline
        diff_str = f"+{diff:.4f}" if diff >= 0 else f"{diff:.4f}"
        if pk == 'patch16': diff_str = "← baseline"
    print(f"{pk:10s} {n_tok:<8d} {acc:<12.4f} {diff_str}")

print("="*65)

# ── Find best model ───────────────────────────────────────────────────────────
best_pk  = max(results, key=lambda k: results[k]['test_acc'])
best_acc = results[best_pk]['test_acc']
print(f"\n🏆 Best model: {best_pk}  (Acc: {best_acc:.4f})")

# ── Save ONLY best model weights to Kaggle output ────────────────────────────
import shutil
best_save_path = f'/kaggle/working/BEST_MODEL_{best_pk}_{best_acc:.4f}.pth'
shutil.copy(results[best_pk]['ckpt_path'], best_save_path)

# Delete the other checkpoints to keep output clean
for pk in patch_keys:
    if pk != best_pk:
        try:
            os.remove(results[pk]['ckpt_path'])
            print(f"  🗑️  Deleted {results[pk]['ckpt_path']}")
        except:
            pass

print(f"\n✅ Best weights saved → {best_save_path}")
size_mb = os.path.getsize(best_save_path) / 1e6
print(f"   File size : {size_mb:.1f} MB")
print(f"\nDownload from Kaggle Output tab on the right ➡️")

# ── Classification report ─────────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"Classification Report — {best_pk}")
print('='*65)
print(classification_report(results[best_pk]['y_true'],
                             results[best_pk]['y_pred'],
                             target_names=class_names))