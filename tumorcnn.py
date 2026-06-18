"""
CNN para Classificação de Tumores Cerebrais — PyTorch
Classificação: glioma | meningioma | pituitary tumor | no tumor
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import os, time

# ── Configurações ──────────────────────────────────────────
DEVICE      = "cpu"
EPOCHS      = 10
BATCH_SIZE  = 32
LR          = 1e-3
NUM_CLASSES = 4
CLASSES     = ["Glioma", "Meningioma", "Pituitary", "No Tumor"]

print(f"Dispositivo: {DEVICE.upper()}\n")

# ── Dataset (sintético para demonstração) ──
class SyntheticMRI(Dataset):
    def __init__(self, n=500):
        self.labels = torch.randint(0, NUM_CLASSES, (n,))
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, i):
        label  = self.labels[i].item()
        img    = torch.randn(1, 224, 224) * 0.1 + 0.2 * (label + 1)
        return img.clamp(0, 1), label

tf_train = transforms.Compose([
    transforms.Grayscale(), transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(), transforms.RandomRotation(15),
    transforms.ToTensor(), transforms.Normalize([0.5], [0.5]),
])
tf_val = transforms.Compose([
    transforms.Grayscale(), transforms.Resize((224, 224)),
    transforms.ToTensor(), transforms.Normalize([0.5], [0.5]),
])

if os.path.isdir("data/Training"):
    print("Dataset real encontrado.")
    train_ds = datasets.ImageFolder("data/Training", tf_train)
    val_ds   = datasets.ImageFolder("data/Testing",   tf_val)
else:
    print("Usando dados sintéticos (demo)\n")
    full = SyntheticMRI(500)
    train_ds, val_ds = random_split(full, [400, 100])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)
print(f"Treino: {len(train_ds)} imagens | Validação: {len(val_ds)} imagens\n")

# ── Modelo ─────────────────────────────────────────────────
def conv_block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1),
        nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.MaxPool2d(2), nn.Dropout2d(0.25),
    )

model = nn.Sequential(
    conv_block(1,   32),   # 224 → 112
    conv_block(32,  64),   # 112 → 56
    conv_block(64, 128),   #  56 → 28
    nn.Flatten(),
    nn.Linear(128 * 28 * 28, 256), nn.ReLU(inplace=True),
    nn.Dropout(0.5),
    nn.Linear(256, NUM_CLASSES),
).to(DEVICE)

print(f"Parâmetros: {sum(p.numel() for p in model.parameters()):,}\n")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# ── Treinamento ────────────────────────────────────────────
history = {"train_loss": [], "val_loss": [], "val_acc": []}

for epoch in range(1, EPOCHS + 1):
    t0 = time.time()

    model.train()
    running = 0.0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(imgs), labels)
        loss.backward()
        optimizer.step()
        running += loss.item() * imgs.size(0)

    model.eval()
    v_loss, correct = 0.0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            out     = model(imgs)
            v_loss += criterion(out, labels).item() * imgs.size(0)
            correct += (out.argmax(1) == labels).sum().item()

    tl = running / len(train_ds)
    vl = v_loss  / len(val_ds)
    va = correct / len(val_ds) * 100
    history["train_loss"].append(tl)
    history["val_loss"].append(vl)
    history["val_acc"].append(va)
    scheduler.step()

    print(f"Época {epoch:02d}/{EPOCHS} | Loss: {tl:.4f} → {vl:.4f} | "
          f"Acurácia: {va:.1f}% | {time.time()-t0:.1f}s")

# ── Gráficos ───────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("CNN — Classificação de Tumores", fontweight="bold")

ax1.plot(history["train_loss"], label="Treino",    marker="o")
ax1.plot(history["val_loss"],   label="Validação", marker="o")
ax1.set(title="Loss", xlabel="Época", ylabel="Loss")
ax1.legend(); ax1.grid(alpha=0.3)

ax2.plot(history["val_acc"], color="green", marker="o")
ax2.set(title="Acurácia de Validação", xlabel="Época", ylabel="%", ylim=(0, 100))
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.show()