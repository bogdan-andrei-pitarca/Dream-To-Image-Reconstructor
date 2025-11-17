"""
case_study_eeg_image.py
A compact case-study implementation that:
 - generates a tiny synthetic image dataset (shapes)
 - generates small synthetic EEG signals conditioned on images
 - trains a small encoder-decoder (PyTorch) to reconstruct images from EEG
 - saves example reconstructions and a zip archive of results

Note: This is designed to run quickly on a CPU (or faster on a GPU).
Adjust SAMPLES_PER_CAT, FS, DURATION, EPOCHS for longer/more realistic runs.
"""

import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from skimage.draw import disk, rectangle
from skimage.metrics import structural_similarity as ssim
from scipy.ndimage import gaussian_filter
import zipfile
import json

# -----------------------
# Config (change as needed)
# -----------------------
OUT = Path("case_study_results")
OUT.mkdir(parents=True, exist_ok=True)

IMG_SIZE = 64
CATEGORIES = ["circle", "square", "triangle"]
SAMPLES_PER_CAT = 200        # small for a quick demo (increase for larger experiments)
EEG_CHANNELS = 8
FS = 128                   # sampling frequency (Hz)
DURATION = 0.5             # seconds
SNR_DB = 10
EMB_DIM = 16
BATCH_SIZE = 8
EPOCHS = 25
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------
# Utility: make toy images
# -----------------------
def make_shape_image(category, size=IMG_SIZE):
    img = np.zeros((size, size), dtype=np.float32)
    if category == "circle":
        rr, cc = disk((size//2, size//2), radius=size//4, shape=img.shape)
        img[rr, cc] = 1.0
    elif category == "square":
        start = size//4
        extent = size//2
        rr, cc = rectangle(start=(start, start), extent=(extent, extent), shape=img.shape)
        img[rr, cc] = 1.0
    elif category == "triangle":
        for r in range(size//2):
            img[size//2 + r, size//2 - r:size//2 + r] = 1.0
    img = gaussian_filter(img, sigma=1.2)
    return img

# -----------------------
# Dataset creation
# -----------------------
images = []
labels = []
for cat in CATEGORIES:
    for _ in range(SAMPLES_PER_CAT):
        img = make_shape_image(cat)
        img = img + np.random.randn(*img.shape) * 0.02
        img = np.clip(img, 0.0, 1.0)
        images.append(img)
        labels.append(cat)
images = np.stack(images)    # shape: (N, H, W)
labels = np.array(labels)

# Save a small montage (for the report)
def save_montage(images, path, rows=3, cols=5):
    fig, axs = plt.subplots(rows, cols, figsize=(cols*1.6, rows*1.2))
    idx = 0
    for r in range(rows):
        for c in range(cols):
            if idx < images.shape[0]:
                axs[r,c].imshow(images[idx], cmap='gray'); axs[r,c].axis('off')
            else:
                axs[r,c].axis('off')
            idx += 1
    plt.suptitle("Dataset examples")
    plt.savefig(path, bbox_inches='tight'); plt.close(fig)

save_montage(images, OUT / "dataset_examples.png")

# -----------------------
# Simple image->embedding projector
# -----------------------
def image_to_embedding(img, emb_dim=EMB_DIM):
    # downsample and linear projection (deterministic random weights)
    small = torch.tensor(img).unsqueeze(0).unsqueeze(0)  # 1x1xHxW
    small = nn.functional.avg_pool2d(small, kernel_size=8)  # down to 8x8
    vec = small.view(-1).numpy()
    rng = np.random.RandomState(12345)
    W = rng.randn(emb_dim, vec.size) * 0.1
    emb = W @ vec
    return emb.astype(np.float32)

# -----------------------
# Synthetic EEG generator
# -----------------------
def generate_synthetic_eeg_from_embedding(emb, N_channels=EEG_CHANNELS, duration=DURATION, fs=FS, snr_db=SNR_DB):
    t = np.arange(0, duration, 1/fs)
    L = len(t)
    eeg = np.zeros((N_channels, L), dtype=np.float32)
    bands = np.array([2, 6, 10, 20], dtype=np.float32)
    rng = np.random.RandomState(42)
    W = rng.randn(len(bands)*N_channels, emb.size) * 0.05
    amps = (W @ emb).reshape(N_channels, len(bands))
    for ch in range(N_channels):
        signal = np.zeros(L)
        for b, f in enumerate(bands):
            A = np.abs(amps[ch, b]) + 0.05
            phase = rng.rand() * 2*np.pi
            signal += A * np.sin(2*np.pi*f*t + phase)
        # sparse artifact spikes
        mask = rng.rand(L) < 0.025
        signal[mask] += rng.randn(mask.sum()) * 3.0
        # add noise
        power = np.mean(signal**2) + 1e-9
        noise_power = power / (10**(snr_db/10))
        noise = rng.randn(L) * np.sqrt(noise_power)
        eeg[ch] = signal + noise
    return eeg

# Create EEGs conditioned on images
eegs = []
embs = []
for i in range(len(images)):
    emb = image_to_embedding(images[i], emb_dim=EMB_DIM)
    eeg = generate_synthetic_eeg_from_embedding(emb)
    eegs.append(eeg)
    embs.append(emb)
eegs = np.stack(eegs)
embs = np.stack(embs)

# save a tiny EEG snippet plot
fig, ax = plt.subplots(3,1, figsize=(8,3))
for ch in range(3):
    ax[ch].plot(eegs[0,ch,:100]); ax[ch].set_ylabel(f"ch{ch}")
ax[-1].set_xlabel("samples")
plt.suptitle("Example EEG snippet")
plt.savefig(OUT / "example_eeg.png"); plt.close(fig)

# -----------------------
# Dataset class and loaders
# -----------------------
class EEGImageDataset(Dataset):
    def __init__(self, eegs, images):
        self.eegs = torch.tensor(eegs)
        self.images = torch.tensor(images)
    def __len__(self):
        return len(self.eegs)
    def __getitem__(self, idx):
        x = self.eegs[idx]
        # simple per-channel normalization
        x = (x - x.mean(dim=1, keepdim=True)) / (x.std(dim=1, keepdim=True) + 1e-6)
        y = self.images[idx].unsqueeze(0)
        return x.float(), y.float()

dataset = EEGImageDataset(eegs, images)
n = len(dataset)
idxs = np.arange(n)
rng = np.random.RandomState(0); rng.shuffle(idxs)
train_idx = idxs[:int(0.7*n)]
val_idx = idxs[int(0.7*n):int(0.85*n)]
test_idx = idxs[int(0.85*n):]

train_loader = DataLoader(Subset(dataset, train_idx), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(Subset(dataset, val_idx), batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(Subset(dataset, test_idx), batch_size=BATCH_SIZE, shuffle=False)

# -----------------------
# Model (encoder 1D -> latent, decoder -> 64x64 image)
# -----------------------
class EEGEncoder(nn.Module):
    def __init__(self, in_ch=EEG_CHANNELS, latent_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, 32, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 64, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(64, latent_dim)
    def forward(self, x):
        h = self.net(x).view(x.size(0), -1)
        return self.fc(h)

class ImageDecoder(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 128*8*8)
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(128,128,4,2,1),  # 8 -> 16
            nn.ReLU(),
            nn.ConvTranspose2d(128,64,4,2,1),   # 16 -> 32
            nn.ReLU(),
            nn.ConvTranspose2d(64,32,4,2,1),    # 32 -> 64
            nn.ReLU(),
            nn.Conv2d(32,1,3,1,1),
            nn.Sigmoid()
        )
    def forward(self, z):
        h = self.fc(z).view(-1,128,8,8)
        return self.conv(h)

enc = EEGEncoder().to(DEVICE)
dec = ImageDecoder().to(DEVICE)

# -----------------------
# Training utilities
# -----------------------
optimizer = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=1e-3)
mse = nn.MSELoss()

def evaluate_on_loader(loader):
    enc.eval(); dec.eval()
    losses = []
    ssim_vals = []
    with torch.no_grad():
        for x,y in loader:
            x = x.to(DEVICE); y = y.to(DEVICE)
            z = enc(x); out = dec(z)
            losses.append(mse(out,y).item())
            out_np = out.cpu().numpy(); y_np = y.cpu().numpy()
            for i in range(out_np.shape[0]):
                a = out_np[i,0]; b = y_np[i,0]
                try:
                    s = ssim(a, b, data_range=1.0)
                except Exception:
                    s = 0.0
                ssim_vals.append(s)
    return float(np.mean(losses)), float(np.mean(ssim_vals))

train_losses, val_losses, val_ssim = [], [], []
for ep in range(EPOCHS):
    enc.train(); dec.train()
    batch_losses = []
    for x,y in train_loader:
        x = x.to(DEVICE); y = y.to(DEVICE)
        z = enc(x); out = dec(z)
        loss = mse(out, y)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        batch_losses.append(loss.item())
    train_losses.append(float(np.mean(batch_losses)))
    vl, vs = evaluate_on_loader(val_loader)
    val_losses.append(vl); val_ssim.append(vs)
    print(f"Epoch {ep+1}/{EPOCHS} TrainLoss={train_losses[-1]:.4f} ValLoss={vl:.4f} ValSSIM={vs:.4f}")

# Save training log
with open(OUT/"train_log.json", "w") as f:
    json.dump({"train_losses": train_losses, "val_losses": val_losses, "val_ssim": val_ssim}, f)

# -----------------------
# Test, save example reconstructions and models
# -----------------------
test_mse, test_ssim = evaluate_on_loader(test_loader)
print("Test MSE:", test_mse, "Test SSIM:", test_ssim)

os.makedirs(OUT/"recon_examples", exist_ok=True)
cnt = 0
for x,y in test_loader:
    x = x.to(DEVICE)
    with torch.no_grad():
        z = enc(x); out = dec(z).cpu().numpy()
    y_np = y.numpy()
    for i in range(out.shape[0]):
        gt = y_np[i,0]; rec = out[i,0]
        fig, ax = plt.subplots(1,2, figsize=(4,2))
        ax[0].imshow(gt, cmap='gray'); ax[0].axis('off'); ax[0].set_title("GT")
        ax[1].imshow(rec, cmap='gray'); ax[1].axis('off'); ax[1].set_title("Rec")
        try:
            s = ssim(gt, rec, data_range=1.0)
        except:
            s = 0.0
        plt.suptitle(f"SSIM={s:.3f}")
        plt.savefig(OUT/f"recon_examples/example_{cnt:03d}.png", bbox_inches='tight', dpi=120)
        plt.close(fig)
        cnt += 1

torch.save(enc.state_dict(), OUT/"encoder.pth")
torch.save(dec.state_dict(), OUT/"decoder.pth")

# Zip the results folder
zip_path = OUT / "case_study_code_and_results.zip"
with zipfile.ZipFile(zip_path, "w") as zf:
    for p in OUT.glob("**/*"):
        if p.is_file():
            zf.write(p, p.relative_to(OUT))

print("Done. Results saved to:", OUT)
print("Zip file:", zip_path)
