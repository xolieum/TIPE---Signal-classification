import os
import numpy as np
import mne
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
import wandb

# === Paramètres ===
DATA_DIR = "/Users/ewen/Desktop/TIPE/BDD/Training-signals"
WINDOW_SEC = 30
BATCH_SIZE = 128
NUM_CLASSES = 5
EPOCHS = 100
LR = 1e-5
sfreq = 100
# === Initialisation wandb ===
wandb.init(
    project="sleep_stage_classification",
    config={
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LR,
        "window_sec": WINDOW_SEC,
        "num_classes": NUM_CLASSES,
    }
)
config = wandb.config

# === Liste des fichiers EDF ===
def list_sleep_edf_files(data_dir):
    psg_files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith("-PSG.edf")])
    hyp_files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith("-Hypnogram.edf")])
    print(f"📄 {len(psg_files)} fichiers PSG trouvés")
    print(f"🧠 {len(hyp_files)} fichiers Hypnogramme trouvés")
    return psg_files, hyp_files

#Fonction du filtre passe bande
def bandpass_filter(data, sfreq, l_freq, h_freq):
    return mne.filter.filter_data(data, sfreq=sfreq, l_freq=l_freq, h_freq=h_freq, verbose=False)

# === Mapping des labels ===
label_map = {
    'Sleep stage W': 0,
    'Sleep stage 1': 1,
    'Sleep stage 2': 2,
    'Sleep stage 3': 3,
    'Sleep stage 4': 3,
    'Sleep stage R': 4,
}

bands = {
    'Beta'  : (12,30),
    'Alpha' : (8,12),
    'Theta' : (4,8),
    'Delta' : (0.5,4),
}

band_names = list(bands.keys())

def extract_band_signals(x_np, sfreq, bands):
    band_signals = []
    for band, (low, high) in bands.items():
        filtered = np.array([bandpass_filter(sig, sfreq, low, high) for sig in x_np[:,0,:]])
        band_signals.append(filtered[:, np.newaxis, :])  # shape: (N, 1, L)
    return band_signals

class MultiBandEEGDataset(Dataset):
    def __init__(self, band_signals, y, band_names):
        self.band_signals = band_signals  # liste de np.ndarray
        self.y = torch.tensor(y, dtype=torch.long)
        self.band_names = band_names

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x_dict = {
            band: torch.tensor(self.band_signals[i][idx], dtype=torch.float32)
            for i, band in enumerate(self.band_names)
        }
        return x_dict, self.y[idx]
    
# === Extraction des fenêtres ===
def extract_windows(psg_files, hyp_files, window_sec):
    x_all, y_all = [], []
    for psg_path, hyp_path in zip(psg_files, hyp_files):
        raw = mne.io.read_raw_edf(psg_path, preload=True)
        raw.pick_channels(['EEG Fpz-Cz'])
        sfreq = int(raw.info['sfreq'])
        samples_per_window = window_sec * sfreq

        annotations = mne.read_annotations(hyp_path)
        raw.set_annotations(annotations)

        for annot in annotations:
            label = label_map.get(annot['description'], -1)
            if label == -1:
                continue
            start = int(annot['onset'] * sfreq)
            end = start + samples_per_window
            if end <= raw.n_times:
                segment = raw.get_data(start=start, stop=end)[0]
                if len(segment) == samples_per_window:
                    x_all.append(segment)
                    y_all.append(label)

    x_np = np.stack(x_all)
    x_np = (x_np - np.mean(x_np, axis=1, keepdims=True)) / (np.std(x_np, axis=1, keepdims=True) + 1e-8)
    x_np = x_np[:, np.newaxis, :]
    y_np = np.array(y_all)
    print(f"✅ Total de fenêtres extraites : {len(x_np)}")
    return x_np, y_np

psg_files, hyp_files = list_sleep_edf_files(DATA_DIR)
if os.path.exists("x_data.npy") and os.path.exists("y_data.npy"):
    x_np = np.load("x_data.npy", mmap_mode='r')
    y_np = np.load("y_data.npy", mmap_mode='r')
else :
    x_np, y_np = extract_windows(psg_files, hyp_files, WINDOW_SEC)

# === Dataset PyTorch ===
class EEGSleepDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):
        return len(self.x)
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

band_signals = extract_band_signals(x_np, sfreq, bands)
dataset = MultiBandEEGDataset(band_signals, y_np, band_names)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# === Modèle CNN ===
class MultiBandCNN(nn.Module):
    def __init__(self, input_length, num_classes=NUM_CLASSES, bands=band_names):
        super().__init__()
        self.branches = nn.ModuleDict({
            band: nn.Sequential(
                nn.Conv1d(1, 32, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Conv1d(32, 64, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool1d(2),
            ) for band in bands
        })
        # Calcule la dimension finale
        with torch.no_grad():
            dummy = torch.zeros(1, 1, input_length)
            out = self.branches[bands[0]](dummy)
            flatten_dim = out.view(1, -1).shape[1] * len(bands)
        self.fc = nn.Sequential(
            nn.Linear(flatten_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x_dict):
        features = []
        for band, x in x_dict.items():
            out = self.branches[band](x)
            features.append(out.view(out.size(0), -1))
        x_cat = torch.cat(features, dim=1)
        return self.fc(x_cat)

    
model = MultiBandCNN(input_length=x_np.shape[2], num_classes=NUM_CLASSES, bands=band_names)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# === Évaluation ===
def evaluate(model, loader):
    model.eval()
    total_loss, total_samples = 0, 0
    with torch.no_grad():
        for x_dict, y_batch in loader:
            preds = model(x_dict)
            loss = criterion(preds, y_batch)
            total_loss += loss.item() * y_batch.size(0)
            total_samples += y_batch.size(0)
    return total_loss / total_samples

# === Boucle d'entraînement avec wandb ===
train_losses, val_losses = [], []

for epoch in range(EPOCHS):
    model.train()
    total_loss, total_samples = 0, 0
    for x_dict, y_batch in train_loader:
        optimizer.zero_grad()
        preds = model(x_dict)
        loss = criterion(preds, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * y_batch.size(0)
        total_samples += y_batch.size(0)
    train_loss = total_loss / total_samples
    val_loss = evaluate(model, val_loader)
    
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    
    # Log dans wandb
    wandb.log({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "val_loss": val_loss
    })
    
    print(f"Époque {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

# === Courbes locales ===
plt.figure(figsize=(10,5))
plt.plot(range(1, EPOCHS+1), train_losses, marker='o', label='Train Loss')
plt.plot(range(1, EPOCHS+1), val_losses, marker='x', color='red', label='Validation Loss')
plt.xlabel('Époque')
plt.ylabel('Loss')
plt.title('Courbe de perte')
plt.grid(True)
plt.legend()
plt.show()

# === Sauvegarde du modèle et fin de run ===
torch.save(model.state_dict(), "sleep_stage_cnn_multyband.pth")
wandb.finish()
print("✅ Modèle sauvegardé et logging wandb terminé")
