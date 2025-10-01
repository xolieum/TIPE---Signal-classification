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

# === Mapping des labels ===
label_map = {
    'Sleep stage W': 0,
    'Sleep stage 1': 1,
    'Sleep stage 2': 2,
    'Sleep stage 3': 3,
    'Sleep stage 4': 3,
    'Sleep stage R': 4,
}

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

dataset = EEGSleepDataset(x_np, y_np)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# === Modèle CNN ===
class SleepStageCNN(nn.Module):
    def __init__(self, input_channels=1, input_length=WINDOW_SEC*100, num_classes=NUM_CLASSES):
        super().__init__()
        self.conv1 = nn.Conv1d(input_channels, 32, 3, padding=1)
        self.pool1 = nn.MaxPool1d(2, padding=1)
        self.conv2 = nn.Conv1d(32, 64, 3, padding=1)
        self.pool2 = nn.MaxPool1d(2, padding=1)

        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, input_length)
            out = self.pool2(F.relu(self.conv2(self.pool1(F.relu(self.conv1(dummy))))))
            self.flatten_dim = out.view(1, -1).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(self.flatten_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = F.relu(self.conv1(x)); x = self.pool1(x)
        x = F.relu(self.conv2(x)); x = self.pool2(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

model = SleepStageCNN(input_channels=1, input_length=x_np.shape[2], num_classes=NUM_CLASSES)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# === Évaluation ===
def evaluate(model, loader):
    model.eval()
    total_loss, total_samples = 0, 0
    with torch.no_grad():
        for x_batch, y_batch in loader:
            preds = model(x_batch)
            loss = criterion(preds, y_batch)
            total_loss += loss.item() * y_batch.size(0)
            total_samples += y_batch.size(0)
    return total_loss / total_samples

# === Boucle d'entraînement avec wandb ===
train_losses, val_losses = [], []

for epoch in range(EPOCHS):
    model.train()
    total_loss, total_samples = 0, 0
    for x_batch, y_batch in train_loader:
        optimizer.zero_grad()
        preds = model(x_batch)
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
torch.save(model.state_dict(), "sleep_stage_cnn.pth")
wandb.finish()
print("✅ Modèle sauvegardé et logging wandb terminé")
