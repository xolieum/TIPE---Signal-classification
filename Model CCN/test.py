import mne
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt  # à ajouter en haut du script
import os
from random import randint
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# === Paramètres ===
DATA_DIR = "./BDD/Training-signals"
WINDOW_SEC = 30
BATCH_SIZE = 128
NUM_CLASSES = 5
EPOCHS = 104
LR = 1e-5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"⚡ Utilisation de {device}")

def list_sleep_edf_files(data_dir):
    psg_file = sorted([
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.endswith("-PSG.edf")
    ])
    
    hypnogram_file = sorted([
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.endswith("-Hypnogram.edf")
    ])
    
    print(f"📄 {len(psg_file)} fichiers PSG trouvés")
    print(f"🧠 {len(hypnogram_file)} fichiers Hypnogramme trouvés")
    
    return psg_file, hypnogram_file

losses = []  # stocker les pertes
# === Chargement des fichiers ===

data_dir = "/Users/ewen/Desktop/TIPE/BDD/Testing-signals"
psg_file, hypnogram_file = list_sleep_edf_files(data_dir)

for psg_path, hypnogram_path in zip(psg_file, hypnogram_file):
    raw = mne.io.read_raw_edf(psg_path, preload=True)
    annotations = mne.read_annotations(hypnogram_path)
    raw.set_annotations(annotations)
    raw.pick_channels(['EEG Fpz-Cz'])

sfreq = int(raw.info["sfreq"])
window_sec = 30
samples_per_window = window_sec * sfreq

# === Conversion des annotations en événements ===
events, event_id = mne.events_from_annotations(raw)
print(f"Nombre d'événements : {len(events)}")
print(f"Event IDs : {event_id}")
print(f"Durée d'enregistrement : {raw.times[-1] / 60:.2f} minutes")

# === Mapping des labels ===
label_map = {
    'Sleep stage W': 0,
    'Sleep stage 1': 1,
    'Sleep stage 2': 2,
    'Sleep stage 3': 3,
    'Sleep stage 4': 3,
    'Sleep stage R': 4,
}

# === Filtrage des fenêtres valides et extraction ===
x_all, y_all = [], []

for psg_path, hypnogram_path in zip(psg_file, hypnogram_file):
    raw = mne.io.read_raw_edf(psg_path, preload=True)
    annotations = mne.read_annotations(hypnogram_path)
    raw.set_annotations(annotations)
    raw.pick_channels(['EEG Fpz-Cz'])

    sfreq = int(raw.info["sfreq"])
    samples_per_window = window_sec * sfreq

    for annot in annotations:
        desc = annot['description']
        label = label_map.get(desc, -1)
        if label == -1:
            continue

        start_sample = int(annot['onset'] * sfreq)
        end_sample = start_sample + samples_per_window

        if end_sample <= raw.n_times:
            segment = raw.get_data(start=start_sample, stop=end_sample)[0]
            if segment.shape[0] == samples_per_window:
                x_all.append(segment)
                y_all.append(label)

print(f"✅ Total de fenêtres extraites : {len(x_all)}")

x_np = np.stack(x_all)
x_np = (x_np - np.mean(x_np)) / np.std(x_np)
x_np = x_np[:, np.newaxis, :]
y_np = np.array(y_all)


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
test_loader = DataLoader(dataset, batch_size=128, shuffle=True)

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
            nn.Dropout(p=0.0),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=0.0),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p=0.0),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = F.relu(self.conv1(x)); x = self.pool1(x)
        x = F.relu(self.conv2(x)); x = self.pool2(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


# === Entraînement ===
model = SleepStageCNN(input_channels=1, input_length=3000, num_classes=5)

# 2. Charger les poids
model.load_state_dict(torch.load("/Users/ewen/Desktop/TIPE---Signal-classification-main/TIPE---Signal-classification/Model CCN/CNN V3/Dropout0/sleep_stage_cnn_dropout0.pth", map_location=device))
model.eval()


# === Prédiction ===
model.eval()
with torch.no_grad():
    test_sample = torch.tensor(x_np[0], dtype=torch.float32).unsqueeze(0)
    pred = torch.argmax(model(test_sample), dim=1).item()
    print(f"✅ Prédiction : {pred} | 🎯 Vrai label : {y_np[0]}")
#torch.save(model.state_dict(), "sleep_stage_cnn.pth")
#print("✅ Modèle sauvegardé sous 'sleep_stage_cnn.pth'")

def evaluate(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            outputs = model(batch_x)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == batch_y).sum().item()
            total += batch_y.size(0)
    return correct / total

# Après entraînement
test_accuracy = evaluate(model, test_loader)
print(f"🎯 Accuracy sur les données de test : {test_accuracy * 100:.2f}%")

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Prédictions sur toutes les données de test
all_preds = []
all_labels = []

model.eval()
with torch.no_grad():
    for batch_x, batch_y in test_loader:
        outputs = model(batch_x)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.numpy())
        all_labels.extend(batch_y.numpy())


# Matrice de confusion
cm = confusion_matrix(all_labels,all_preds)

# 🔹 Row-wise normalization
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

print("Raw confusion matrix:\n", cm)
print("\nRow-wise normalized confusion matrix:\n", cm_normalized)

disp = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=[
    "W", "N1", "N2", "N3", "R"
])
disp.plot(cmap=plt.cm.Blues, values_format=".2f")
plt.title("Matrice de confusion - Données de test")
plt.show()

print(f"all_preds : {all_preds}\nall_labels : {all_labels}")