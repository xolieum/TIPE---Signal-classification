import mne
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt  # à ajouter en haut du script
import os
from random import randint

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
loss_validation = [] #stocker les résultats de l'évaluiation intra entrainement
epochs = [] #Sotcker le nombre d'épochs
# === Chargement des fichiers ===

data_dir = "/Users/ewen/Desktop/TIPE/BDD/Training-signals"
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
train_loader = DataLoader(dataset, batch_size=128, shuffle=True)


# === CNN pour classification des stades de sommeil ===
class SleepStageCNN(nn.Module):
    def __init__(self, input_channels=1, input_length=3000, num_classes=5):
        super(SleepStageCNN, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool1d(kernel_size=2, padding=1)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2, padding=1)

        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool1d(kernel_size=2, padding=1)

        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, input_length)
            out = self.pool3(F.relu(self.conv4(F.relu(self.conv3(
                self.pool2(F.relu(self.conv2(
                    self.pool1(F.relu(self.conv1(dummy)))
                )))
            )))))
            self.flatten_dim = out.view(1, -1).shape[1]

        self.fc1 = nn.Linear(self.flatten_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.out = nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x)); x = self.pool1(x)
        x = F.relu(self.conv2(x)); x = self.pool2(x)
        x = F.relu(self.conv3(x)); x = F.relu(self.conv4(x)); x = self.pool3(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.out(x)

# === Entraînement ===
model = SleepStageCNN(input_channels=1, input_length=3000, num_classes=5)
#model.load_state_dict(torch.load("sleep_stage_cnn.pth"))
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

def evaluate(model, data_loader, criterion):
    model.eval()
    total_loss = 0
    total_samples = 0
    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)  # CrossEntropyLoss
            total_loss += loss.item() * batch_y.size(0)  # perte totale pondérée
            total_samples += batch_y.size(0)
        print('total_samples:',total_samples)
    return total_loss / total_samples

for cycle in range(5):
    for epoch in range(2):
        model.train()
        total_loss = 0
        total_samples = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_samples += batch_y.size(0)
        avg_loss=total_loss / total_samples
        losses.append(avg_loss)  # stocker la perte
        print(f"Époque {epoch + 1}, Perte: {avg_loss:.4f}")
    model.eval()
    val_total_loss = 0
    val_total_samples = 0
    with torch.no_grad():
        for batch_x, batch_y in train_loader:  # ou juste un échantillon
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            val_total_loss += loss.item()
            val_total_samples += batch_y.size(0)
        
    val_avg_loss = val_total_loss / val_total_samples
    loss_validation.append(val_avg_loss)
    print(f"✅ Validation Loss: {val_avg_loss:.4f}")

# === Affichage de la courbe de perte ===
plt.figure(figsize=(12, 5))

X_train = range(1, len(losses) + 1)
X_val = range(2,len(losses)+1, 2)
Y1, Y2 = losses, loss_validation
plt.plot(X_train, Y1, marker='o', label="Loss")
plt.plot(X_val, Y2, marker='x', label="Validation Loss", color='red')
plt.title("Courbe de perte pendant l'entraînement")
plt.xlabel("Époque")
plt.ylabel("Perte totale")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show(block=True)

#Sauvegarde du modèle après entrainement
torch.save(model.state_dict(), "sleep_stage_cnn.pth")
print("✅ Modèle sauvegardé sous 'sleep_stage_cnn.pth'")