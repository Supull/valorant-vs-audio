"""
VALORANT SOUND RADAR — Phase 2 Training Script
================================================
Trains a small CNN to classify audio clips as
footstep, gunshot, or silence using spectrograms.

Requirements:
  pip3 install torch torchvision torchaudio soundfile

Run: python3 train.py
Output: model.pth (trained model weights)
"""

import os
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

# ─── CONFIG ───────────────────────────────────────────────
SAMPLE_RATE  = 44100
CLASSES      = ["footstep", "gunshot", "silence"]
N_MELS       = 64          # spectrogram frequency resolution
N_FFT        = 1024
HOP_LENGTH   = 256
BATCH_SIZE   = 16
EPOCHS       = 40
LEARNING_RATE = 0.003
MODEL_PATH   = "model.pth"
# ──────────────────────────────────────────────────────────


def wav_to_melspec(filepath):
    """Convert a wav file to a mel spectrogram (2D array)"""
    audio, sr = sf.read(filepath)

    # Convert to mono if stereo
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    audio = torch.tensor(audio, dtype=torch.float32)

    # Resample if needed
    if sr != SAMPLE_RATE:
        import torchaudio
        audio = torchaudio.functional.resample(audio, sr, SAMPLE_RATE)

    import torchaudio
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
    )
    mel = mel_transform(audio)

    # Convert to dB scale (log)
    mel_db = torchaudio.transforms.AmplitudeToDB()(mel)

    return mel_db


class AudioDataset(Dataset):
    def __init__(self, data_dir="data"):
        self.samples = []
        self.target_frames = None

        # First pass — find target size and collect file paths
        for label_idx, label in enumerate(CLASSES):
            folder = os.path.join(data_dir, label)
            if not os.path.isdir(folder):
                continue
            for fname in sorted(os.listdir(folder)):
                if fname.endswith(".wav"):
                    self.samples.append((os.path.join(folder, fname), label_idx))

        print(f"Found {len(self.samples)} total clips")

        # Precompute spectrograms and find common shape
        self.specs  = []
        self.labels = []
        shapes = []

        for filepath, label_idx in self.samples:
            try:
                mel = wav_to_melspec(filepath)
                shapes.append(mel.shape)
                self.specs.append(mel)
                self.labels.append(label_idx)
            except Exception as e:
                print(f"Skipping {filepath}: {e}")

        # Use the most common time dimension, pad/crop others to match
        time_dims = [s[1] for s in shapes]
        target_time = int(np.median(time_dims))
        self.target_frames = target_time
        print(f"Target spectrogram shape: ({N_MELS}, {target_time})")

        # Pad or crop all specs to target shape
        fixed_specs = []
        for mel in self.specs:
            mel = self._fix_size(mel, target_time)
            fixed_specs.append(mel)
        self.specs = fixed_specs

        # Normalize each spectrogram individually (zero mean, unit std)
        normalized_specs = []
        for mel in self.specs:
            mean = mel.mean()
            std  = mel.std()
            if std > 1e-6:
                mel = (mel - mean) / std
            else:
                mel = mel - mean
            normalized_specs.append(mel)
        self.specs = normalized_specs

    def _fix_size(self, mel, target_time):
        current_time = mel.shape[1]
        if current_time == target_time:
            return mel
        elif current_time > target_time:
            return mel[:, :target_time]
        else:
            pad_amount = target_time - current_time
            return torch.nn.functional.pad(mel, (0, pad_amount))

    def __len__(self):
        return len(self.specs)

    def __getitem__(self, idx):
        spec  = self.specs[idx].unsqueeze(0)  # add channel dim -> (1, n_mels, time)
        label = self.labels[idx]
        return spec, label


class SoundCNN(nn.Module):
    """Small CNN for classifying audio spectrograms"""
    def __init__(self, n_classes=3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, n_classes),
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x


def train():
    print("=" * 55)
    print("  VALORANT SOUND RADAR — Phase 2 Training")
    print("=" * 55)

    device = torch.device("cuda" if torch.cuda.is_available()
                           else "mps" if torch.backends.mps.is_available()
                           else "cpu")
    print(f"Using device: {device}\n")

    # Load dataset
    dataset = AudioDataset("data")

    if len(dataset) < 30:
        print("\nNot enough data — need at least 30 clips total.")
        return

    # Split train/val
    val_size   = max(1, int(len(dataset) * 0.2))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    print(f"Train samples: {train_size}")
    print(f"Val samples:   {val_size}\n")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    # Compute class weights to handle imbalance
    from collections import Counter
    label_counts = Counter(dataset.labels)
    total = sum(label_counts.values())
    class_weights = torch.tensor(
        [total / (len(CLASSES) * label_counts[i]) for i in range(len(CLASSES))],
        dtype=torch.float32
    ).to(device)
    print(f"Class counts: {dict(label_counts)}")
    print(f"Class weights: {class_weights.tolist()}\n")

    # Model
    model     = SoundCNN(n_classes=len(CLASSES)).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    best_val_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for specs, labels in train_loader:
            specs, labels = specs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(specs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * specs.size(0)
            _, predicted = torch.max(outputs, 1)
            train_correct += (predicted == labels).sum().item()
            train_total += labels.size(0)

        train_acc = train_correct / train_total
        train_loss = train_loss / train_total

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for specs, labels in val_loader:
                specs, labels = specs.to(device), labels.to(device)
                outputs = model(specs)
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total if val_total > 0 else 0

        scheduler.step(val_acc)
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch:2d}/{EPOCHS}  "
              f"train_loss={train_loss:.4f}  "
              f"train_acc={train_acc:.2%}  "
              f"val_acc={val_acc:.2%}  "
              f"lr={current_lr:.5f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'target_frames': dataset.target_frames,
                'n_mels': N_MELS,
                'classes': CLASSES,
                'sample_rate': SAMPLE_RATE,
                'n_fft': N_FFT,
                'hop_length': HOP_LENGTH,
            }, MODEL_PATH)
            print(f"  -> New best model saved (val_acc={val_acc:.2%})")

    print(f"\nTraining complete!")
    print(f"Best validation accuracy: {best_val_acc:.2%}")
    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    train()