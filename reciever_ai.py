"""
VALORANT SOUND RADAR — Mac Receiver v4 (AI Powered)
====================================================
Uses a trained CNN (model.pth) to classify audio as
footstep, gunshot, or silence in real time.

Requirements:
  pip3 install numpy pygame torch torchaudio soundfile

Run: python3 receiver_ai.py
"""

import socket
import numpy as np
import pygame
import threading
import math
import time
import torch
import torch.nn as nn
import torchaudio
from collections import deque

# ─── CONFIG ───────────────────────────────────────────────
PORT          = 5005
SAMPLE_RATE   = 44100
CHANNELS      = 2
CLIP_DURATION = 0.5
BUFFER_SAMPLES = int(SAMPLE_RATE * CLIP_DURATION)
RING_SIZE     = SAMPLE_RATE * 3

MODEL_PATH    = "model.pth"

INFERENCE_INTERVAL   = 0.1   # run model 10x per second
CONFIDENCE_THRESHOLD = 0.6
# ──────────────────────────────────────────────────────────

# ─── DISPLAY ───────────────────────────────────────────────
WIDTH        = 700
HEIGHT       = 700
CENTER       = (WIDTH // 2, HEIGHT // 2)
RADAR_RADIUS = 270
FPS          = 60
# ──────────────────────────────────────────────────────────

# ─── COLORS ────────────────────────────────────────────────
DARK_GREEN  = (0,   18,  5)
GREEN       = (0,   255, 70)
GREEN_DIM   = (0,   55,  18)
GRID_COLOR  = (0,   38,  12)
RED         = (255, 60,  60)
YELLOW      = (255, 220, 50)
BLUE        = (50,  150, 255)
WHITE       = (255, 255, 255)
ORANGE      = (255, 140, 0)
DIM         = (80,  80,  80)
# ──────────────────────────────────────────────────────────


class SoundCNN(nn.Module):
    """Must match architecture in train.py"""
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


print("Loading model...")
checkpoint    = torch.load(MODEL_PATH, map_location="cpu")
CLASSES       = checkpoint['classes']
TARGET_FRAMES = checkpoint['target_frames']
N_MELS        = checkpoint['n_mels']
N_FFT         = checkpoint['n_fft']
HOP_LENGTH    = checkpoint['hop_length']

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model  = SoundCNN(n_classes=len(CLASSES))
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    n_mels=N_MELS,
)
db_transform = torchaudio.transforms.AmplitudeToDB()

print(f"Model loaded — classes: {CLASSES}")
print(f"Device: {device}\n")


def classify_clip(clip):
    mono = clip.mean(axis=1)
    audio_tensor = torch.tensor(mono, dtype=torch.float32)

    mel    = mel_transform(audio_tensor)
    mel_db = db_transform(mel)

    current_time = mel_db.shape[1]
    if current_time > TARGET_FRAMES:
        mel_db = mel_db[:, :TARGET_FRAMES]
    elif current_time < TARGET_FRAMES:
        pad = TARGET_FRAMES - current_time
        mel_db = torch.nn.functional.pad(mel_db, (0, pad))

    mean = mel_db.mean()
    std  = mel_db.std()
    if std > 1e-6:
        mel_db = (mel_db - mean) / std
    else:
        mel_db = mel_db - mean

    x = mel_db.unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs  = torch.softmax(logits, dim=1)[0]

    pred_idx   = int(torch.argmax(probs))
    pred_class = CLASSES[pred_idx]
    confidence = float(probs[pred_idx])

    return pred_class, confidence, probs.cpu().numpy()


# ─── RING BUFFER ─────────────────────────────────────────────
ring_buffer   = np.zeros((RING_SIZE, CHANNELS), dtype=np.float32)
write_pos     = 0
total_written = 0
buffer_lock   = threading.Lock()
# ──────────────────────────────────────────────────────────

# ─── STATE ─────────────────────────────────────────────────
pulses               = []
sweep_angle          = 0.0
last_audio_time      = 0
signal_log           = deque(maxlen=8)
last_inference_time  = 0
last_probs           = np.array([0.33, 0.33, 0.33])
last_prediction      = "silence"
# ──────────────────────────────────────────────────────────


class Pulse:
    def __init__(self, angle, radius, color, size, label, energy):
        self.angle    = angle
        self.radius   = radius
        self.color    = color
        self.size     = size
        self.label    = label
        self.energy   = energy
        self.alpha    = 255
        self.birth    = time.time()
        self.lifetime = 3.5

    def update(self):
        age        = time.time() - self.birth
        self.alpha = max(0, int(255 * (1 - age / self.lifetime)))
        return self.alpha > 0

    def draw(self, surface):
        if self.alpha <= 0:
            return
        r, g, b = self.color
        sz = self.size
        s  = pygame.Surface((sz * 2 + 10, sz * 2 + 10), pygame.SRCALPHA)
        pygame.draw.circle(s, (r, g, b, self.alpha // 4), (sz + 5, sz + 5), sz + 4)
        pygame.draw.circle(s, (r, g, b, self.alpha),      (sz + 5, sz + 5), sz)
        pygame.draw.circle(s, (255, 255, 255, self.alpha // 2),
                           (sz + 5, sz + 5), max(1, sz // 3))
        rad = math.radians(self.angle)
        x   = int(CENTER[0] + self.radius * math.sin(rad))
        y   = int(CENTER[1] - self.radius * math.cos(rad))
        surface.blit(s, (x - sz - 5, y - sz - 5))


def get_band_energy(fft_mag, freqs, low, high):
    mask = (freqs >= low) & (freqs <= high)
    return float(np.mean(fft_mag[mask])) if np.any(mask) else 0.0


def detect_direction(left_ch, right_ch):
    left_e  = float(np.mean(np.abs(left_ch)))
    right_e = float(np.mean(np.abs(right_ch)))
    total_e = left_e + right_e
    if total_e < 1e-6:
        return None, 0.0

    lr_ratio = (right_e - left_e) / total_e

    mono     = (left_ch + right_ch) / 2.0
    window   = np.hanning(len(mono))
    fft_vals = np.fft.rfft(mono * window)
    fft_mag  = np.abs(fft_vals)
    freqs    = np.fft.rfftfreq(len(mono), 1.0 / SAMPLE_RATE)

    front_e  = get_band_energy(fft_mag, freqs, 4000, 8000)
    back_e   = get_band_energy(fft_mag, freqs, 8000, 16000)
    fb_total = front_e + back_e
    fb_ratio = (front_e - back_e) / fb_total if fb_total > 1e-8 else 0.0

    if abs(lr_ratio) > 0.1:
        base  = 90 if lr_ratio > 0 else 270
        angle = base - fb_ratio * 45
    else:
        angle = 0 if fb_ratio >= 0 else 180

    confidence = min(1.0, total_e * 20)
    return angle % 360, confidence


def udp_receiver():
    global write_pos, total_written, last_audio_time

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", PORT))
    sock.settimeout(1.0)
    print(f"Listening on port {PORT}...")

    while True:
        try:
            data, _ = sock.recvfrom(65507)
            audio = np.frombuffer(data, dtype=np.float32)
            audio = audio[:len(audio) - len(audio) % CHANNELS]
            if len(audio) == 0:
                continue
            audio = audio.reshape(-1, CHANNELS)
            n = len(audio)

            with buffer_lock:
                end_pos = write_pos + n
                if end_pos <= RING_SIZE:
                    ring_buffer[write_pos:end_pos] = audio
                else:
                    first_part  = RING_SIZE - write_pos
                    second_part = n - first_part
                    ring_buffer[write_pos:RING_SIZE] = audio[:first_part]
                    ring_buffer[0:second_part]       = audio[first_part:]
                write_pos = (write_pos + n) % RING_SIZE
                total_written += n

            last_audio_time = time.time()

        except socket.timeout:
            continue
        except Exception:
            continue


def get_last_clip():
    with buffer_lock:
        if total_written < BUFFER_SAMPLES:
            return None
        pos = write_pos
        if pos >= BUFFER_SAMPLES:
            clip = ring_buffer[pos - BUFFER_SAMPLES:pos].copy()
        else:
            part1 = ring_buffer[RING_SIZE - (BUFFER_SAMPLES - pos):RING_SIZE]
            part2 = ring_buffer[0:pos]
            clip = np.concatenate([part1, part2], axis=0)
        return clip


def run_inference():
    global last_probs, last_prediction

    clip = get_last_clip()
    if clip is None:
        return []

    pred_class, confidence, probs = classify_clip(clip)
    last_probs      = probs
    last_prediction = pred_class

    if confidence < CONFIDENCE_THRESHOLD:
        return []
    if pred_class == "silence":
        return []

    left_ch  = clip[:, 0]
    right_ch = clip[:, 1]
    energy   = float(np.abs(clip).mean())

    angle, conf = detect_direction(left_ch, right_ch)
    if angle is None or conf < 0.05:
        return []

    now = time.strftime('%H:%M:%S')

    if pred_class == "footstep":
        radius  = max(50, min(RADAR_RADIUS - 30,
                      int(RADAR_RADIUS * (1 - min(1, energy * 80)))))
        size    = max(5, min(16, int(energy * 800)))
        dir_str = "L" if 180 < angle <= 360 else "R" if 0 < angle <= 180 else "C"
        log = f"{now}  FOOTSTEP  conf={confidence:.2f}  {dir_str}  {angle:.0f}deg"
        signal_log.appendleft(log)
        print(f"  >>> FOOTSTEP  conf={confidence:.2f}  energy={energy:.4f}  dir={dir_str}  angle={angle:.0f}deg")
        return [Pulse(angle, radius, RED, size, "footstep", energy)]

    elif pred_class == "gunshot":
        radius = max(50, min(RADAR_RADIUS - 30,
                     int(RADAR_RADIUS * (1 - min(1, energy * 15)))))
        size   = max(6, min(18, int(energy * 200)))
        log = f"{now}  GUNSHOT   conf={confidence:.2f}"
        signal_log.appendleft(log)
        print(f"  >>> GUNSHOT   conf={confidence:.2f}  energy={energy:.4f}")
        return [Pulse(angle, radius, YELLOW, size, "gunshot", energy)]

    return []


def draw_radar_bg(surface):
    surface.fill(DARK_GREEN)
    for r in range(RADAR_RADIUS // 4, RADAR_RADIUS + 1, RADAR_RADIUS // 4):
        pygame.draw.circle(surface, GRID_COLOR, CENTER, r, 1)
    pygame.draw.line(surface, GRID_COLOR,
                     (CENTER[0], CENTER[1] - RADAR_RADIUS),
                     (CENTER[0], CENTER[1] + RADAR_RADIUS), 1)
    pygame.draw.line(surface, GRID_COLOR,
                     (CENTER[0] - RADAR_RADIUS, CENTER[1]),
                     (CENTER[0] + RADAR_RADIUS, CENTER[1]), 1)
    off = int(RADAR_RADIUS * 0.707)
    pygame.draw.line(surface, GRID_COLOR,
                     (CENTER[0] - off, CENTER[1] - off),
                     (CENTER[0] + off, CENTER[1] + off), 1)
    pygame.draw.line(surface, GRID_COLOR,
                     (CENTER[0] + off, CENTER[1] - off),
                     (CENTER[0] - off, CENTER[1] + off), 1)
    pygame.draw.circle(surface, GREEN_DIM, CENTER, RADAR_RADIUS, 2)


def draw_sweep(surface, angle_deg):
    surf  = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    trail = 65
    steps = 35
    for i in range(steps):
        a   = angle_deg - (trail * i / steps)
        alp = int(90 * (1 - i / steps))
        rad = math.radians(a)
        ex  = int(CENTER[0] + RADAR_RADIUS * math.sin(rad))
        ey  = int(CENTER[1] - RADAR_RADIUS * math.cos(rad))
        pygame.draw.line(surf, (0, 255, 70, alp), CENTER, (ex, ey), 2)
    rad = math.radians(angle_deg)
    ex  = int(CENTER[0] + RADAR_RADIUS * math.sin(rad))
    ey  = int(CENTER[1] - RADAR_RADIUS * math.cos(rad))
    pygame.draw.line(surf, (0, 255, 70, 220), CENTER, (ex, ey), 2)
    surface.blit(surf, (0, 0))


def draw_labels(surface, fs, ft):
    p = 20
    for text, pos in [
        ("N", (CENTER[0] - 6, CENTER[1] - RADAR_RADIUS - p)),
        ("S", (CENTER[0] - 6, CENTER[1] + RADAR_RADIUS + 5)),
        ("E", (CENTER[0] + RADAR_RADIUS + 7, CENTER[1] - 8)),
        ("W", (CENTER[0] - RADAR_RADIUS - p, CENTER[1] - 8)),
    ]:
        surface.blit(fs.render(text, True, GREEN_DIM), pos)
    you = ft.render("YOU", True, WHITE)
    surface.blit(you, (CENTER[0] - you.get_width() // 2, CENTER[1] + 10))
    pygame.draw.circle(surface, WHITE, CENTER, 5)
    pygame.draw.circle(surface, GREEN, CENTER, 8, 1)


def draw_sidebar(surface, ft, ftiny, connected):
    sx = WIDTH - 180
    pygame.draw.line(surface, GREEN_DIM, (sx - 5, 0), (sx - 5, HEIGHT), 1)

    y     = 15
    title = ft.render("SOUND RADAR — AI", True, GREEN)
    surface.blit(title, (sx + 90 - title.get_width() // 2, y))
    y += 20

    sub = ftiny.render("CNN Model Active", True, GREEN_DIM)
    surface.blit(sub, (sx + 90 - sub.get_width() // 2, y))
    y += 25

    col    = GREEN if connected else ORANGE
    status = "LIVE" if connected else "WAITING"
    s      = ft.render(status, True, col)
    surface.blit(s, (sx + 90 - s.get_width() // 2, y))
    y += 30

    pygame.draw.line(surface, GREEN_DIM, (sx, y), (sx + 175, y), 1)
    y += 10
    surface.blit(ft.render("LEGEND", True, DIM), (sx, y))
    y += 18
    for color, label in [
        (RED,    "Red    Footstep"),
        (YELLOW, "Yellow Gunshot"),
    ]:
        surface.blit(ftiny.render(label, True, color), (sx, y))
        y += 16

    y += 5
    pygame.draw.line(surface, GREEN_DIM, (sx, y), (sx + 175, y), 1)
    y += 10

    surface.blit(ft.render("MODEL OUTPUT", True, DIM), (sx, y))
    y += 18
    for i, cls in enumerate(CLASSES):
        prob  = last_probs[i]
        color = RED if cls == "footstep" else YELLOW if cls == "gunshot" else DIM
        bar_w = int(150 * prob)
        pygame.draw.rect(surface, (30, 30, 30), (sx, y, 150, 10))
        pygame.draw.rect(surface, color, (sx, y, bar_w, 10))
        label = ftiny.render(f"{cls} {prob:.2f}", True, WHITE)
        surface.blit(label, (sx, y + 11))
        y += 24

    y += 5
    pygame.draw.line(surface, GREEN_DIM, (sx, y), (sx + 175, y), 1)
    y += 10
    surface.blit(ft.render("SIGNAL LOG", True, DIM), (sx, y))
    y += 18
    for entry in signal_log:
        parts = entry.split("  ")
        color = RED if "FOOTSTEP" in entry else YELLOW
        surface.blit(ftiny.render(parts[0], True, DIM), (sx, y))
        y += 12
        surface.blit(ftiny.render("  ".join(parts[1:]), True, color), (sx, y))
        y += 14


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Valorant Sound Radar — AI")
    clock  = pygame.time.Clock()

    fs    = pygame.font.SysFont("monospace", 14, bold=True)
    ft    = pygame.font.SysFont("monospace", 11, bold=True)
    ftiny = pygame.font.SysFont("monospace", 10)

    recv_thread = threading.Thread(target=udp_receiver, daemon=True)
    recv_thread.start()

    global sweep_angle, pulses, last_inference_time

    print("Valorant Sound Radar (AI) started.")
    print("Press Q to quit\n")

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                running = False

        now = time.time()
        if now - last_inference_time >= INFERENCE_INTERVAL:
            last_inference_time = now
            new_pulses = run_inference()
            if new_pulses:
                pulses.extend(new_pulses)

        sweep_angle = (sweep_angle + 1.5) % 360
        pulses      = [p for p in pulses if p.update()]

        draw_radar_bg(screen)
        draw_sweep(screen, sweep_angle)

        ps = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        for p in pulses:
            p.draw(ps)
        screen.blit(ps, (0, 0))

        draw_labels(screen, fs, ftiny)
        connected = (time.time() - last_audio_time) < 2.0
        draw_sidebar(screen, ft, ftiny, connected)

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()


if __name__ == "__main__":
    main()