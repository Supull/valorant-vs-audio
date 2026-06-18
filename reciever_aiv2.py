"""
VALORANT SOUND RADAR — Mac Receiver v4.3 (Minimal Orbital HUD)
================================================================
Uses a trained CNN (modelv2.pth) to classify audio in real time.

Clean, minimal HUD: a center crosshair dot surrounded by a faint
orbit ring. When a sound is detected, a curved arc illuminates on
that ring in the direction of the source — flaring outward with
layered glow trails before narrowing and fading. No sidebar, no
clutter — the arc animation itself conveys direction and intensity.

Requirements:
  pip3 install numpy pygame torch torchaudio soundfile

Run: python3 receiver_ai_v2.py
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

MODEL_PATH    = "modelv2.pth"

INFERENCE_INTERVAL   = 0.1   # run model 10x per second
CONFIDENCE_THRESHOLD = 0.6
# ──────────────────────────────────────────────────────────

# ─── DISPLAY ───────────────────────────────────────────────
WIDTH        = 450
HEIGHT       = 450
CENTER       = (WIDTH // 2, HEIGHT // 2)
RADAR_RADIUS = 270           # kept for compatibility with detect_direction scaling
FPS          = 60

# Pulse ring specific settings — reactive HUD style (no travel motion)
PULSE_DURATION    = 0.6      # seconds each pulse stays visible
CROSSHAIR_SIZE    = 14
# ──────────────────────────────────────────────────────────

# ─── COLORS ────────────────────────────────────────────────
DARK_GREEN  = (0,   18,  5)
GREEN       = (0,   255, 70)
GREEN_DIM   = (0,   55,  18)
GRID_COLOR  = (0,   38,  12)
RED         = (255, 60,  60)
YELLOW      = (255, 220, 50)
BLUE        = (50,  150, 255)
CYAN        = (80,  230, 230)
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
last_audio_time      = 0
signal_log           = deque(maxlen=8)
last_inference_time  = 0
last_probs           = np.ones(len(CLASSES)) / len(CLASSES)
last_prediction      = "silence"
# ──────────────────────────────────────────────────────────


class PulseArc:
    """
    Dynamic radial HUD pulse — a curved arc segment that lights up
    on a fixed orbit ring around the crosshair, at the angle the
    sound came from. The arc expands in angular width on spawn,
    glows brightly, then contracts and fades while leaving a
    trailing afterglow — like a comet sweep along the ring rather
    than a static blob.
    """
    ORBIT_RADIUS = 150   # fixed distance of the orbit ring from center

    def __init__(self, angle, color, label, energy, intensity):
        self.angle     = angle                          # center angle of the arc
        self.color     = color
        self.label     = label
        self.energy    = energy
        self.intensity = max(0.15, min(1.0, intensity))

        self.birth      = time.time()
        self.duration   = PULSE_DURATION
        self.max_span   = 26 + 34 * self.intensity       # max angular width in degrees
        self.max_thick  = 4 + 7 * self.intensity          # arc stroke thickness
        self.alpha      = 255

    def update(self):
        age = time.time() - self.birth
        t   = age / self.duration
        if t >= 1.0:
            return False
        # Fade starts after the initial flare (first 35%)
        if t > 0.35:
            fade_t     = (t - 0.35) / 0.65
            self.alpha = max(0, int(255 * (1 - fade_t) ** 1.3))
        else:
            self.alpha = 255
        return True

    def _to_pygame_rad(self, deg):
        # our convention: 0=front(up), 90=right, clockwise
        # pygame arc: 0=right(east), increases counter-clockwise
        return math.radians(90 - deg)

    def draw(self, surface):
        if self.alpha <= 0:
            return

        age = time.time() - self.birth
        t   = age / self.duration

        # Angular span: flares wide quickly then narrows back down
        if t < 0.35:
            grow_t = t / 0.35
            span   = self.max_span * (1 - (1 - grow_t) ** 3)   # fast ease-out grow
        else:
            shrink_t = (t - 0.35) / 0.65
            span     = self.max_span * (1 - shrink_t * 0.4)     # gently narrows

        r, g, b = self.color

        size   = int((self.ORBIT_RADIUS + 40) * 2)
        s      = pygame.Surface((size, size), pygame.SRCALPHA)
        local_center = (size // 2, size // 2)
        rect = pygame.Rect(local_center[0] - self.ORBIT_RADIUS,
                            local_center[1] - self.ORBIT_RADIUS,
                            self.ORBIT_RADIUS * 2, self.ORBIT_RADIUS * 2)

        start_deg = self.angle - span / 2
        end_deg   = self.angle + span / 2

        # Outer soft glow trail (wider arc, thicker, dim)
        try:
            pygame.draw.arc(s, (r, g, b, max(0, self.alpha // 4)), rect,
                            self._to_pygame_rad(end_deg + 6),
                            self._to_pygame_rad(start_deg - 6),
                            int(self.max_thick * 2.4))
        except Exception:
            pass

        # Mid glow layer
        try:
            pygame.draw.arc(s, (r, g, b, max(0, self.alpha // 2)), rect,
                            self._to_pygame_rad(end_deg + 2),
                            self._to_pygame_rad(start_deg - 2),
                            int(self.max_thick * 1.4))
        except Exception:
            pass

        # Bright core arc
        try:
            pygame.draw.arc(s, (r, g, b, self.alpha), rect,
                            self._to_pygame_rad(end_deg),
                            self._to_pygame_rad(start_deg),
                            max(1, int(self.max_thick)))
        except Exception:
            pass

        # Hot white highlight at the exact center angle (peak intensity point)
        rad = math.radians(self.angle)
        hx  = local_center[0] + self.ORBIT_RADIUS * math.sin(rad)
        hy  = local_center[1] - self.ORBIT_RADIUS * math.cos(rad)
        glow_r = 3 + 4 * self.intensity
        pygame.draw.circle(s, (255, 255, 255, self.alpha // 2),
                           (int(hx), int(hy)), int(glow_r))

        surface.blit(s, (CENTER[0] - size // 2, CENTER[1] - size // 2))


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

    # Suppress detections directly on the front/back vertical axis —
    # these are almost always your own footsteps/movement, not enemies
    CENTER_DEAD_ZONE = 12  # degrees on either side of 0 and 180
    dist_from_front = min(angle, 360 - angle)
    dist_from_back   = abs(angle - 180)
    if dist_from_front < CENTER_DEAD_ZONE or dist_from_back < CENTER_DEAD_ZONE:
        return []

    now = time.strftime('%H:%M:%S')

    if pred_class == "footstep":
        intensity = min(1.0, energy * 80)
        dir_str = "L" if 180 < angle <= 360 else "R" if 0 < angle <= 180 else "C"
        log = f"{now}  FOOTSTEP  conf={confidence:.2f}  {dir_str}  {angle:.0f}deg"
        signal_log.appendleft(log)
        print(f"  >>> FOOTSTEP  conf={confidence:.2f}  energy={energy:.4f}  dir={dir_str}  angle={angle:.0f}deg")
        return [PulseArc(angle, RED, "footstep", energy, intensity)]

    elif pred_class in ("gunshot", "spectre"):
        intensity = min(1.0, energy * 15)
        label = "GUNSHOT" if pred_class == "gunshot" else "SPECTRE"
        log = f"{now}  {label:8} conf={confidence:.2f}"
        signal_log.appendleft(log)
        print(f"  >>> {label}   conf={confidence:.2f}  energy={energy:.4f}")
        return [PulseArc(angle, YELLOW, pred_class, energy, intensity)]

    elif pred_class == "jump":
        intensity = min(1.0, energy * 100)
        dir_str = "L" if 180 < angle <= 360 else "R" if 0 < angle <= 180 else "C"
        log = f"{now}  JUMP      conf={confidence:.2f}  {dir_str}  {angle:.0f}deg"
        signal_log.appendleft(log)
        print(f"  >>> JUMP      conf={confidence:.2f}  energy={energy:.4f}  dir={dir_str}  angle={angle:.0f}deg")
        return [PulseArc(angle, CYAN, "jump", energy, intensity)]

    return []


def draw_background(surface):
    """Clean dark HUD background — minimal, just the orbit track"""
    surface.fill(DARK_GREEN)
    # Single subtle orbit ring — where all arcs appear
    pygame.draw.circle(surface, GREEN_DIM, CENTER, PulseArc.ORBIT_RADIUS, 1)


def draw_crosshair(surface):
    """Center dot representing the player, with small crosshair ticks"""
    cx, cy = CENTER
    s = CROSSHAIR_SIZE

    # Outer glow
    glow = pygame.Surface((s * 6, s * 6), pygame.SRCALPHA)
    pygame.draw.circle(glow, (*GREEN, 40), (s * 3, s * 3), s * 2)
    surface.blit(glow, (cx - s * 3, cy - s * 3))

    # Crosshair ticks
    gap = 6
    pygame.draw.line(surface, GREEN, (cx - s, cy), (cx - gap, cy), 2)
    pygame.draw.line(surface, GREEN, (cx + gap, cy), (cx + s, cy), 2)
    pygame.draw.line(surface, GREEN, (cx, cy - s), (cx, cy - gap), 2)
    pygame.draw.line(surface, GREEN, (cx, cy + gap), (cx, cy + s), 2)

    # Center dot
    pygame.draw.circle(surface, WHITE, (cx, cy), 4)
    pygame.draw.circle(surface, GREEN, (cx, cy), 7, 1)


def draw_status(surface, ftiny, connected):
    """Minimal connection indicator — top left corner only"""
    col    = GREEN if connected else ORANGE
    status = "LIVE" if connected else "WAITING"
    dot_r  = 4
    pygame.draw.circle(surface, col, (16, 16), dot_r)
    label = ftiny.render(status, True, col)
    surface.blit(label, (26, 10))


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Sound HUD")
    clock  = pygame.time.Clock()

    ftiny = pygame.font.SysFont("monospace", 10)

    recv_thread = threading.Thread(target=udp_receiver, daemon=True)
    recv_thread.start()

    global pulses, last_inference_time

    print("Valorant Sound Radar — Orbital Pulse HUD (AI) started.")
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

        pulses = [p for p in pulses if p.update()]

        draw_background(screen)

        ps = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        for p in pulses:
            p.draw(ps)
        screen.blit(ps, (0, 0))

        draw_crosshair(screen)
        connected = (time.time() - last_audio_time) < 2.0
        draw_status(screen, ftiny, connected)

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()


if __name__ == "__main__":
    main()