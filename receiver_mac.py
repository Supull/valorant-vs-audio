"""
VALORANT SOUND RADAR — Mac Receiver + Radar Display v2
=======================================================
Now includes live frequency diagnostic panel
so you can identify exact footstep frequencies.

Run: python3 receiver_mac.py
     python3 receiver_mac.py --diag     (diagnostic mode only)
"""

import socket
import numpy as np
import pygame
import threading
import math
import time
import sys
from scipy import signal
from collections import deque

# ─── NETWORK CONFIG ────────────────────────────────────────
PORT        = 5005
SAMPLE_RATE = 44100
CHUNK_SIZE  = 1024
CHANNELS    = 2
# ──────────────────────────────────────────────────────────

# ─── DISPLAY CONFIG ────────────────────────────────────────
RADAR_W      = 600
DIAG_W       = 400
HEIGHT       = 600
RADAR_CENTER = (RADAR_W // 2, HEIGHT // 2)
RADAR_RADIUS = 240
FPS          = 60
DIAG_MODE    = "--diag" in sys.argv   # full screen diagnostic
WIDTH        = DIAG_W if DIAG_MODE else RADAR_W + DIAG_W
# ──────────────────────────────────────────────────────────

# ─── FREQUENCY BANDS TO MONITOR ────────────────────────────
# Each entry: (label, low_hz, high_hz, color)
FREQ_BANDS = [
    ("Sub Bass",   20,   60,   (100, 50,  200)),
    ("Footstep1",  60,   120,  (255, 50,  50)),    # low footstep thud
    ("Footstep2",  120,  200,  (255, 100, 50)),    # mid footstep
    ("Footstep3",  200,  300,  (255, 160, 50)),    # high footstep
    ("Low Mid",    300,  500,  (255, 220, 50)),
    ("Mid",        500,  1000, (180, 255, 50)),
    ("Upper Mid",  1000, 2000, (50,  255, 120)),
    ("Presence",   2000, 4000, (50,  200, 255)),
    ("Brilliance", 4000, 8000, (50,  100, 255)),
    ("Air",        8000, 16000,(100, 50,  255)),
]

# ─── FOOTSTEP DETECTION RANGE (tunable) ────────────────────
FOOTSTEP_LOW  = 60
FOOTSTEP_HIGH = 300

# ─── COLORS ────────────────────────────────────────────────
BLACK      = (0,   0,   0)
DARK_BG    = (8,   12,  20)
DARK_GREEN = (0,   20,  0)
GREEN      = (0,   255, 70)
GREEN_DIM  = (0,   60,  20)
RED        = (255, 50,  50)
YELLOW     = (255, 220, 50)
BLUE       = (50,  150, 255)
WHITE      = (255, 255, 255)
ORANGE     = (255, 140, 0)
GRID_COLOR = (0,   45,  15)
DIM_WHITE  = (140, 140, 140)
# ──────────────────────────────────────────────────────────

# ─── GLOBAL STATE ──────────────────────────────────────────
audio_buffer   = deque(maxlen=10)
pulses         = []
sweep_angle    = 0.0
lock           = threading.Lock()

# Diagnostic state — rolling history per band
HISTORY_LEN    = 120   # frames of history shown in sparkline
band_history   = {b[0]: deque([0.0] * HISTORY_LEN, maxlen=HISTORY_LEN) for b in FREQ_BANDS}
band_current   = {b[0]: 0.0 for b in FREQ_BANDS}
band_peak      = {b[0]: 0.0 for b in FREQ_BANDS}
band_peak_time = {b[0]: 0.0 for b in FREQ_BANDS}
PEAK_HOLD      = 2.0   # seconds to hold peak marker

# Footstep log
footstep_log   = deque(maxlen=6)
last_audio_time = 0
# ──────────────────────────────────────────────────────────


class Pulse:
    def __init__(self, angle, radius, color, size, label):
        self.angle    = angle
        self.radius   = radius
        self.color    = color
        self.size     = size
        self.label    = label
        self.alpha    = 255
        self.birth    = time.time()
        self.lifetime = 3.0

    def update(self):
        age        = time.time() - self.birth
        self.alpha = max(0, int(255 * (1 - age / self.lifetime)))
        return self.alpha > 0

    def draw(self, surface):
        if self.alpha <= 0:
            return
        r, g, b = self.color
        surf = pygame.Surface((self.size * 2 + 6, self.size * 2 + 6), pygame.SRCALPHA)
        pygame.draw.circle(surf, (r, g, b, self.alpha),
                           (self.size + 3, self.size + 3), self.size)
        pygame.draw.circle(surf, (r, g, b, self.alpha // 3),
                           (self.size + 3, self.size + 3), self.size + 3, 2)
        rad = math.radians(self.angle)
        x   = int(RADAR_CENTER[0] + self.radius * math.sin(rad))
        y   = int(RADAR_CENTER[1] - self.radius * math.cos(rad))
        surface.blit(surf, (x - self.size - 3, y - self.size - 3))


def get_band_energy(fft_mag, freqs, low, high):
    mask = (freqs >= low) & (freqs <= high)
    if not np.any(mask):
        return 0.0
    return float(np.mean(fft_mag[mask]))


def detect_direction(left_ch, right_ch):
    left_energy  = float(np.mean(np.abs(left_ch)))
    right_energy = float(np.mean(np.abs(right_ch)))
    total_energy = left_energy + right_energy
    if total_energy < 1e-6:
        return None, 0.0

    lr_ratio  = (right_energy - left_energy) / total_energy
    lr_angle  = lr_ratio * 90.0
    mono      = (left_ch + right_ch) / 2.0
    fft_vals  = np.fft.rfft(mono * np.hanning(len(mono)))
    fft_mag   = np.abs(fft_vals)
    freqs     = np.fft.rfftfreq(len(mono), 1.0 / SAMPLE_RATE)

    front_e   = get_band_energy(fft_mag, freqs, 4000, 8000)
    back_e    = get_band_energy(fft_mag, freqs, 8000, 16000)
    fb_total  = front_e + back_e
    fb_ratio  = (front_e - back_e) / fb_total if fb_total > 1e-8 else 0.0

    if abs(lr_ratio) > 0.1:
        base_angle = 90 if lr_ratio > 0 else 270
        angle      = base_angle - fb_ratio * 45
    else:
        angle      = 0 if fb_ratio >= 0 else 180

    confidence = min(1.0, total_energy * 20)
    return angle % 360, confidence


def analyze_chunk(raw_bytes):
    global band_current, band_peak, band_peak_time, footstep_log
    try:
        audio = np.frombuffer(raw_bytes, dtype=np.float32)
        if len(audio) < CHANNELS:
            return None

        audio     = audio[:len(audio) - len(audio) % CHANNELS].reshape(-1, CHANNELS)
        left_ch   = audio[:, 0]
        right_ch  = audio[:, 1]
        mono      = (left_ch + right_ch) / 2.0

        window    = np.hanning(len(mono))
        fft_vals  = np.fft.rfft(mono * window)
        fft_mag   = np.abs(fft_vals)
        freqs     = np.fft.rfftfreq(len(mono), 1.0 / SAMPLE_RATE)

        # ── Update diagnostic band levels ────────────────
        now = time.time()
        for name, low, high, color in FREQ_BANDS:
            energy = get_band_energy(fft_mag, freqs, low, high)
            band_current[name] = energy
            band_history[name].append(energy)
            if energy > band_peak[name]:
                band_peak[name]      = energy
                band_peak_time[name] = now
            elif now - band_peak_time[name] > PEAK_HOLD:
                band_peak[name] = max(band_peak[name] * 0.95, energy)

        # ── Footstep detection ───────────────────────────
        footstep_e = get_band_energy(fft_mag, freqs, FOOTSTEP_LOW, FOOTSTEP_HIGH)
        mid_e      = get_band_energy(fft_mag, freqs, 300, 4000)
        high_e     = get_band_energy(fft_mag, freqs, 4000, 20000)
        total_e    = footstep_e + mid_e + high_e

        if total_e < 0.001:
            return None

        results = []

        if footstep_e > 0.008 and footstep_e > mid_e * 0.5:
            angle, conf = detect_direction(left_ch, right_ch)
            if angle is not None and conf > 0.05:
                radius = max(40, min(RADAR_RADIUS - 20,
                             int(RADAR_RADIUS * (1 - min(1, footstep_e * 30)))))
                size   = max(4, min(14, int(footstep_e * 300)))
                results.append(Pulse(angle, radius, RED, size, "footstep"))
                footstep_log.appendleft(
                    f"{time.strftime('%H:%M:%S')}  {footstep_e:.4f}  "
                    f"{'L' if angle > 180 else 'R' if angle < 180 else 'C'}"
                )

        if mid_e > 0.02:
            angle, conf = detect_direction(left_ch, right_ch)
            if angle is not None and conf > 0.1:
                radius = max(40, min(RADAR_RADIUS - 20,
                             int(RADAR_RADIUS * (1 - min(1, mid_e * 10)))))
                size   = max(5, min(16, int(mid_e * 150)))
                results.append(Pulse(angle, radius, YELLOW, size, "gunshot"))

        if high_e > 0.015:
            angle, conf = detect_direction(left_ch, right_ch)
            if angle is not None and conf > 0.1:
                results.append(Pulse(angle, RADAR_RADIUS - 40, BLUE,
                                     max(3, min(8, int(high_e * 100))), "ability"))

        return results if results else None

    except Exception:
        return None


def udp_receiver():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", PORT))
    sock.settimeout(1.0)
    print(f"📡 Listening on port {PORT}...")
    while True:
        try:
            data, _ = sock.recvfrom(65507)
            with lock:
                audio_buffer.append(data)
        except socket.timeout:
            continue
        except Exception as e:
            print(f"UDP error: {e}")
            break


# ── DRAWING HELPERS ────────────────────────────────────────

def draw_radar_grid(surface):
    surface.fill(DARK_GREEN)
    for r in range(RADAR_RADIUS // 4, RADAR_RADIUS + 1, RADAR_RADIUS // 4):
        pygame.draw.circle(surface, GRID_COLOR, RADAR_CENTER, r, 1)
    pygame.draw.line(surface, GRID_COLOR,
                     (RADAR_CENTER[0], RADAR_CENTER[1] - RADAR_RADIUS),
                     (RADAR_CENTER[0], RADAR_CENTER[1] + RADAR_RADIUS), 1)
    pygame.draw.line(surface, GRID_COLOR,
                     (RADAR_CENTER[0] - RADAR_RADIUS, RADAR_CENTER[1]),
                     (RADAR_CENTER[0] + RADAR_RADIUS, RADAR_CENTER[1]), 1)
    pygame.draw.circle(surface, GREEN_DIM, RADAR_CENTER, RADAR_RADIUS, 2)


def draw_sweep(surface, angle_deg):
    surf  = pygame.Surface((RADAR_W, HEIGHT), pygame.SRCALPHA)
    steps = 30
    trail = 60
    for i in range(steps):
        a     = angle_deg - (trail * i / steps)
        alpha = int(80 * (1 - i / steps))
        rad   = math.radians(a)
        ex    = int(RADAR_CENTER[0] + RADAR_RADIUS * math.sin(rad))
        ey    = int(RADAR_CENTER[1] - RADAR_RADIUS * math.cos(rad))
        pygame.draw.line(surf, (0, 255, 70, alpha), RADAR_CENTER, (ex, ey), 2)
    rad = math.radians(angle_deg)
    ex  = int(RADAR_CENTER[0] + RADAR_RADIUS * math.sin(rad))
    ey  = int(RADAR_CENTER[1] - RADAR_RADIUS * math.cos(rad))
    pygame.draw.line(surf, (0, 255, 70, 200), RADAR_CENTER, (ex, ey), 2)
    surface.blit(surf, (0, 0))


def draw_radar_labels(surface, font_s, font_t):
    p = 18
    surface.blit(font_s.render("N", True, GREEN_DIM),
                 (RADAR_CENTER[0] - 6, RADAR_CENTER[1] - RADAR_RADIUS - p))
    surface.blit(font_s.render("S", True, GREEN_DIM),
                 (RADAR_CENTER[0] - 6, RADAR_CENTER[1] + RADAR_RADIUS + 4))
    surface.blit(font_s.render("E", True, GREEN_DIM),
                 (RADAR_CENTER[0] + RADAR_RADIUS + 6, RADAR_CENTER[1] - 8))
    surface.blit(font_s.render("W", True, GREEN_DIM),
                 (RADAR_CENTER[0] - RADAR_RADIUS - p, RADAR_CENTER[1] - 8))
    you = font_t.render("YOU", True, WHITE)
    surface.blit(you, (RADAR_CENTER[0] - you.get_width() // 2, RADAR_CENTER[1] + 8))
    pygame.draw.circle(surface, WHITE, RADAR_CENTER, 5)


def draw_diag_panel(surface, font_t, font_tiny, ox):
    """
    Draw the frequency diagnostic panel.
    ox = x offset (0 in diag-only mode, RADAR_W in split mode)
    """
    panel_rect = pygame.Rect(ox, 0, DIAG_W, HEIGHT)
    pygame.draw.rect(surface, DARK_BG, panel_rect)
    pygame.draw.line(surface, (30, 40, 60), (ox, 0), (ox, HEIGHT), 2)

    x0    = ox + 10
    BAR_W = DIAG_W - 20
    y     = 10

    # Title
    title = font_t.render("FREQUENCY ANALYSER", True, WHITE)
    surface.blit(title, (ox + DIAG_W // 2 - title.get_width() // 2, y))
    y += 22

    hint = font_tiny.render("Walk around — watch which bands spike", True, DIM_WHITE)
    surface.blit(hint, (ox + DIAG_W // 2 - hint.get_width() // 2, y))
    y += 18

    pygame.draw.line(surface, (30, 40, 60), (ox, y), (ox + DIAG_W, y), 1)
    y += 8

    # Band rows
    BAR_H     = 14
    SPARK_H   = 20
    ROW_GAP   = 8
    MAX_LEVEL = 0.08   # tune this to your audio levels

    for name, low, high, color in FREQ_BANDS:
        energy = band_current[name]
        peak   = band_peak[name]

        # Band name + freq range
        label     = font_tiny.render(f"{name}", True, color)
        hz_label  = font_tiny.render(f"{low}-{high}Hz", True, DIM_WHITE)
        surface.blit(label,    (x0, y))
        surface.blit(hz_label, (x0 + BAR_W - hz_label.get_width(), y))
        y += 14

        # Bar background
        pygame.draw.rect(surface, (20, 25, 35), (x0, y, BAR_W, BAR_H))

        # Bar fill
        fill = min(1.0, energy / MAX_LEVEL)
        if fill > 0:
            bar_color = color
            if fill > 0.8:
                bar_color = WHITE   # clip indicator
            pygame.draw.rect(surface, bar_color,
                             (x0, y, int(BAR_W * fill), BAR_H))

        # Peak marker
        peak_fill = min(1.0, peak / MAX_LEVEL)
        if peak_fill > 0:
            px = x0 + int(BAR_W * peak_fill)
            pygame.draw.line(surface, WHITE, (px, y), (px, y + BAR_H), 2)

        # Energy value text
        val = font_tiny.render(f"{energy:.5f}", True, DIM_WHITE)
        surface.blit(val, (x0 + BAR_W - val.get_width(), y + 1))

        y += BAR_H + 2

        # Sparkline (rolling history)
        hist   = list(band_history[name])
        spark_max = max(max(hist), 1e-6)
        pts    = []
        for i, v in enumerate(hist):
            sx = x0 + int(i * BAR_W / HISTORY_LEN)
            sy = y + SPARK_H - int((v / spark_max) * SPARK_H)
            pts.append((sx, sy))
        if len(pts) > 1:
            pygame.draw.lines(surface, (*color, 180), False, pts, 1)

        y += SPARK_H + ROW_GAP

    # Divider
    pygame.draw.line(surface, (30, 40, 60), (ox, y), (ox + DIAG_W, y), 1)
    y += 8

    # Footstep log
    log_title = font_t.render("FOOTSTEP LOG", True, RED)
    surface.blit(log_title, (x0, y))
    y += 18

    header = font_tiny.render("Time       Energy   Dir", True, DIM_WHITE)
    surface.blit(header, (x0, y))
    y += 14

    for entry in footstep_log:
        row = font_tiny.render(entry, True, RED)
        surface.blit(row, (x0, y))
        y += 14

    # Tuning hint at bottom
    hint2 = font_tiny.render(f"Footstep range: {FOOTSTEP_LOW}-{FOOTSTEP_HIGH}Hz", True, ORANGE)
    surface.blit(hint2, (x0, HEIGHT - 30))
    hint3 = font_tiny.render("Edit FOOTSTEP_LOW/HIGH in script to tune", True, DIM_WHITE)
    surface.blit(hint3, (x0, HEIGHT - 16))


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Valorant Sound Radar + Frequency Analyser")
    clock  = pygame.time.Clock()

    font_s    = pygame.font.SysFont("monospace", 14, bold=True)
    font_t    = pygame.font.SysFont("monospace", 12, bold=True)
    font_tiny = pygame.font.SysFont("monospace", 10)

    recv_thread = threading.Thread(target=udp_receiver, daemon=True)
    recv_thread.start()

    global sweep_angle, pulses, last_audio_time

    print("🎮 Valorant Sound Radar + Frequency Analyser")
    print("   Walk around in Valorant and watch which bands spike")
    print("   Red bands (60-300Hz) = footstep range")
    print("   Press Q to quit\n")

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                running = False

        # Process audio
        chunks = []
        with lock:
            while audio_buffer:
                chunks.append(audio_buffer.popleft())

        if chunks:
            last_audio_time = time.time()

        for chunk in chunks:
            new_pulses = analyze_chunk(chunk)
            if new_pulses:
                pulses.extend(new_pulses)

        sweep_angle = (sweep_angle + 1.5) % 360
        pulses      = [p for p in pulses if p.update()]

        # Draw radar (left panel)
        if not DIAG_MODE:
            radar_surf = pygame.Surface((RADAR_W, HEIGHT))
            draw_radar_grid(radar_surf)
            draw_sweep(radar_surf, sweep_angle)

            pulse_surf = pygame.Surface((RADAR_W, HEIGHT), pygame.SRCALPHA)
            for p in pulses:
                p.draw(pulse_surf)
            radar_surf.blit(pulse_surf, (0, 0))
            draw_radar_labels(radar_surf, font_s, font_tiny)

            # Status
            connected = (time.time() - last_audio_time) < 2.0
            status    = "● CONNECTED" if connected else "● WAITING..."
            col       = GREEN if connected else ORANGE
            radar_surf.blit(font_tiny.render(status, True, col), (10, 10))

            screen.blit(radar_surf, (0, 0))
            diag_ox = RADAR_W
        else:
            screen.fill(DARK_BG)
            diag_ox = 0

        # Draw diagnostic panel (right panel or full screen)
        draw_diag_panel(screen, font_t, font_tiny, diag_ox)

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    print("Closed.")


if __name__ == "__main__":
    main()