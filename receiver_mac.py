"""
VALORANT SOUND RADAR — Mac Receiver v3 (Tuned)
===============================================
Tuned to exact Valorant footstep frequencies:
  Core: 86.1, 129.2, 172.3, 215.3Hz
  Range: 60-220Hz

Run: python3 receiver_mac.py
"""

import socket
import numpy as np
import pygame
import threading
import math
import time
import sys
from collections import deque
from config import PORT, SAMPLE_RATE, CHANNELS

# ─── TUNED FOOTSTEP DETECTION ──────────────────────────────
FOOTSTEP_LOW          = 60      # Hz — captures 86Hz base
FOOTSTEP_HIGH         = 220     # Hz — cuts off above 215Hz harmonic
FOOTSTEP_ENERGY_MIN   = 0.001   # below this = silence, ignore
FOOTSTEP_ENERGY_MAX   = 0.014   # above this = likely gunshot bleed
GUNSHOT_FREQ_MIN      = 1500    # Hz — gunshot lives here
GUNSHOT_RATIO_THRESH  = 0.3     # high/low energy ratio — if exceeded = gunshot not footstep

# ─── DISPLAY ───────────────────────────────────────────────
WIDTH        = 700
HEIGHT       = 700
CENTER       = (WIDTH // 2, HEIGHT // 2)
RADAR_RADIUS = 270
FPS          = 60

# ─── COLORS ────────────────────────────────────────────────
DARK_GREEN  = (0,   18,  5)
GREEN       = (0,   255, 70)
GREEN_DIM   = (0,   55,  18)
GRID_COLOR  = (0,   38,  12)
RED         = (255, 60,  60)
RED_DARK    = (120, 20,  20)
YELLOW      = (255, 220, 50)
BLUE        = (50,  150, 255)
WHITE       = (255, 255, 255)
ORANGE      = (255, 140, 0)
DIM         = (80,  80,  80)

# ─── STATE ─────────────────────────────────────────────────
audio_buffer    = deque(maxlen=10)
pulses          = []
sweep_angle     = 0.0
lock            = threading.Lock()
last_audio_time = 0
signal_log      = deque(maxlen=8)   # recent detections for sidebar
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
        sz  = self.size
        s   = pygame.Surface((sz * 2 + 10, sz * 2 + 10), pygame.SRCALPHA)
        # Outer glow
        pygame.draw.circle(s, (r, g, b, self.alpha // 4),
                           (sz + 5, sz + 5), sz + 4)
        # Main dot
        pygame.draw.circle(s, (r, g, b, self.alpha),
                           (sz + 5, sz + 5), sz)
        # Bright center
        pygame.draw.circle(s, (255, 255, 255, self.alpha // 2),
                           (sz + 5, sz + 5), max(1, sz // 3))

        rad = math.radians(self.angle)
        x   = int(CENTER[0] + self.radius * math.sin(rad))
        y   = int(CENTER[1] - self.radius * math.cos(rad))
        surface.blit(s, (x - sz - 5, y - sz - 5))


def get_band_energy(fft_mag, freqs, low, high):
    mask = (freqs >= low) & (freqs <= high)
    return float(np.mean(fft_mag[mask])) if np.any(mask) else 0.0


def detect_direction(left_ch, right_ch, mono, freqs, fft_mag):
    """
    Returns (angle_degrees, confidence)
    0=front, 90=right, 180=back, 270=left
    """
    left_e  = float(np.mean(np.abs(left_ch)))
    right_e = float(np.mean(np.abs(right_ch)))
    total_e = left_e + right_e
    if total_e < 1e-6:
        return None, 0.0

    lr_ratio = (right_e - left_e) / total_e

    # HRTF front/back via frequency shape
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


def is_footstep(footstep_e, gunshot_e, total_e):
    """
    Returns True only if signal matches footstep profile
    not gunshot bleed
    """
    if footstep_e < FOOTSTEP_ENERGY_MIN:
        return False   # too quiet
    if footstep_e > FOOTSTEP_ENERGY_MAX:
        return False   # too loud = gunshot bleed
    if total_e > 1e-6 and (gunshot_e / total_e) > GUNSHOT_RATIO_THRESH:
        return False   # too much high freq = gunshot
    return True


def analyze_chunk(raw_bytes):
    global signal_log
    try:
        audio = np.frombuffer(raw_bytes, dtype=np.float32)
        if len(audio) < CHANNELS:
            return None

        audio    = audio[:len(audio) - len(audio) % CHANNELS].reshape(-1, CHANNELS)
        left_ch  = audio[:, 0]
        right_ch = audio[:, 1]
        mono     = (left_ch + right_ch) / 2.0

        window   = np.hanning(len(mono))
        fft_vals = np.fft.rfft(mono * window)
        fft_mag  = np.abs(fft_vals)
        freqs    = np.fft.rfftfreq(len(mono), 1.0 / SAMPLE_RATE)

        footstep_e = get_band_energy(fft_mag, freqs, FOOTSTEP_LOW, FOOTSTEP_HIGH)
        gunshot_e  = get_band_energy(fft_mag, freqs, GUNSHOT_FREQ_MIN, 20000)
        mid_e      = get_band_energy(fft_mag, freqs, 300, 1500)
        total_e    = footstep_e + gunshot_e + mid_e

        if total_e < FOOTSTEP_ENERGY_MIN:
            return None

        results = []
        now     = time.strftime('%H:%M:%S')

        # ── Footstep detection (tuned) ───────────────────
        if is_footstep(footstep_e, gunshot_e, total_e):
            angle, conf = detect_direction(left_ch, right_ch, mono, freqs, fft_mag)
            if angle is not None and conf > 0.05:
                # Radius = distance (louder = closer = smaller radius)
                radius  = max(50, min(RADAR_RADIUS - 30,
                              int(RADAR_RADIUS * (1 - min(1, footstep_e / FOOTSTEP_ENERGY_MAX)))))
                size    = max(5, min(16, int(footstep_e * 800)))
                dir_str = "L" if 180 < angle <= 360 else "R" if 0 < angle <= 180 else "C"
                results.append(Pulse(angle, radius, RED, size, "footstep", footstep_e))
                signal_log.appendleft(
                    f"{now}  FOOTSTEP  {footstep_e:.4f}  {dir_str}  {angle:.0f}°"
                )

        # ── Gunshot detection ────────────────────────────
        elif gunshot_e > 0.003 and gunshot_e > footstep_e * 2:
            angle, conf = detect_direction(left_ch, right_ch, mono, freqs, fft_mag)
            if angle is not None and conf > 0.1:
                radius = max(50, min(RADAR_RADIUS - 30,
                             int(RADAR_RADIUS * (1 - min(1, gunshot_e * 15)))))
                size   = max(6, min(18, int(gunshot_e * 200)))
                results.append(Pulse(angle, radius, YELLOW, size, "gunshot", gunshot_e))
                signal_log.appendleft(f"{now}  GUNSHOT   {gunshot_e:.4f}")

        # ── Ability / high freq ──────────────────────────
        elif mid_e > 0.02:
            angle, conf = detect_direction(left_ch, right_ch, mono, freqs, fft_mag)
            if angle is not None and conf > 0.1:
                results.append(Pulse(angle, RADAR_RADIUS - 50,
                                     BLUE, max(4, min(10, int(mid_e * 100))),
                                     "ability", mid_e))

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


def draw_radar_bg(surface):
    surface.fill(DARK_GREEN)
    # Concentric rings
    for r in range(RADAR_RADIUS // 4, RADAR_RADIUS + 1, RADAR_RADIUS // 4):
        pygame.draw.circle(surface, GRID_COLOR, CENTER, r, 1)
    # Cross
    pygame.draw.line(surface, GRID_COLOR,
                     (CENTER[0], CENTER[1] - RADAR_RADIUS),
                     (CENTER[0], CENTER[1] + RADAR_RADIUS), 1)
    pygame.draw.line(surface, GRID_COLOR,
                     (CENTER[0] - RADAR_RADIUS, CENTER[1]),
                     (CENTER[0] + RADAR_RADIUS, CENTER[1]), 1)
    # Diagonals
    off = int(RADAR_RADIUS * 0.707)
    pygame.draw.line(surface, GRID_COLOR,
                     (CENTER[0] - off, CENTER[1] - off),
                     (CENTER[0] + off, CENTER[1] + off), 1)
    pygame.draw.line(surface, GRID_COLOR,
                     (CENTER[0] + off, CENTER[1] - off),
                     (CENTER[0] - off, CENTER[1] + off), 1)
    # Border
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
    """Right side info panel"""
    sx = WIDTH - 180
    pygame.draw.line(surface, GREEN_DIM, (sx - 5, 0), (sx - 5, HEIGHT), 1)

    y = 15
    title = ft.render("SOUND RADAR", True, GREEN)
    surface.blit(title, (sx + 90 - title.get_width() // 2, y))
    y += 20

    sub = ftiny.render("Valorant Footstep Tuned", True, GREEN_DIM)
    surface.blit(sub, (sx + 90 - sub.get_width() // 2, y))
    y += 25

    # Status
    col    = GREEN if connected else ORANGE
    status = "● LIVE" if connected else "● WAITING"
    s      = ft.render(status, True, col)
    surface.blit(s, (sx + 90 - s.get_width() // 2, y))
    y += 30

    # Legend
    pygame.draw.line(surface, GREEN_DIM, (sx, y), (sx + 175, y), 1)
    y += 10
    surface.blit(ft.render("LEGEND", True, DIM), (sx, y))
    y += 18
    for color, label in [
        (RED,    "● Footstep (60-220Hz)"),
        (YELLOW, "● Gunshot (1500Hz+)"),
        (BLUE,   "● Ability / Mid"),
    ]:
        surface.blit(ftiny.render(label, True, color), (sx, y))
        y += 16

    y += 5
    pygame.draw.line(surface, GREEN_DIM, (sx, y), (sx + 175, y), 1)
    y += 10

    # Pulse size guide
    surface.blit(ft.render("DISTANCE", True, DIM), (sx, y))
    y += 18
    surface.blit(ftiny.render("Bigger = closer", True, DIM), (sx, y))
    y += 14
    surface.blit(ftiny.render("Center = very close", True, DIM), (sx, y))
    y += 14
    surface.blit(ftiny.render("Edge   = far away", True, DIM), (sx, y))
    y += 20

    pygame.draw.line(surface, GREEN_DIM, (sx, y), (sx + 175, y), 1)
    y += 10

    # Signal log
    surface.blit(ft.render("SIGNAL LOG", True, DIM), (sx, y))
    y += 18
    for entry in signal_log:
        parts = entry.split("  ")
        color = RED if "FOOTSTEP" in entry else YELLOW if "GUNSHOT" in entry else BLUE
        # Time
        surface.blit(ftiny.render(parts[0], True, DIM), (sx, y))
        y += 12
        # Type + detail
        detail = "  ".join(parts[1:])
        surface.blit(ftiny.render(detail, True, color), (sx, y))
        y += 14


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Valorant Sound Radar — Footstep Tuned")
    clock  = pygame.time.Clock()

    fs    = pygame.font.SysFont("monospace", 14, bold=True)
    ft    = pygame.font.SysFont("monospace", 11, bold=True)
    ftiny = pygame.font.SysFont("monospace", 10)

    recv_thread = threading.Thread(target=udp_receiver, daemon=True)
    recv_thread.start()

    global sweep_angle, pulses, last_audio_time

    print("🎮 Valorant Sound Radar — Footstep Tuned")
    print(f"   Footstep range: {FOOTSTEP_LOW}-{FOOTSTEP_HIGH}Hz")
    print("   Press Q to quit\n")

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                running = False

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
    print("Closed.")


if __name__ == "__main__":
    main()