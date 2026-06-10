"""
VALORANT SOUND RADAR — Phase 2 Data Recorder
=============================================
Records labeled audio clips for AI training.

Controls:
  F = footstep heard
  G = gunshot heard
  S = silence (nothing happening)
  Q = quit

Setup:
  1. Keep sender_windows.py running on gaming laptop
  2. Run this script on Mac instead of receiver_mac.py
  3. Play Valorant and press keys when you hear sounds
  4. Clips saved to data/footstep/, data/gunshot/, data/silence/

Run: python3 recorder.py
"""

import socket
import numpy as np
import soundfile as sf
import os
import time
import threading
from collections import deque
import sys
import tty
import termios

# ─── CONFIG ───────────────────────────────────────────────
PORT          = 5005
SAMPLE_RATE   = 44100
CHANNELS      = 2
CLIP_DURATION = 0.5              # seconds per saved clip
BUFFER_SIZE   = int(SAMPLE_RATE * CLIP_DURATION)  # samples in rolling buffer
# ──────────────────────────────────────────────────────────

# ─── DIRECTORIES ───────────────────────────────────────────
CLASSES = ["footstep", "gunshot", "silence"]
for c in CLASSES:
    os.makedirs(f"data/{c}", exist_ok=True)
# ──────────────────────────────────────────────────────────

# ─── STATE ─────────────────────────────────────────────────
# Rolling buffer — always holds last CLIP_DURATION seconds
audio_buffer  = deque(maxlen=50)   # raw UDP chunks
rolling_audio = np.zeros((BUFFER_SIZE, CHANNELS), dtype=np.float32)
lock          = threading.Lock()
counts        = {c: 0 for c in CLASSES}
running       = True
# ──────────────────────────────────────────────────────────


def udp_receiver():
    """Background thread — receives audio from Windows"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", PORT))
    sock.settimeout(1.0)
    print(f"Listening on port {PORT}...")
    while running:
        try:
            data, _ = sock.recvfrom(65507)
            with lock:
                audio_buffer.append(data)
        except socket.timeout:
            continue
        except Exception as e:
            print(f"UDP error: {e}")
            break


def update_rolling_buffer():
    """Continuously updates rolling audio buffer from UDP chunks"""
    global rolling_audio
    while running:
        chunks = []
        with lock:
            while audio_buffer:
                chunks.append(audio_buffer.popleft())

        for chunk in chunks:
            try:
                audio = np.frombuffer(chunk, dtype=np.float32)
                audio = audio[:len(audio) - len(audio) % CHANNELS]
                audio = audio.reshape(-1, CHANNELS)

                # Append to rolling buffer
                if len(audio) >= BUFFER_SIZE:
                    rolling_audio = audio[-BUFFER_SIZE:]
                else:
                    rolling_audio = np.roll(rolling_audio, -len(audio), axis=0)
                    rolling_audio[-len(audio):] = audio
            except Exception:
                continue

        time.sleep(0.005)


def save_clip(label):
    """Save current rolling buffer as a labeled wav file"""
    with lock:
        clip = rolling_audio.copy()

    counts[label] += 1
    idx      = counts[label]
    filename = f"data/{label}/{label}_{idx:04d}.wav"

    sf.write(filename, clip, SAMPLE_RATE)
    return filename, idx


def get_keypress():
    """Read single keypress without enter"""
    fd       = sys.stdin.fileno()
    old      = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    return ch.lower()


def print_status():
    """Print current clip counts"""
    parts = [f"{c}: {counts[c]}" for c in CLASSES]
    print(f"\r  Clips saved — {' | '.join(parts)}    ", end="", flush=True)


def main():
    global running

    print("=" * 55)
    print("  VALORANT SOUND RADAR — Phase 2 Data Recorder")
    print("=" * 55)
    print()
    print("Controls:")
    print("  F = footstep    G = gunshot    S = silence")
    print("  Q = quit")
    print()
    print("Tips:")
    print("  Press F the moment you HEAR a footstep")
    print("  Press G the moment you HEAR a gunshot")
    print("  Press S during any quiet moment")
    print("  Press repeatedly — each press = one clip saved")
    print()
    print("Target: 200+ clips per class for good accuracy")
    print()

    # Start background threads
    recv_thread   = threading.Thread(target=udp_receiver,       daemon=True)
    buffer_thread = threading.Thread(target=update_rolling_buffer, daemon=True)
    recv_thread.start()
    buffer_thread.start()

    # Wait for audio to start flowing
    print("Waiting for audio stream from Windows laptop...")
    time.sleep(1.0)
    print("Ready — start playing Valorant and press keys!\n")
    print_status()

    KEY_MAP = {
        '3': 'footstep',
        '4': 'gunshot',
        '5': 'silence',
    }

    while running:
        try:
            key = get_keypress()

            if key == 'q':
                print("\n\nStopping recorder...")
                running = False
                break

            if key in KEY_MAP:
                label         = KEY_MAP[key]
                filename, idx = save_clip(label)
                print(f"\n  Saved {filename}")
                print_status()

        except KeyboardInterrupt:
            running = False
            break

    print(f"\n\nRecording session complete!")
    print(f"Clips saved:")
    total = 0
    for c in CLASSES:
        print(f"  {c:10} {counts[c]:4d} clips  →  data/{c}/")
        total += counts[c]
    print(f"  {'TOTAL':10} {total:4d} clips")
    print()
    print("Next step: run train.py to train the AI model")


if __name__ == "__main__":
    main()