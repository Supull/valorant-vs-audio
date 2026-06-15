"""
VALORANT SOUND RADAR — Phase 2 Data Recorder v2
================================================
Records labeled audio clips for AI training.
Fixed: robust ring buffer, no zero clips, real-time energy shown.

Controls:
  3 = footstep heard
  4 = gunshot heard
  5 = silence (nothing happening)
  Q = quit

Run: python3 recorder.py
"""

import socket
import numpy as np
import soundfile as sf
import os
import time
import threading
import sys
import tty
import termios
import select

# ─── CONFIG ───────────────────────────────────────────────
PORT          = 5005
SAMPLE_RATE   = 44100
CHANNELS      = 2
CLIP_DURATION = 0.5
BUFFER_SAMPLES = int(SAMPLE_RATE * CLIP_DURATION)
# ──────────────────────────────────────────────────────────

CLASSES = ["footstep", "gunshot", "silence"]
for c in CLASSES:
    os.makedirs(f"data/{c}", exist_ok=True)

# ─── RING BUFFER ────────────────────────────────────────────
RING_SIZE   = SAMPLE_RATE * 3   # 3 seconds of history
ring_buffer = np.zeros((RING_SIZE, CHANNELS), dtype=np.float32)
write_pos   = 0
total_written = 0
buffer_lock = threading.Lock()
running     = True
packets_received = 0
# ──────────────────────────────────────────────────────────


def get_existing_count(label):
    folder = f"data/{label}"
    files = [f for f in os.listdir(folder) if f.endswith(".wav")]
    return len(files)

counts = {c: get_existing_count(c) for c in CLASSES}


def udp_receiver():
    global write_pos, total_written, packets_received

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", PORT))
    sock.settimeout(1.0)

    while running:
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
                packets_received += 1

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


def save_clip(label):
    clip = get_last_clip()
    if clip is None:
        return None, None, "not enough audio buffered yet"

    energy = float(np.abs(clip).mean())

    counts[label] += 1
    idx      = counts[label]
    filename = f"data/{label}/{label}_{idx:04d}.wav"
    sf.write(filename, clip, SAMPLE_RATE)
    return filename, idx, energy


def setup_terminal():
    fd  = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    tty.setraw(fd)
    return fd, old


def restore_terminal(fd, old):
    termios.tcsetattr(fd, termios.TCSADRAIN, old)


def key_available():
    return select.select([sys.stdin], [], [], 0.05)[0] != []


def print_status():
    parts = [f"{c}: {counts[c]}" for c in CLASSES]
    print(f"\r  Clips — {' | '.join(parts)}  |  packets: {packets_received}    ",
          end="", flush=True)


def main():
    global running

    print("=" * 55)
    print("  VALORANT SOUND RADAR — Phase 2 Data Recorder v2")
    print("=" * 55)
    print()
    print("Controls:")
    print("  3 = footstep    4 = gunshot    5 = silence")
    print("  Q = quit")
    print()
    print("Press the key the MOMENT you hear the sound.")
    print()

    recv_thread = threading.Thread(target=udp_receiver, daemon=True)
    recv_thread.start()

    print("Waiting for audio stream from Windows laptop...")
    while total_written < BUFFER_SAMPLES and running:
        time.sleep(0.1)
    print("Buffer ready — start playing Valorant and press keys!\n")
    print_status()

    KEY_MAP = {'3': 'footstep', '4': 'gunshot', '5': 'silence'}

    fd, old = setup_terminal()
    try:
        while running:
            if key_available():
                key = sys.stdin.read(1).lower()

                if key == 'q':
                    running = False
                    break

                if key in KEY_MAP:
                    label = KEY_MAP[key]
                    filename, idx, info = save_clip(label)
                    if filename:
                        print(f"\n  Saved {filename}  (energy={info:.5f})")
                    else:
                        print(f"\n  Skipped — {info}")
                    print_status()
    finally:
        restore_terminal(fd, old)

    print(f"\n\nRecording session complete!")
    print(f"Clips saved:")
    total = 0
    for c in CLASSES:
        print(f"  {c:10} {counts[c]:4d} clips  ->  data/{c}/")
        total += counts[c]
    print(f"  {'TOTAL':10} {total:4d} clips")
    print()
    print("Run diagnose.py to verify clips have real audio")
    print("Then run train.py to train the model")


if __name__ == "__main__":
    main()