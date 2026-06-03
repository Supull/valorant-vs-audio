"""
FREQUENCY FINDER — Run this on secondary laptop
================================================
Prints dominant frequencies of incoming audio in real time.
Walk around in Valorant and note what frequencies appear.
Use those values to tune FOOTSTEP_LOW/HIGH in receiver_mac.py

Setup:
  1. pip3 install numpy
  2. python3 freq_finder.py
"""

import socket
import numpy as np
import time

# ─── CONFIG ───────────────────────────────────────────────
PORT        = 5005
SAMPLE_RATE = 44100
CHANNELS    = 2
# ──────────────────────────────────────────────────────────

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("0.0.0.0", PORT))
sock.settimeout(1.0)

print("=" * 55)
print("  FREQUENCY FINDER")
print("=" * 55)
print("Listening for audio from Windows laptop...")
print("Walk around in Valorant and watch the numbers")
print("Press Ctrl+C to stop\n")
print(f"{'Time':10} {'Top Freq':>14} {'Energy':>10} {'Label':>18}")
print("-" * 55)

last_print = 0

while True:
    try:
        data, _ = sock.recvfrom(65507)

        audio = np.frombuffer(data, dtype=np.float32)
        if len(audio) < CHANNELS:
            continue

        audio  = audio[:len(audio) - len(audio) % CHANNELS].reshape(-1, CHANNELS)
        mono   = (audio[:, 0] + audio[:, 1]) / 2.0
        energy = float(np.mean(np.abs(mono)))

        if energy < 0.001:
            continue

        window   = np.hanning(len(mono))
        fft_vals = np.fft.rfft(mono * window)
        fft_mag  = np.abs(fft_vals)
        freqs    = np.fft.rfftfreq(len(mono), 1.0 / SAMPLE_RATE)

        top_indices = np.argsort(fft_mag)[-3:][::-1]
        top_freqs   = freqs[top_indices]
        top_mags    = fft_mag[top_indices]
        dominant    = top_freqs[0]

        if dominant < 60:
            label = "Sub Bass"
        elif dominant < 300:
            label = ">>> FOOTSTEP? <<<"
        elif dominant < 2000:
            label = "Mid (gunshot?)"
        elif dominant < 8000:
            label = "High (ability?)"
        else:
            label = "Very High"

        now = time.time()
        if now - last_print > 0.1:
            last_print = now
            t = time.strftime('%H:%M:%S')
            print(f"{t:10} {dominant:>11.1f}Hz {energy:>10.5f}  {label}")
            for i in range(1, 3):
                if top_mags[i] > top_mags[0] * 0.5:
                    print(f"{'':10} {top_freqs[i]:>11.1f}Hz (secondary)")

    except socket.timeout:
        continue
    except KeyboardInterrupt:
        print("\nDone.")
        break