"""
FREQUENCY FINDER — Run this on Mac
===================================
Prints the dominant frequency of incoming audio in real time.
Walk around in Valorant and note what frequency appears.
That's your footstep frequency.

Run: python3 freq_finder.py
"""

import socket
import numpy as np
import time

from config import PORT, SAMPLE_RATE, CHANNELS

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("0.0.0.0", PORT))
sock.settimeout(1.0)

print("=" * 50)
print("  FREQUENCY FINDER")
print("=" * 50)
print("Listening for audio from Windows laptop...")
print("Walk around in Valorant and watch the numbers")
print("Press Ctrl+C to stop\n")
print(f"{'Time':10} {'Top Freq (Hz)':>15} {'Energy':>10} {'Sound Type':>15}")
print("-" * 55)

last_print = 0

while True:
    try:
        data, _ = sock.recvfrom(65507)

        audio = np.frombuffer(data, dtype=np.float32)
        if len(audio) < CHANNELS:
            continue

        audio = audio[:len(audio) - len(audio) % CHANNELS].reshape(-1, CHANNELS)
        mono  = (audio[:, 0] + audio[:, 1]) / 2.0

        # Total energy — skip silence
        energy = float(np.mean(np.abs(mono)))
        if energy < 0.001:
            continue

        # FFT
        window   = np.hanning(len(mono))
        fft_vals = np.fft.rfft(mono * window)
        fft_mag  = np.abs(fft_vals)
        freqs    = np.fft.rfftfreq(len(mono), 1.0 / SAMPLE_RATE)

        # Find top 3 dominant frequencies
        top_indices = np.argsort(fft_mag)[-3:][::-1]
        top_freqs   = freqs[top_indices]
        top_mags    = fft_mag[top_indices]

        dominant_freq = top_freqs[0]

        # Rough label
        if dominant_freq < 60:
            label = "Sub Bass"
        elif dominant_freq < 300:
            label = ">>> FOOTSTEP? <<<"
        elif dominant_freq < 2000:
            label = "Mid (gunshot?)"
        elif dominant_freq < 8000:
            label = "High (ability?)"
        else:
            label = "Very High"

        # Print every 0.1 seconds max
        now = time.time()
        if now - last_print > 0.1:
            last_print = now
            t = time.strftime('%H:%M:%S')
            print(f"{t:10} {dominant_freq:>12.1f}Hz {energy:>10.5f}  {label}")
            # Also print 2nd and 3rd freq if significant
            for i in range(1, 3):
                if top_mags[i] > top_mags[0] * 0.5:
                    print(f"{'':10} {top_freqs[i]:>12.1f}Hz (secondary)")

    except socket.timeout:
        continue
    except KeyboardInterrupt:
        print("\nDone.")
        break