"""
DIAGNOSTIC v2 — Check collected audio clips in detail
======================================================
Run: python3 diagnose.py
"""

import soundfile as sf
import numpy as np
import os

CLASSES = ["footstep", "gunshot", "silence"]

print("=" * 60)
print("  DIAGNOSTIC v2 — Audio Clip Inspection")
print("=" * 60)

for c in CLASSES:
    folder = f"data/{c}"
    if not os.path.isdir(folder):
        print(f"\n--- {c} --- (folder not found)")
        continue

    files = sorted(os.listdir(folder))
    wav_files = [f for f in files if f.endswith(".wav")]
    print(f"\n--- {c} ({len(wav_files)} files) ---")

    zero_count = 0
    nonzero_energies = []

    for f in wav_files:
        audio, sr = sf.read(os.path.join(folder, f))
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        energy = float(np.abs(audio).mean())
        if energy < 1e-9:
            zero_count += 1
        else:
            nonzero_energies.append(energy)

    print(f"  Total files:     {len(wav_files)}")
    print(f"  All-zero clips:  {zero_count}  ({100*zero_count/len(wav_files):.1f}%)")
    print(f"  Non-zero clips:  {len(nonzero_energies)}")
    if nonzero_energies:
        arr = np.array(nonzero_energies)
        print(f"  Non-zero energy — min={arr.min():.5f}  max={arr.max():.5f}  mean={arr.mean():.5f}")

print("\n" + "=" * 60)