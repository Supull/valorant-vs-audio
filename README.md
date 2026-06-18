# Valorant Sound Radar

A real-time external sound visualizer for Valorant — because the game has no visual audio indicator.

<table>
  <tr>
    <td><img width="448" alt="Radar HUD view" src="https://github.com/user-attachments/assets/857ca94c-0a79-4cde-8a9c-955f21b460bc" /></td>
    <td><img width="441" alt="Orbital HUD view" src="https://github.com/user-attachments/assets/4a9c1cc0-db40-4d40-9acb-dc19ca2b1564" /></td>
  </tr>
</table>


Streams game audio from a Windows gaming laptop to a secondary device over WiFi, classifies sounds using a custom-trained CNN, and displays the result as a real-time directional HUD showing where sounds are coming from.

---

## What It Does

```
Gaming Laptop (Windows)          Secondary Laptop (Mac)
────────────────────────         ──────────────────────
Valorant plays audio      WiFi   Receives audio stream
       |                 ──────►        |
WASAPI captures it                 CNN classification
       |                          (footstep/gunshot/
Streams over UDP                   spectre/jump/silence)
                                        |
                                  Direction detection
                                  (stereo + HRTF analysis)
                                        |
                                  Real-time HUD display
```

Two interchangeable visualization modes are included:

- `receiver_mac.py` — top-down radar view with a rotating sweep line, signal log, and frequency-threshold detection (the original Phase 1 approach, kept for reference)
- `receiver_ai.py` — same radar view, but detection is powered by the trained CNN instead of frequency thresholds
- `receiver_ai_v2.py` — minimal orbital HUD: a center crosshair surrounded by a faint ring, with curved glowing arcs flaring up in the direction of detected sounds

---

## Why a CNN Instead of Pure Frequency Analysis

Early versions used FFT-based frequency thresholds to separate footsteps from gunshots. This worked until real gameplay data revealed that several sound types occupy nearly identical frequency ranges and energy levels, most notably footsteps versus the Spectre's suppressed gunshot, and jump landings versus regular footsteps. No amount of threshold tuning could reliably separate these, because the distinguishing feature is the *shape* of the sound over time, not a single frequency snapshot.

The project pivoted to training a small CNN on mel spectrograms of self-recorded gameplay audio, letting the model learn that temporal shape directly. This raised real-world reliability substantially, especially on the previously confused edge cases.

---

## Model Details

| | |
|---|---|
| Classes | footstep, gunshot, spectre, jump, silence |
| Training clips | 1,120 self-collected, labeled from live gameplay |
| Validation accuracy | 97.3% |
| Architecture | Small CNN over mel spectrograms (3 conv blocks + FC head) |
| Inference | ~44ms per clip on Apple Silicon (MPS) |

---

## Requirements

### Gaming Laptop (Windows)
- Python 3.x
- `pip install sounddevice numpy`
- Stereo Mix enabled in Windows Recording devices

### Secondary Laptop (Mac or any OS)
- Python 3.x
- `pip3 install numpy pygame torch torchaudio soundfile scipy`

---

## Setup

### 1. Enable Stereo Mix on Windows
```
Right click speaker icon -> Sounds -> Recording tab
Right click empty area -> Show Disabled Devices
Right click Stereo Mix -> Enable -> Set as Default Device
```

### 2. Valorant Audio Settings
```
Settings -> Audio -> HRTF -> ON
Windows -> Spatial Sound -> OFF
```
Turning off Spatial Sound prevents double-processing of the HRTF signal, which would otherwise corrupt direction detection.

### 3. Find Your Secondary Laptop's IP
On Mac:
```bash
ipconfig getifaddr en0
```
On Windows:
```cmd
ipconfig
```
Look for the IPv4 address under the WiFi adapter.

### 4. Set the IP in the sender script
Open `sender_windows.py` and update:
```python
MAC_IP = "your.secondary.laptop.ip"
```

---

## Running

### Step 1 — Start the receiver on the secondary laptop first
```bash
python3 receiver_ai_v2.py
```

### Step 2 — Start the sender on the gaming laptop
```cmd
python sender_windows.py
```
Expected output:
```
Streaming via: Stereo Mix
```

### Step 3 — Launch Valorant and play

---

## Collecting Training Data

`recorder.py` lets you build your own labeled dataset directly from gameplay:

```bash
python3 recorder.py
```

Controls:
```
3 = footstep      4 = gunshot       5 = silence
6 = jump          7 = spectre       Q = quit
```

Press the corresponding key the moment you hear each sound. Clips are saved as 0.5-second `.wav` files into `data/<class>/`, and numbering automatically continues across sessions without overwriting existing clips.

Run `diagnose.py` afterward to verify no clips were saved as silent/corrupted before training:
```bash
python3 diagnose.py
```

---

## Training

```bash
python3 train.py
```

Loads everything in `data/`, converts each clip to a normalized mel spectrogram, trains the CNN with class-weighted loss to handle imbalance, and saves the best-performing checkpoint to `modelv2.pth`.

---

## How Direction Detection Works

**Left / Right — accurate**
```
Measured from stereo channel energy difference.
Left channel louder  -> sound from the left
Right channel louder -> sound from the right
```

**Front / Back — estimated**
```
HRTF encodes direction via subtle frequency shaping.
Front sounds boost 4-8kHz slightly.
Back sounds cut 8-16kHz.
Requires Valorant HRTF ON to function correctly.
```

**Center dead-zone filtering**
```
Detections landing within ~12 degrees of the front/back axis
are suppressed, since these are almost always the player's own
footsteps rather than nearby enemies.
```

---

## Why Not an Overlay

Vanguard (Valorant's anti-cheat) flags transparent, always-on-top windows drawn directly over the game, since this is the same window signature used by ESP/aimbot overlays. This project instead runs on a separate physical device, which is completely outside Vanguard's reach, and is functionally equivalent to recording a speaker with a microphone. It never touches the Valorant process, reads game memory, or injects code.

---

## Files

| File | Device | Purpose |
|---|---|---|
| `sender_windows.py` | Gaming laptop (Windows) | Captures WASAPI/Stereo Mix audio, streams over UDP |
| `receiver_mac.py` | Secondary laptop | Radar view, frequency-threshold detection (Phase 1) |
| `receiver_ai.py` | Secondary laptop | Radar view, CNN-based detection (Phase 2) |
| `receiver_ai_v2.py` | Secondary laptop | Minimal orbital HUD, CNN-based detection (Phase 2) |
| `recorder.py` | Secondary laptop | Collects labeled training clips from live gameplay |
| `diagnose.py` | Secondary laptop | Verifies collected clips contain real audio |
| `train.py` | Secondary laptop | Trains the CNN and saves `modelv2.pth` |
| `freq_finder.py` | Secondary laptop | Calibration tool — prints dominant frequencies live |

---

## Roadmap

- Active learning loop — auto-flag low-confidence predictions during play for quick review and retraining
- More training data across different maps and agents for robustness
- Single-laptop borderless-window mode
- ESP32 physical HUD device port

---

## Contributing

This started as a personal project but contributions are welcome — 
whether that's more training data, support for other games, bug 
fixes, or new visualization styles. Open an issue or submit a pull 
request.

If you collect your own training data and retrain the model, 
consider sharing your `data/` clips or trained checkpoint so others 
can build on it.

---

## Disclaimer

This tool only reads audio that is already playing on the system. It does not interact with the Valorant game process, read game memory, or inject code of any kind. It is fully external and does not violate Valorant's Terms of Service.
