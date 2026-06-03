# Valorant Sound Radar

A real-time external sound visualizer for Valorant — because the game has no visual audio indicator.

Streams game audio from your Windows gaming laptop to a secondary device over WiFi, runs FFT analysis to isolate footstep frequencies, and displays a live directional radar showing where sounds are coming from.

---

## What It Does

```
Gaming Laptop (Windows)          Secondary Laptop (Mac)
────────────────────────         ──────────────────────
Valorant plays audio      WiFi   Receives audio stream
       |                 ──────►        |
WASAPI captures it                 FFT analysis
       |                         splits into bands
Streams over UDP                       |
                                  Radar display
                                 shows direction
                                  of footsteps
```

- Red pulses    — Footsteps (60-220Hz, tuned to Valorant's exact footstep harmonics)
- Yellow pulses — Gunshots / abilities (1500Hz+)
- Blue pulses   — Mid frequency sounds
- Pulse size    — louder = bigger = closer
- Pulse position — left/right direction via stereo channel analysis
- Front/back    — inferred from HRTF frequency shaping (requires HRTF ON in Valorant)

---

## Requirements

### Gaming Laptop (Windows)
- Python 3.x
- `pip install sounddevice numpy`
- Stereo Mix enabled in Windows Recording devices

### Secondary Laptop (Mac or any OS)
- Python 3.x
- `pip3 install numpy pygame scipy`

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

Turning off Spatial Sound is important — it prevents double processing of the HRTF signal which would corrupt direction detection.

### 3. Find Your Secondary Laptop IP
On Mac:
```bash
ipconfig getifaddr en0
```
On Windows:
```cmd
ipconfig
```
Look for IPv4 Address under the WiFi adapter.

### 4. Set the IP in the sender script
Open `sender_windows.py` and update:
```python
MAC_IP = "your.secondary.laptop.ip"
```

---

## Running

### Step 1 — Start receiver on secondary laptop first
```bash
python3 receiver_mac.py
```
Radar window opens and waits for the audio stream.

### Step 2 — Start sender on gaming laptop
```cmd
python sender_windows.py
```
Expected output:
```
Streaming via: Stereo Mix
Radar should show pulses on Mac now!
```

### Step 3 — Launch Valorant and play

---

## Frequency Finder (Calibration Tool)

Use this to verify footstep frequencies for your specific setup:

```bash
python3 freq_finder.py
```

Walk around in Valorant and watch which frequencies appear in the terminal. Then update `FOOTSTEP_LOW` and `FOOTSTEP_HIGH` in `receiver_mac.py` accordingly.

Measured Valorant footstep profile:
```
Base frequency:  86.1Hz
Harmonics:       129.2Hz, 172.3Hz, 215.3Hz
Detection range: 60-220Hz
Energy range:    0.001-0.014
```

---

## How Direction Detection Works

Left / Right — accurate
```
Measured from stereo channel energy difference.
Left channel louder  -> enemy on left
Right channel louder -> enemy on right
```

Front / Back — estimated
```
HRTF encodes direction via subtle frequency shaping.
Front sounds boost 4-8kHz slightly.
Back sounds cut 8-16kHz.
Requires Valorant HRTF ON to function.
```

---

## Why Not an Overlay

Vanguard (Valorant's anti-cheat) flags transparent windows drawn over the game. This project runs entirely on a separate physical device — Vanguard cannot detect it at all. It is the audio equivalent of putting a microphone next to your speaker and does not touch the game process in any way.

---

## Tuning

In `receiver_mac.py`:

```python
FOOTSTEP_LOW         = 60     # Hz lower bound
FOOTSTEP_HIGH        = 220    # Hz upper bound
FOOTSTEP_ENERGY_MIN  = 0.001  # sensitivity — lower = more sensitive
FOOTSTEP_ENERGY_MAX  = 0.014  # upper limit before classified as gunshot bleed
```

---

## Files

| File | Device | Purpose |
|---|---|---|
| `sender_windows.py` | Gaming laptop (Windows) | Captures WASAPI audio, streams over UDP |
| `receiver_mac.py` | Secondary laptop | Receives stream, runs FFT, shows radar |
| `freq_finder.py` | Secondary laptop | Calibration tool — prints dominant frequencies live |

---

## Roadmap

- Phase 2 — AI model trained on Valorant footstep recordings for better front/back accuracy
- Wired headphone support via Y-splitter + USB audio adapter
- Configurable UI (radar size, colors, sensitivity sliders)

---

## Disclaimer

This tool only reads audio that is already playing on your system. It does not interact with the Valorant game process, read game memory, or inject code of any kind. It is fully external and does not violate Valorant's Terms of Service.
