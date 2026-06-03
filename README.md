# 🎯 Valorant Sound Radar

> A real-time external sound visualizer for Valorant — because the game has no visual audio indicator.

Streams game audio from your Windows gaming laptop to a secondary device over WiFi, runs FFT analysis to isolate footstep frequencies, and displays a live directional radar showing where sounds are coming from.

---

## 📸 What It Does

```
Gaming Laptop (Windows)          Secondary Laptop (Mac)
────────────────────────         ──────────────────────
Valorant plays audio      WiFi   Receives audio stream
       ↓                 ──────►        ↓
WASAPI captures it                 FFT analysis
       ↓                         splits into bands
Streams over UDP                       ↓
                                  Radar display
                                 shows direction
                                  of footsteps
```

- 🔴 **Red pulses** — Footsteps (60–220Hz, tuned to Valorant's exact footstep harmonics)
- 🟡 **Yellow pulses** — Gunshots / abilities (1500Hz+)
- 🔵 **Blue pulses** — Mid frequency sounds
- **Pulse size** — louder = bigger = closer
- **Pulse position** — left/right direction via stereo channel analysis
- **Front/back** — inferred from HRTF frequency shaping (requires HRTF ON in Valorant)

---

## 🛠 Requirements

### Gaming Laptop (Windows)
- Python 3.x
- `pip install sounddevice numpy`
- Stereo Mix enabled in Windows Recording devices

### Secondary Laptop (Mac or any OS)
- Python 3.x
- `pip3 install numpy pygame scipy`

---

## ⚙️ Setup

### 1. Enable Stereo Mix on Windows
```
Right click speaker icon → Sounds → Recording tab
Right click empty area → Show Disabled Devices
Right click Stereo Mix → Enable → Set as Default Device
```

### 2. Valorant Audio Settings
```
Settings → Audio → HRTF → ON
Windows → Spatial Sound → OFF  (important — disables double processing)
```

### 3. Find Your Secondary Laptop's IP
**On Mac:**
```bash
ipconfig getifaddr en0
```
**On Windows:**
```cmd
ipconfig
```
Look for IPv4 Address under WiFi adapter.

### 4. Set the IP in sender script
Open `sender_windows.py` and set:
```python
MAC_IP = "your.secondary.laptop.ip"
```

---

## 🚀 Running It

### Step 1 — Secondary laptop first
```bash
python3 receiver_mac.py
```
Radar window opens and waits for audio stream.

### Step 2 — Gaming laptop
```cmd
python sender_windows.py
```
Should output:
```
✅ Streaming via: Stereo Mix
🚀 Radar should show pulses on Mac now!
```

### Step 3 — Launch Valorant and play

---

## 🔍 Frequency Finder (Calibration Tool)

Use this to find exact footstep frequencies for your setup:

```bash
python3 freq_finder.py
```

Walk around in Valorant and watch which frequencies appear in terminal. Tell the script what your footstep frequencies are, then update `FOOTSTEP_LOW` and `FOOTSTEP_HIGH` in `receiver_mac.py`.

Our measured Valorant footstep profile:
```
Core:       86.1Hz  (base thud)
Harmonics:  129.2Hz, 172.3Hz, 215.3Hz
Range used: 60–220Hz
Energy:     0.001–0.014
```

---

## 🎮 How Direction Detection Works

**Left / Right** — accurate
```
Stereo channel energy difference
Left channel louder  → enemy on left
Right channel louder → enemy on right
```

**Front / Back** — estimated
```
HRTF encodes direction via frequency shaping
Front sounds boost 4–8kHz slightly
Back sounds cut 8–16kHz
Requires Valorant HRTF ON to work
```

---

## ❓ Why Not Just Use an Overlay

Vanguard (Valorant's anti-cheat) flags transparent windows drawn over the game. This project runs entirely on a **separate physical device** — Vanguard cannot detect it at all. It's the audio equivalent of putting a microphone next to your speaker.

---

## 🔧 Tuning

In `receiver_mac.py` you can adjust:

```python
FOOTSTEP_LOW         = 60     # Hz lower bound
FOOTSTEP_HIGH        = 220    # Hz upper bound
FOOTSTEP_ENERGY_MIN  = 0.001  # sensitivity (lower = more sensitive)
FOOTSTEP_ENERGY_MAX  = 0.014  # max before classified as gunshot
```

---

## 📁 Files

| File | Device | Purpose |
|---|---|---|
| `sender_windows.py` | Gaming laptop (Windows) | Captures WASAPI audio, streams over UDP |
| `receiver_mac.py` | Secondary laptop | Receives stream, runs FFT, shows radar |
| `freq_finder.py` | Secondary laptop | Calibration tool — prints live frequencies |

---

## ⚠️ Disclaimer

This tool only reads audio that is already playing on your system — it does not interact with the Valorant game process, read game memory, or inject anything. It is external hardware/software equivalent and does not violate Valorant's Terms of Service.

---

## 🗺 Roadmap

- [ ] Phase 2 — AI model trained on Valorant footstep recordings for better front/back accuracy
- [ ] Wired headphone support via Y-splitter + USB audio adapter
- [ ] Configurable UI (radar size, colors, sensitivity sliders)
- [ ] Windows receiver support (run both scripts on one machine)
- [ ] ESP32 physical radar device port