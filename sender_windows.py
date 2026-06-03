"""
VALORANT SOUND RADAR — Windows Sender
======================================
Run this on your gaming laptop while playing Valorant.
Captures Windows audio output and streams to secondary laptop.

Setup:
  1. Set MAC_IP to your secondary laptop's IP address
  2. pip install sounddevice numpy
  3. python sender_windows.py
"""

import sounddevice as sd
import numpy as np
import socket

# ─── CONFIG ───────────────────────────────────────────────
MAC_IP      = "xxx.xxx.x.xx"   # change this to your secondary laptop's IP
PORT        = 5005
SAMPLE_RATE = 44100
CHUNK_SIZE  = 1024
CHANNELS    = 2
# ──────────────────────────────────────────────────────────

sock   = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
target = (MAC_IP, PORT)

def audio_callback(indata, frames, time_info, status):
    audio_bytes = indata.astype(np.float32).tobytes()
    max_udp = 65507
    if len(audio_bytes) <= max_udp:
        sock.sendto(audio_bytes, target)
    else:
        for i in range(0, len(audio_bytes), max_udp):
            sock.sendto(audio_bytes[i:i + max_udp], target)

def try_open_stream(device_idx, use_loopback=False):
    try:
        kwargs = dict(
            device=device_idx,
            channels=CHANNELS,
            samplerate=SAMPLE_RATE,
            blocksize=CHUNK_SIZE,
            dtype='float32',
            callback=audio_callback,
        )
        if use_loopback:
            try:
                kwargs['extra_settings'] = sd.WasapiSettings(loopback=True)
            except TypeError:
                try:
                    ws = sd.WasapiSettings()
                    ws.loopback = True
                    kwargs['extra_settings'] = ws
                except:
                    pass
        stream = sd.InputStream(**kwargs)
        stream.start()
        stream.stop()
        return sd.InputStream(**kwargs)
    except Exception:
        return None

def main():
    print("=" * 52)
    print("  VALORANT SOUND RADAR — Windows Sender")
    print("=" * 52)
    print(f"Streaming to {MAC_IP}:{PORT}\n")

    devices    = sd.query_devices()
    hostapis   = sd.query_hostapis()

    wasapi_idx = None
    for i, api in enumerate(hostapis):
        if 'wasapi' in api['name'].lower():
            wasapi_idx = i
            break

    stream      = None
    stream_name = ""

    # Strategy 1 — Stereo Mix by name
    print("Trying Strategy 1 — Stereo Mix...")
    for i, dev in enumerate(devices):
        if 'stereo mix' in dev['name'].lower() and dev['max_input_channels'] > 0:
            s = try_open_stream(i)
            if s:
                stream      = s
                stream_name = f"Stereo Mix: {dev['name']} (index {i})"
                break

    # Strategy 2 — WASAPI loopback
    if stream is None and wasapi_idx is not None:
        print("Trying Strategy 2 — WASAPI loopback...")
        for i, dev in enumerate(devices):
            if dev['hostapi'] == wasapi_idx:
                s = try_open_stream(i, use_loopback=True)
                if s:
                    stream      = s
                    stream_name = f"WASAPI loopback: {dev['name']}"
                    break

    # Strategy 3 — Default output loopback
    if stream is None:
        print("Trying Strategy 3 — Default output loopback...")
        try:
            idx = sd.default.device[1]
            s   = try_open_stream(idx, use_loopback=True)
            if s:
                stream      = s
                stream_name = f"Default output: {devices[idx]['name']}"
        except:
            pass

    # Strategy 4 — Any stereo input fallback
    if stream is None:
        print("Trying Strategy 4 — Any stereo input...")
        for i, dev in enumerate(devices):
            if dev['max_input_channels'] >= 2:
                s = try_open_stream(i)
                if s:
                    stream      = s
                    stream_name = f"Input fallback: {dev['name']}"
                    break

    if stream is None:
        print("\nCould not open any audio stream.")
        print("\nFix — enable Stereo Mix:")
        print("  1. Right click speaker icon in taskbar")
        print("  2. Sounds -> Recording tab")
        print("  3. Right click empty area -> Show Disabled Devices")
        print("  4. Right click Stereo Mix -> Enable -> Set as Default")
        print("  5. Run this script again")
        return

    print(f"\nStreaming via: {stream_name}")
    print(f"Radar should show pulses on secondary laptop.")
    print(f"Press Ctrl+C to stop.\n")

    try:
        with stream:
            while True:
                sd.sleep(1000)
    except KeyboardInterrupt:
        print("\nSender stopped.")
    except Exception as e:
        print(f"\nError: {e}")

if __name__ == "__main__":
    main()