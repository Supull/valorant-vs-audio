"""
VALORANT SOUND RADAR — Windows Sender Script v3
================================================
Automatically captures Windows audio output
Works with speakers AND Bluetooth earphones
"""

import sounddevice as sd
import numpy as np
import socket

# ─── CONFIG ───────────────────────────────────────────────
MAC_IP      = "192.168.1.34"
PORT        = 5005
SAMPLE_RATE = 44100
CHUNK_SIZE  = 1024
CHANNELS    = 2
# ──────────────────────────────────────────────────────────

sock   = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
target = (MAC_IP, PORT)

def audio_callback(indata, frames, time_info, status):
    """Send each audio chunk over UDP to Mac"""
    audio_bytes = indata.astype(np.float32).tobytes()
    max_udp = 65507
    if len(audio_bytes) <= max_udp:
        sock.sendto(audio_bytes, target)
    else:
        for i in range(0, len(audio_bytes), max_udp):
            sock.sendto(audio_bytes[i:i + max_udp], target)

def try_open_stream(device_idx, use_loopback=False):
    """Try opening a stream, return it or None"""
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
            # Try different WasapiSettings signatures
            try:
                kwargs['extra_settings'] = sd.WasapiSettings(loopback=True)
            except TypeError:
                try:
                    ws = sd.WasapiSettings()
                    ws.loopback = True
                    kwargs['extra_settings'] = ws
                except:
                    pass  # skip loopback flag, try anyway

        stream = sd.InputStream(**kwargs)
        # Test it actually opens
        stream.start()
        stream.stop()
        return sd.InputStream(**kwargs)
    except Exception as e:
        return None

def main():
    print("=" * 52)
    print("  VALORANT SOUND RADAR — Windows Sender v3")
    print("=" * 52)
    print(f"📡 Streaming to Mac at {MAC_IP}:{PORT}\n")

    devices    = sd.query_devices()
    hostapis   = sd.query_hostapis()

    # Find WASAPI host api index
    wasapi_idx = None
    for i, api in enumerate(hostapis):
        if 'wasapi' in api['name'].lower():
            wasapi_idx = i
            break

    print(f"WASAPI host api index: {wasapi_idx}\n")

    stream      = None
    stream_name = ""

    # ── Strategy 1: Stereo Mix (works for speaker mode) ──
    print("Trying Strategy 1 — Stereo Mix (index 11)...")
    stream = try_open_stream(11)
    if stream:
        stream_name = "Stereo Mix"

    # ── Strategy 2: WASAPI output devices with loopback ──
    if stream is None and wasapi_idx is not None:
        print("Trying Strategy 2 — WASAPI output loopback...")
        for i, dev in enumerate(devices):
            if dev['hostapi'] == wasapi_idx:
                s = try_open_stream(i, use_loopback=True)
                if s:
                    stream = s
                    stream_name = f"WASAPI loopback: {dev['name']}"
                    break

    # ── Strategy 3: Current default output loopback ──────
    if stream is None:
        print("Trying Strategy 3 — Default output loopback...")
        try:
            default_out_idx = sd.default.device[1]
            s = try_open_stream(default_out_idx, use_loopback=True)
            if s:
                stream = s
                stream_name = f"Default output loopback: {devices[default_out_idx]['name']}"
        except:
            pass

    # ── Strategy 4: Headphones device 17 or 20 ───────────
    if stream is None:
        print("Trying Strategy 4 — Bluetooth headphone devices...")
        for idx in [17, 20, 10]:
            s = try_open_stream(idx, use_loopback=True)
            if s:
                stream = s
                stream_name = f"Headphone loopback (index {idx})"
                break

    # ── Strategy 5: Any input device as fallback ─────────
    if stream is None:
        print("Trying Strategy 5 — Any available input device...")
        for i, dev in enumerate(devices):
            if dev['max_input_channels'] >= 2:
                s = try_open_stream(i)
                if s:
                    stream = s
                    stream_name = f"Input fallback: {dev['name']}"
                    break

    if stream is None:
        print("\n❌ Could not open any audio stream.")
        print("\nFix — enable Stereo Mix manually:")
        print("  1. Right click speaker icon in taskbar")
        print("  2. Sounds → Recording tab")
        print("  3. Right click empty area → Show Disabled Devices")
        print("  4. Right click Stereo Mix → Enable → Set as Default")
        print("  5. Run this script again")
        return

    print(f"\n✅ Streaming via: {stream_name}")
    print(f"🚀 Radar should show pulses on Mac now!")
    print(f"   Press Ctrl+C to stop.\n")

    try:
        with stream:
            while True:
                sd.sleep(1000)
    except KeyboardInterrupt:
        print("\n⛔ Sender stopped.")
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()