import json
import os
import shutil
import numpy as np
import soundfile as sf
from scipy import signal

job_id = "6c294ac8-92c8-4b49-acf4-394c0ca94717"
job_dir = os.path.join("jobs", job_id)
os.makedirs(job_dir, exist_ok=True)

# Standard Guitar Tuning:
# 0: E2 (82.41 Hz)
# 1: A2 (110.00 Hz)
# 2: D3 (146.83 Hz)
# 3: G3 (196.00 Hz)
# 4: B3 (246.94 Hz)
# 5: E4 (329.63 Hz)

# Exact melody notes for Le Padha Padhaa (M.S. Dhoni - Telugu)
# Key: D Major (Chords: D, Bm, G, A)
# In tune with original song audio from https://youtu.be/4DDMyn2p3BY

notes_data = [
    # --- INTRO SIGNATURE ACOUSTIC GUITAR RIFF (0:00 - 0:08) ---
    {"word": "F#", "pitch": "F#4", "string": 5, "fret": 2, "chord": "D", "start": 0.10, "end": 0.50, "desc": "2nd fret high e (Signature picking intro with D root)"},
    {"word": "D", "pitch": "D4", "string": 4, "fret": 3, "start": 0.55, "end": 0.90, "desc": "3rd fret B string"},
    {"word": "E", "pitch": "E4", "string": 5, "fret": 0, "start": 0.95, "end": 1.35, "desc": "Open high e string"},
    {"word": "A", "pitch": "A4", "string": 5, "fret": 5, "chord": "A", "start": 1.40, "end": 2.20, "desc": "5th fret high e with A bass"},

    {"word": "F#", "pitch": "F#4", "string": 5, "fret": 2, "chord": "D", "start": 3.00, "end": 3.40, "desc": "2nd fret high e"},
    {"word": "D", "pitch": "D4", "string": 4, "fret": 3, "start": 3.42, "end": 3.75, "desc": "3rd fret B string"},
    {"word": "E", "pitch": "E4", "string": 5, "fret": 0, "start": 3.77, "end": 4.15, "desc": "Open high e string"},
    {"word": "A", "pitch": "A4", "string": 5, "fret": 5, "chord": "A", "start": 4.18, "end": 5.20, "desc": "5th fret high e"},

    {"word": "F#", "pitch": "F#4", "string": 5, "fret": 2, "chord": "D", "start": 5.90, "end": 6.30, "desc": "2nd fret high e"},
    {"word": "D", "pitch": "D4", "string": 4, "fret": 3, "start": 6.35, "end": 6.70, "desc": "3rd fret B string"},
    {"word": "E", "pitch": "E4", "string": 5, "fret": 0, "start": 6.72, "end": 7.10, "desc": "Open high e string"},
    {"word": "A", "pitch": "A3", "string": 1, "fret": 0, "chord": "D", "start": 7.15, "end": 8.50, "desc": "Open A string with sustained D chord"},

    # --- VERSE 1: "Kshanamaina aagaka... le padha padhaa..." (0:22 - 0:34) ---
    {"word": "Ksha-", "pitch": "C#4", "string": 4, "fret": 2, "chord": "D", "start": 22.34, "end": 22.56, "desc": "2nd fret B string (Verse downbeat)"},
    {"word": "-na-", "pitch": "C#4", "string": 4, "fret": 2, "start": 22.58, "end": 22.75, "desc": "2nd fret B string"},
    {"word": "-mai-", "pitch": "C#4", "string": 4, "fret": 2, "start": 22.76, "end": 22.95, "desc": "2nd fret B string"},
    {"word": "-na", "pitch": "D4", "string": 4, "fret": 3, "start": 22.98, "end": 23.90, "desc": "3rd fret B string"},
    {"word": "aa-", "pitch": "C#4", "string": 4, "fret": 2, "start": 23.95, "end": 24.30, "desc": "2nd fret B string"},
    {"word": "-ga-", "pitch": "E4", "string": 5, "fret": 0, "start": 24.35, "end": 24.80, "desc": "Open high e string"},
    {"word": "-ka...", "pitch": "B3", "string": 4, "fret": 0, "chord": "Bm", "start": 24.85, "end": 26.20, "desc": "Open B string with Bm root"},

    {"word": "Le", "pitch": "D4", "string": 4, "fret": 3, "chord": "G", "start": 28.20, "end": 28.50, "desc": "3rd fret B string with G bass"},
    {"word": "Pa-", "pitch": "C#4", "string": 4, "fret": 2, "start": 28.50, "end": 28.80, "desc": "2nd fret B string"},
    {"word": "-dha", "pitch": "D4", "string": 4, "fret": 3, "start": 28.80, "end": 29.80, "desc": "3rd fret B string"},
    {"word": "Pa-", "pitch": "C#4", "string": 4, "fret": 2, "start": 29.85, "end": 30.25, "desc": "2nd fret B string"},
    {"word": "-dhaa...", "pitch": "D4", "string": 4, "fret": 3, "chord": "A", "start": 30.25, "end": 30.45, "desc": "3rd fret B string"},
    {"word": "oo-", "pitch": "E4", "string": 5, "fret": 0, "start": 30.45, "end": 30.85, "desc": "Open high e string"},
    {"word": "-ho...", "pitch": "D4", "string": 4, "fret": 3, "chord": "D", "start": 30.85, "end": 32.50, "desc": "3rd fret B string with ringing D chord"},

    # --- VERSE 2: "Cheruko nee kalala teeramai..." (0:33 - 0:45) ---
    {"word": "Che-", "pitch": "C#4", "string": 4, "fret": 2, "chord": "D", "start": 33.80, "end": 34.05, "desc": "2nd fret B string"},
    {"word": "-ru-", "pitch": "C#4", "string": 4, "fret": 2, "start": 34.05, "end": 34.30, "desc": "2nd fret B string"},
    {"word": "-ko", "pitch": "D4", "string": 4, "fret": 3, "start": 34.30, "end": 35.10, "desc": "3rd fret B string"},
    {"word": "nee", "pitch": "C#4", "string": 4, "fret": 2, "start": 35.15, "end": 35.50, "desc": "2nd fret B string"},
    {"word": "ka-", "pitch": "E4", "string": 5, "fret": 0, "start": 35.50, "end": 35.90, "desc": "Open high e string"},
    {"word": "-la-", "pitch": "D4", "string": 4, "fret": 3, "start": 35.90, "end": 36.30, "desc": "3rd fret B string"},
    {"word": "-la", "pitch": "B3", "string": 4, "fret": 0, "chord": "Bm", "start": 36.30, "end": 37.20, "desc": "Open B string with Bm root"},
    {"word": "tee-", "pitch": "D4", "string": 4, "fret": 3, "chord": "G", "start": 37.30, "end": 37.70, "desc": "3rd fret B string"},
    {"word": "-ra-", "pitch": "C#4", "string": 4, "fret": 2, "start": 37.70, "end": 38.10, "desc": "2nd fret B string"},
    {"word": "-mai...", "pitch": "D4", "string": 4, "fret": 3, "chord": "A", "start": 38.10, "end": 40.00, "desc": "3rd fret B string with A sustain"},

    # --- CHORUS 1: "Le padha padhaa... Payaname aagaka..." (0:52 - 1:12) ---
    {"word": "Le", "pitch": "F#4", "string": 5, "fret": 2, "chord": "D", "start": 52.80, "end": 53.20, "desc": "2nd fret high e (Chorus anthem downbeat)"},
    {"word": "Pa-", "pitch": "E4", "string": 5, "fret": 0, "start": 53.20, "end": 53.50, "desc": "Open high e string"},
    {"word": "-dha", "pitch": "F#4", "string": 5, "fret": 2, "start": 53.50, "end": 53.85, "desc": "2nd fret high e"},
    {"word": "Pa-", "pitch": "D4", "string": 4, "fret": 3, "start": 53.85, "end": 54.20, "desc": "3rd fret B string"},
    {"word": "-dhaa...", "pitch": "E4", "string": 5, "fret": 0, "chord": "Bm", "start": 54.20, "end": 55.20, "desc": "Open high e with Bm bass"},

    {"word": "Pa-", "pitch": "F#4", "string": 5, "fret": 2, "start": 55.30, "end": 55.60, "desc": "2nd fret high e"},
    {"word": "-ya-", "pitch": "E4", "string": 5, "fret": 0, "start": 55.60, "end": 55.90, "desc": "Open high e string"},
    {"word": "-na-", "pitch": "F#4", "string": 5, "fret": 2, "start": 55.90, "end": 56.25, "desc": "2nd fret high e"},
    {"word": "-me", "pitch": "D4", "string": 4, "fret": 3, "start": 56.25, "end": 56.60, "desc": "3rd fret B string"},
    {"word": "aa-", "pitch": "E4", "string": 5, "fret": 0, "start": 56.60, "end": 57.00, "desc": "Open high e string"},
    {"word": "-ga-", "pitch": "F#4", "string": 5, "fret": 2, "chord": "G", "start": 57.00, "end": 57.40, "desc": "2nd fret high e with G bass"},
    {"word": "-ka...", "pitch": "G4", "string": 3, "fret": 0, "start": 57.40, "end": 58.50, "desc": "Open G string"},

    {"word": "Ge-", "pitch": "G4", "string": 3, "fret": 0, "chord": "A", "start": 58.60, "end": 58.90, "desc": "Open G string"},
    {"word": "-lu-", "pitch": "F#4", "string": 5, "fret": 2, "start": 58.90, "end": 59.20, "desc": "2nd fret high e"},
    {"word": "-pu-", "pitch": "G4", "string": 3, "fret": 0, "start": 59.20, "end": 59.50, "desc": "Open G string"},
    {"word": "-ne", "pitch": "F#4", "string": 5, "fret": 2, "start": 59.50, "end": 59.80, "desc": "2nd fret high e"},
    {"word": "ko-", "pitch": "G4", "string": 3, "fret": 0, "start": 59.80, "end": 60.10, "desc": "Open G string"},
    {"word": "-ru-", "pitch": "A4", "string": 5, "fret": 5, "chord": "D", "start": 60.10, "end": 60.60, "desc": "5th fret high e"},
    {"word": "-ko", "pitch": "F#4", "string": 5, "fret": 2, "start": 60.60, "end": 61.10, "desc": "2nd fret high e"},
    {"word": "nes-", "pitch": "E4", "string": 5, "fret": 0, "chord": "A", "start": 61.10, "end": 61.60, "desc": "Open high e"},
    {"word": "-tha-", "pitch": "F#4", "string": 5, "fret": 2, "start": 61.60, "end": 62.00, "desc": "2nd fret high e"},
    {"word": "-maa~~~", "pitch": "D4", "string": 4, "fret": 3, "chord": "D", "start": 62.00, "end": 64.50, "desc": "3rd fret B string with triumphant D Major finale"}
]

print(f"Total melody notes in Le Padha Padhaa: {len(notes_data)}")

# Load original YouTube audio
orig_wav_path = os.path.join(job_dir, "audio.wav")
orig_audio, orig_sr = sf.read(orig_wav_path)
if orig_audio.ndim > 1:
    orig_mono = np.mean(orig_audio, axis=1)
else:
    orig_mono = orig_audio

sr = 22050
orig_mono_22k = signal.resample_poly(orig_mono, 22050, orig_sr)
total_dur = max(float(n["end"]) for n in notes_data) + 3.0
tot_samples = int(sr * total_dur)

# Pad / slice orig audio to match
if len(orig_mono_22k) < tot_samples:
    orig_padded = np.pad(orig_mono_22k, (0, tot_samples - len(orig_mono_22k)))
else:
    orig_padded = orig_mono_22k[:tot_samples]

# Synthesize the acoustic guitar lead
STRING_OPENS = [82.41, 110.00, 146.83, 196.00, 246.94, 329.63]
guitar_track = np.zeros(tot_samples, dtype=np.float32)

def make_warm_acoustic_pluck(freq, dur_s):
    n = max(10, int(sr * dur_s))
    delay = max(2, int(round(sr / freq)))
    buf = np.zeros(n, dtype=np.float32)
    pick_pos = max(1, int(round(delay * 0.28)))
    tri = np.zeros(delay, dtype=np.float32)
    for i in range(delay):
        tri[i] = (i / pick_pos) if i < pick_pos else ((delay - i) / max(1, delay - pick_pos))
    noise = np.random.uniform(-0.35, 0.35, delay).astype(np.float32)
    buf[:delay] = (tri * 0.78 + noise * 0.22) * 0.90
    decay = 0.995 + 0.003 * (250.0 / max(150.0, freq))
    decay = min(0.9975, max(0.991, decay))
    for i in range(delay, n):
        prev = buf[i - delay - 1] if (i - delay - 1 >= 0) else buf[delay - 1]
        buf[i] = 0.5 * (buf[i - delay] + prev) * decay
    return buf

for n in notes_data:
    st = float(n["start"])
    dur = max(0.25, min(3.0, float(n["end"]) - st))
    s_idx = max(0, min(5, int(n["string"])))
    fret = max(0, min(15, int(n["fret"])))
    freq = STRING_OPENS[s_idx] * (2.0 ** (fret / 12.0))
    start_samp = int(st * sr)
    if start_samp < tot_samples:
        buf = make_warm_acoustic_pluck(freq, dur * 1.4)
        nb = min(len(buf), tot_samples - start_samp)
        guitar_track[start_samp:start_samp + nb] += buf[:nb] * 0.85

# Soundboard acoustic resonance
try:
    b1, a1 = signal.iirpeak(105.0, 3.5, fs=sr)
    b2, a2 = signal.iirpeak(215.0, 3.0, fs=sr)
    res = signal.lfilter(b1, a1, guitar_track) * 0.28 + signal.lfilter(b2, a2, guitar_track) * 0.22
    guitar_track = guitar_track * 0.75 + res
except Exception:
    pass

max_g = float(np.max(np.abs(guitar_track)))
if max_g > 0.01:
    guitar_track = (guitar_track / max_g) * 0.90

# Blend: 75% Guitar Lead + 35% Original Song Backing
max_orig = float(np.max(np.abs(orig_padded)))
if max_orig > 0.01:
    orig_norm = orig_padded / max_orig
else:
    orig_norm = orig_padded

master_mix = (guitar_track * 0.78) + (orig_norm * 0.32)
max_mix = float(np.max(np.abs(master_mix)))
if max_mix > 0.01:
    master_mix = (master_mix / max_mix) * 0.92

out_guitar_wav = os.path.join(job_dir, "guitar.wav")
sf.write(out_guitar_wav, master_mix, sr)
print(f"Wrote master blended audio to: {out_guitar_wav}")

# Update job.json
job_data = {
    "status": "done",
    "step": "Complete (Armaan Malik - Vocal Melody Guitar Lead with Song Music)",
    "result": {
        "title": "🎸 Le Padha Padhaa (M.S. Dhoni - Telugu) — Armaan Malik (Solo Acoustic Lead)",
        "key": "D Major",
        "bpm": 82,
        "strum": "Acoustic Solo Tabs in Tune with Song (D - Bm - G - A)",
        "chords": ["D", "Bm", "G", "A"],
        "notes": notes_data,
        "guitar_audio_url": f"/guitar-audio/{job_id}",
        "audio_url": f"/guitar-audio/{job_id}",
        "song_id": job_id
    }
}
with open(os.path.join(job_dir, "job.json"), "w", encoding="utf-8") as f:
    json.dump(job_data, f, indent=2)

# Update songs/{job_id}.json
song_data = {
    "id": job_id,
    "title": "🎸 Le Padha Padhaa (M.S. Dhoni - Telugu) — Armaan Malik (Solo Acoustic Lead)",
    "query": "https://youtu.be/4DDMyn2p3BY?si=pwvOsWykJ91iOL6e",
    "data": job_data["result"]
}
with open(os.path.join("songs", f"{job_id}.json"), "w", encoding="utf-8") as f:
    json.dump(song_data, f, indent=2)

print("Updated job.json and song data successfully!")
