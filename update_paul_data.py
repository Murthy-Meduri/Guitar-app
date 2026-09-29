import json
import os
import shutil
import numpy as np
import soundfile as sf
from pipeline import synthesize_guitar_audio

# Define the exact note structure matching Paul Acoustic Tabs (YouTube Short 4t-xuv9Hw68)
# Song: "Oh Priya Priya" from Ishq (2012)
# Tuning: Standard, Lead played in 10th-15th fret box (Strings B and high e)

lines_data = [
    # Line 1: Signature Chorus Hook 1
    # "Oh Priya Priya..."
    [
        {"word": "Oh", "pitch": "Eb5", "string": 5, "fret": 11, "chord": "Cm", "start": 0.95, "end": 1.40, "desc": "11th fret high e (Vocal 'Oh' with Cm bass)"},
        {"word": "Pri-", "pitch": "F5", "string": 5, "fret": 13, "start": 1.40, "end": 1.75, "desc": "13th fret high e"},
        {"word": "-ya", "pitch": "G5", "string": 5, "fret": 15, "start": 1.75, "end": 2.30, "desc": "15th fret high e (Vocal vibrato peak)"},
        {"word": "Pri-ya", "pitch": "Eb5", "string": 5, "fret": 11, "chord": "Cm", "start": 2.30, "end": 3.00, "desc": "11th fret high e"}
    ],
    # Line 2: Signature Chorus Hook 2
    # "Oh my dear Priya..."
    [
        {"word": "Oh", "pitch": "Bb4", "string": 4, "fret": 11, "chord": "Bb", "start": 3.50, "end": 3.75, "desc": "11th fret B string"},
        {"word": "my", "pitch": "C5", "string": 4, "fret": 13, "start": 3.75, "end": 4.05, "desc": "13th fret B string"},
        {"word": "dear", "pitch": "Bb4", "string": 4, "fret": 11, "start": 4.05, "end": 4.60, "desc": "11th fret B string"},
        {"word": "Pri-", "pitch": "D5", "string": 5, "fret": 10, "chord": "Gm", "start": 6.05, "end": 6.35, "desc": "10th fret high e"},
        {"word": "-ya", "pitch": "C5", "string": 5, "fret": 8, "start": 6.35, "end": 6.65, "desc": "8th fret high e"},
        {"word": "Pri-", "pitch": "D5", "string": 5, "fret": 10, "start": 6.65, "end": 6.85, "desc": "10th fret high e slide"},
        {"word": "-ya...", "pitch": "Eb5", "string": 5, "fret": 11, "chord": "Cm", "start": 6.85, "end": 8.00, "desc": "11th fret high e ringing sustain"}
    ],
    # Line 3: Signature Chorus Hook Repeat 1
    # "Oh Priya Priya..."
    [
        {"word": "Oh", "pitch": "Eb5", "string": 5, "fret": 11, "chord": "Cm", "start": 10.80, "end": 11.30, "desc": "11th fret high e"},
        {"word": "Pri-", "pitch": "F5", "string": 5, "fret": 13, "start": 11.30, "end": 11.75, "desc": "13th fret high e"},
        {"word": "-ya", "pitch": "G5", "string": 5, "fret": 15, "start": 11.75, "end": 12.20, "desc": "15th fret high e"},
        {"word": "Pri-ya", "pitch": "Eb5", "string": 5, "fret": 11, "chord": "Cm", "start": 12.20, "end": 13.00, "desc": "11th fret high e"}
    ],
    # Line 4: Signature Chorus Hook Repeat 2
    # "Oh my dear Priya..."
    [
        {"word": "Oh", "pitch": "Bb4", "string": 4, "fret": 11, "chord": "Bb", "start": 13.15, "end": 13.65, "desc": "11th fret B string"},
        {"word": "my", "pitch": "C5", "string": 4, "fret": 13, "start": 13.65, "end": 13.95, "desc": "13th fret B string"},
        {"word": "dear", "pitch": "Bb4", "string": 4, "fret": 11, "start": 13.95, "end": 14.50, "desc": "11th fret B string"},
        {"word": "Pri-", "pitch": "D5", "string": 5, "fret": 10, "chord": "Gm", "start": 15.90, "end": 16.25, "desc": "10th fret high e"},
        {"word": "-ya", "pitch": "C5", "string": 5, "fret": 8, "start": 16.25, "end": 16.65, "desc": "8th fret high e"},
        {"word": "Pri-", "pitch": "D5", "string": 5, "fret": 10, "start": 16.65, "end": 16.85, "desc": "10th fret high e slide"},
        {"word": "-ya...", "pitch": "Eb5", "string": 5, "fret": 11, "chord": "Cm", "start": 16.85, "end": 18.00, "desc": "11th fret high e sustain"}
    ],
    # Line 5: Stanza / Pallavi 1
    # "Nee premalo manase..."
    [
        {"word": "Nee", "pitch": "Eb5", "string": 5, "fret": 11, "chord": "Ab", "start": 19.80, "end": 20.30, "desc": "11th fret high e (Stanza downbeat with Ab bass)"},
        {"word": "pre-", "pitch": "D5", "string": 5, "fret": 10, "start": 20.30, "end": 20.75, "desc": "10th fret high e"},
        {"word": "-ma-", "pitch": "Eb5", "string": 5, "fret": 11, "start": 20.75, "end": 20.90, "desc": "11th fret high e"},
        {"word": "-lo", "pitch": "F5", "string": 5, "fret": 13, "start": 20.90, "end": 21.20, "desc": "13th fret high e"},
        {"word": "ma-", "pitch": "Eb5", "string": 5, "fret": 11, "start": 21.20, "end": 21.50, "desc": "11th fret high e"},
        {"word": "-na-", "pitch": "D5", "string": 5, "fret": 10, "start": 21.50, "end": 21.70, "desc": "10th fret high e"},
        {"word": "-se...", "pitch": "C5", "string": 5, "fret": 8, "chord": "Bb", "start": 21.70, "end": 22.20, "desc": "8th fret high e"}
    ],
    # Line 6: Stanza / Pallavi 2
    # "munigindevela..."
    [
        {"word": "mu-", "pitch": "Eb5", "string": 5, "fret": 11, "chord": "Bb", "start": 22.20, "end": 22.75, "desc": "11th fret high e with Bb bass"},
        {"word": "-ni-", "pitch": "D5", "string": 5, "fret": 10, "start": 22.75, "end": 23.20, "desc": "10th fret high e"},
        {"word": "-gin-", "pitch": "Eb5", "string": 5, "fret": 11, "start": 23.20, "end": 23.40, "desc": "11th fret high e"},
        {"word": "-dee...", "pitch": "F5", "string": 5, "fret": 13, "start": 23.40, "end": 24.30, "desc": "13th fret high e"},
        {"word": "ve-", "pitch": "Eb5", "string": 5, "fret": 11, "chord": "Gm", "start": 24.65, "end": 25.20, "desc": "11th fret high e with Gm bass"},
        {"word": "-laa...", "pitch": "D5", "string": 5, "fret": 10, "start": 25.20, "end": 25.70, "desc": "10th fret high e"}
    ],
    # Line 7: Stanza / Pallavi 3
    # "telusa neekaina..."
    [
        {"word": "te-", "pitch": "Eb5", "string": 5, "fret": 11, "chord": "Gm", "start": 25.70, "end": 25.85, "desc": "11th fret high e"},
        {"word": "-lu-", "pitch": "F5", "string": 5, "fret": 13, "start": 25.85, "end": 26.15, "desc": "13th fret high e"},
        {"word": "-sa", "pitch": "Eb5", "string": 5, "fret": 11, "start": 26.15, "end": 26.40, "desc": "11th fret high e"},
        {"word": "nee-", "pitch": "D5", "string": 5, "fret": 10, "start": 26.40, "end": 27.15, "desc": "10th fret high e"},
        {"word": "-kai-", "pitch": "D5", "string": 5, "fret": 10, "start": 27.15, "end": 27.35, "desc": "10th fret high e"},
        {"word": "-na...", "pitch": "Eb5", "string": 5, "fret": 11, "chord": "Ab", "start": 27.35, "end": 28.50, "desc": "11th fret high e"}
    ],
    # Line 8: Stanza / Pallavi 4
    # "ontari oohallo..."
    [
        {"word": "on-", "pitch": "D5", "string": 5, "fret": 10, "chord": "Ab", "start": 29.25, "end": 29.55, "desc": "10th fret high e"},
        {"word": "-ta-", "pitch": "D5", "string": 5, "fret": 10, "start": 29.55, "end": 29.85, "desc": "10th fret high e"},
        {"word": "-ri", "pitch": "D5", "string": 5, "fret": 10, "start": 29.85, "end": 30.15, "desc": "10th fret high e"},
        {"word": "oo-", "pitch": "C5", "string": 5, "fret": 8, "start": 30.15, "end": 30.35, "desc": "8th fret high e"},
        {"word": "-hal-lo...", "pitch": "C5", "string": 5, "fret": 8, "chord": "Gm", "start": 30.35, "end": 31.40, "desc": "8th fret high e with Gm bass"}
    ],
    # Line 9: Stanza / Pallavi 5 (Resolution)
    # "voopirilo nuvvele Priya..."
    [
        {"word": "voo-", "pitch": "Eb5", "string": 5, "fret": 11, "chord": "Ab", "start": 31.85, "end": 32.35, "desc": "11th fret high e"},
        {"word": "-pi-ri-lo", "pitch": "Eb5", "string": 5, "fret": 11, "start": 32.35, "end": 32.65, "desc": "11th fret high e"},
        {"word": "nuv-ve-le", "pitch": "D5", "string": 5, "fret": 10, "chord": "Bb", "start": 32.65, "end": 33.45, "desc": "10th fret high e with Bb bass"},
        {"word": "Pri-ya~~~", "pitch": "C5", "string": 5, "fret": 8, "chord": "Cm", "start": 33.50, "end": 35.00, "desc": "8th fret high e with resonant Cm root finale"}
    ]
]

flat_notes = []
for line in lines_data:
    for n in line:
        flat_notes.append(n)

print(f"Total notes in Paul Acoustic Tabs transcription: {len(flat_notes)}")

# Update job.json
job_id = "eb60bc12-eb03-4e2c-b1ca-2c6bd2e935f8"
job_dir = os.path.join("jobs", job_id)
os.makedirs(job_dir, exist_ok=True)

# Copy paul_audio.wav as audio.wav and guitar.wav
shutil.copy("paul_audio.wav", os.path.join(job_dir, "audio.wav"))
shutil.copy("paul_audio.wav", os.path.join(job_dir, "guitar.wav"))

# Synthesize pure guitar solo track as guitar_clean.wav
tot_dur = 35.5
synthesize_guitar_audio(flat_notes, tot_dur, os.path.join(job_dir, "guitar_clean.wav"), bpm=96, chords=["Cm", "Bb", "Ab", "Gm"])

job_data = {
    "status": "done",
    "step": "Complete (Paul Acoustic Tabs - Lyrics Melody in Tune with Song)",
    "result": {
        "title": "🎸 Oh Priya Priya (Ishq - 2012) — Paul Acoustic Tabs (Lyrics Melody Lead)",
        "key": "C Minor",
        "bpm": 96,
        "strum": "Acoustic Melody Lead (10th-15th Fret Box on High E & B Strings)",
        "chords": ["Cm", "Bb", "Ab", "Gm"],
        "notes": flat_notes,
        "guitar_audio_url": f"/guitar-audio/{job_id}",
        "audio_url": f"/guitar-audio/{job_id}",
        "song_id": job_id
    }
}

with open(os.path.join(job_dir, "job.json"), "w", encoding="utf-8") as f:
    json.dump(job_data, f, indent=2)

# Also update songs/eb60bc12-eb03-4e2c-b1ca-2c6bd2e935f8.json
song_file = os.path.join("songs", f"{job_id}.json")
song_data = {
    "id": job_id,
    "title": "🎸 Oh Priya Priya (Ishq - 2012) — Paul Acoustic Tabs (Lyrics Melody Lead)",
    "query": "https://youtube.com/shorts/4t-xuv9Hw68",
    "data": job_data["result"]
}
with open(song_file, "w", encoding="utf-8") as f:
    json.dump(song_data, f, indent=2)

print("Updated job.json and song file successfully!")
