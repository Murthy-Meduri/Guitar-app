"""
Sargam Strings — audio-to-guitar-tab pipeline.

Stages:
  1. retrieve_audio()   -> download/locate source audio (yt-dlp)
  2. separate_vocals()  -> isolate lead melody from the mix (Demucs)
  3. detect_pitch()     -> per-frame f0 + note onsets (Basic Pitch / CREPE)
  4. transcribe_lyrics()-> word-level timestamps (Whisper)
  5. align()            -> merge pitch + lyric timing into per-syllable notes
  6. optimize_fretting() -> DP fingering optimizer -> string/fret per note

NOTE: This is real, runnable code meant for a GPU-capable host (a laptop
with a decent CPU works for short clips, but Demucs/Whisper are much
faster on GPU). It has not been executed in this chat session — there is
no network or GPU available here — so treat first run as a shakeout:
pin library versions if a call signature has drifted since this was written.
"""

import os
import sys
import math
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# 1. Media retrieval
# ---------------------------------------------------------------------------

def extract_video_id(url_or_query: str) -> Optional[str]:
    import re
    patterns = [
        r'(?:v=|\/embed\/|\/v\/|youtu\.be\/|\/shorts\/)([a-zA-Z0-9_-]{11})',
        r'^([a-zA-Z0-9_-]{11})$'
    ]
    for p in patterns:
        m = re.search(p, url_or_query.strip())
        if m:
            return m.group(1)
    return None


def download_via_rapidapi(video_id: str, out_wav_path: str, api_key: str) -> bool:
    import urllib.request
    import json
    import time

    url = f"https://youtube-mp36.p.rapidapi.com/dl?id={video_id}"
    headers = {
        "x-rapidapi-key": api_key.strip(),
        "x-rapidapi-host": "youtube-mp36.p.rapidapi.com",
        "User-Agent": "Mozilla/5.0"
    }

    dl_link = None
    last_data = {}
    # Poll RapidAPI for up to 60 seconds (conversion takes a few seconds on new videos)
    for attempt in range(20):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=25) as res:
                last_data = json.loads(res.read().decode())
        except Exception:
            time.sleep(3)
            continue

        st = last_data.get("status")
        pr = last_data.get("progress", 0)
        link = last_data.get("link")

        if st == "ok" and (pr == 100 or pr is None) and link:
            dl_link = link
            break
        elif st == "fail":
            raise RuntimeError(f"RapidAPI conversion failed: {last_data.get('msg')}")

        time.sleep(3)

    if not dl_link:
        raise RuntimeError(f"Timed out waiting for RapidAPI audio conversion. Response: {last_data}")

    tmp_mp3 = out_wav_path.replace(".wav", ".mp3")
    downloaded = False

    # Download converted MP3 (retry up to 5 times if storage server has propagation delay)
    for dl_attempt in range(6):
        try:
            dl_req = urllib.request.Request(dl_link, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
            with urllib.request.urlopen(dl_req, timeout=40) as resp, open(tmp_mp3, "wb") as out_f:
                shutil.copyfileobj(resp, out_f)
            downloaded = True
            break
        except urllib.error.HTTPError as he:
            if he.code == 404 and dl_attempt < 5:
                time.sleep(3)
                continue
            raise

    if not downloaded or not os.path.exists(tmp_mp3):
        raise RuntimeError("Failed to download converted MP3 from RapidAPI storage")

    subprocess.run(["ffmpeg", "-y", "-i", tmp_mp3, "-ar", "44100", "-ac", "1", out_wav_path], check=True, capture_output=True)
    if os.path.exists(tmp_mp3):
        os.remove(tmp_mp3)
    return os.path.exists(out_wav_path)


def retrieve_audio(youtube_url_or_query: str, out_dir: str) -> str:
    """Download best-quality audio via RapidAPI fallback or yt-dlp."""
    target = youtube_url_or_query.strip()
    if target.startswith("http"):
        target = target.split("?si=")[0].split("&si=")[0]
    
    wav_path = os.path.join(out_dir, "source.wav")

    # 1. Check if RAPIDAPI_KEY is configured for cloud datacenter bypass
    rapidapi_key = os.environ.get("RAPIDAPI_KEY") or "3252427cd0msh1e6df2ca0f9eeb6p13901cjsn25e93e92623b"
    video_id = extract_video_id(target)
    if rapidapi_key and video_id:
        if download_via_rapidapi(video_id, wav_path, rapidapi_key):
            return wav_path

    # 2. Otherwise use yt-dlp (with optional YT_PROXY or cookies if provided)
    if not target.startswith("http"):
        target = f"ytsearch1:{target}"

    out_template = os.path.join(out_dir, "source.%(ext)s")
    ytdlp_bin = shutil.which("yt-dlp")
    ytdlp_base = [ytdlp_bin] if ytdlp_bin else [sys.executable, "-m", "yt_dlp"]
    cmd = ytdlp_base + [
        "-x", "--audio-format", "wav",
        "--audio-quality", "0",
        "--no-playlist",
        "--no-check-certificates",
        "-o", out_template,
    ]

    cookies_b64 = os.environ.get("YT_COOKIES_B64")
    cookies_file = os.environ.get("YT_COOKIES_FILE")

    if cookies_b64:
        import base64
        cookies_path = os.path.join(out_dir, "cookies.txt")
        with open(cookies_path, "wb") as f:
            f.write(base64.b64decode(cookies_b64))
        cmd += ["--cookies", cookies_path]
    elif cookies_file and os.path.exists(cookies_file):
        cmd += ["--cookies", cookies_file]

    if os.environ.get("YT_PROXY"):
        cmd += ["--proxy", os.environ["YT_PROXY"]]

    cmd.append(target)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        err_msg = (result.stderr or result.stdout or "Unknown error").strip()
        raise RuntimeError(
            f"yt-dlp failed (code {result.returncode}): {err_msg}\n"
            "To bypass cloud datacenter IP blocks, set RAPIDAPI_KEY or YT_PROXY in Render Environment Variables."
        )
    if not os.path.exists(wav_path):
        raise FileNotFoundError("Audio downloader did not produce the expected wav file")
    return wav_path


# ---------------------------------------------------------------------------
# 2. Vocal / lead-melody separation
# ---------------------------------------------------------------------------

def separate_vocals(wav_path: str, out_dir: str) -> str:
    """Demucs vocal separation is a heavy multi-layer neural network that takes 5-10+ minutes on CPU.
    Spotify Basic Pitch transcribes polyphonic audio mixes directly in 10-15 seconds.
    To ensure fast 20-30s turnaround, Demucs is bypassed by default on CPU cloud hosts unless USE_DEMUCS=1."""
    if os.environ.get("USE_DEMUCS", "0") != "1":
        return wav_path

    try:
        cmd = ["demucs", "-n", "htdemucs", "--two-stems", "vocals",
               "-o", out_dir, wav_path]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if res.returncode == 0:
            stem_name = os.path.splitext(os.path.basename(wav_path))[0]
            vocals_path = os.path.join(out_dir, "htdemucs", stem_name, "vocals.wav")
            if os.path.exists(vocals_path):
                return vocals_path
        print(f"Demucs returned code {res.returncode}: {res.stderr or res.stdout}")
    except Exception as e:
        print(f"Demucs execution failed or timed out: {e}")

    # Fall back to using the full audio mix directly
    return wav_path


# ---------------------------------------------------------------------------
# 3. Pitch detection
# ---------------------------------------------------------------------------

@dataclass
class PitchEvent:
    start: float      # seconds
    end: float        # seconds
    freq_hz: float
    note_name: str    # e.g. "F#4"
    confidence: float


NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def freq_to_note(freq_hz: float) -> str:
    if freq_hz <= 0:
        return ""
    midi = round(69 + 12 * math.log2(freq_hz / 440.0))
    name = NOTE_NAMES[midi % 12]
    octave = midi // 12 - 1
    return f"{name}{octave}"


def detect_pitch(vocals_wav_path: str) -> List[PitchEvent]:
    """Use Basic Pitch (Spotify) to get discrete note events with onset/offset
    and confidence. Basic Pitch is preferred over raw CREPE here because it
    already segments continuous pitch into note events, which is what the
    tab layer needs (CREPE alone gives a pitch curve, not note boundaries)."""
    from basic_pitch.inference import predict
    from basic_pitch import ICASSP_2022_MODEL_PATH

    model_output, midi_data, note_events = predict(
        vocals_wav_path, ICASSP_2022_MODEL_PATH
    )

    events: List[PitchEvent] = []
    for start_s, end_s, pitch_midi, amplitude, _bends in note_events:
        freq = 440.0 * (2 ** ((pitch_midi - 69) / 12))
        events.append(PitchEvent(
            start=start_s, end=end_s, freq_hz=freq,
            note_name=freq_to_note(freq), confidence=float(amplitude),
        ))
    events.sort(key=lambda e: e.start)
    return events


# ---------------------------------------------------------------------------
# 4. Lyric transcription (word-level timestamps)
# ---------------------------------------------------------------------------

@dataclass
class Word:
    text: str
    start: float
    end: float


def transcribe_lyrics(vocals_wav_path: str, language: Optional[str] = None) -> List[Word]:
    try:
        import whisper

        model_name = os.environ.get("WHISPER_MODEL", "tiny")
        model = whisper.load_model(model_name)
        result = model.transcribe(vocals_wav_path, language=language, word_timestamps=True, fp16=False)

        words: List[Word] = []
        for segment in result.get("segments", []):
            for w in segment.get("words", []):
                words.append(Word(text=w["word"].strip(), start=w["start"], end=w["end"]))
        return words
    except Exception as e:
        print(f"Whisper transcription skipped: {e}")
        return []


# ---------------------------------------------------------------------------
# 5. Alignment: assign each lyric word the pitch event(s) under its time span
# ---------------------------------------------------------------------------

@dataclass
class AlignedNote:
    word: str
    start: float
    end: float
    note_name: str   # pitch class only, e.g. "F#" (octave used for synthesis only)
    octave: int


def align(words: List[Word], pitch_events: List[PitchEvent]) -> List[AlignedNote]:
    """For each lyric word, attach the pitch event(s) under its time span.
    A word held across two clearly distinct pitches (melisma / a slide)
    is split into one AlignedNote per pitch, all sharing the word's text —
    this matches how these notebook-style sheets usually mark a held
    syllable riding across more than one note, instead of collapsing it
    into a single (wrong) average pitch."""
    aligned: List[AlignedNote] = []

    # If no lyrics were recognized (instrumental or solo), generate notes directly from pitch events
    if not words and pitch_events:
        for e in pitch_events:
            pitch_class = e.note_name[:-1] if e.note_name[-1].isdigit() else e.note_name[:-2]
            octave = int("".join(filter(str.isdigit, e.note_name)) or 4)
            aligned.append(AlignedNote(
                word=pitch_class, start=e.start, end=e.end,
                note_name=pitch_class, octave=octave,
            ))
        return aligned

    for w in words:
        overlapping = [e for e in pitch_events if e.start < w.end and e.end > w.start]
        if not overlapping:
            nearest = min(pitch_events, key=lambda e: abs(e.start - w.start), default=None)
            overlapping = [nearest] if nearest else []
        if not overlapping:
            continue

        overlapping.sort(key=lambda e: e.start)
        # collapse consecutive same-pitch-class events into one continuous
        # note when there's effectively no gap between them (detector jitter
        # producing several short back-to-back readings of the same pitch),
        # while keeping genuinely distinct pitches as separate notes. Checks
        # the GAP between events, not either event's own duration — checking
        # duration alone missed the common case of a short spurious blip
        # immediately followed by the real, longer sustained note.
        merged: List[PitchEvent] = []
        for e in overlapping:
            if merged and merged[-1].note_name[:-1] == e.note_name[:-1] and (e.start - merged[-1].end) < 0.05:
                prev = merged[-1]
                merged[-1] = PitchEvent(
                    start=prev.start, end=e.end, freq_hz=e.freq_hz,
                    note_name=e.note_name, confidence=max(prev.confidence, e.confidence),
                )
            else:
                merged.append(e)

        for e in merged:
            pitch_class = e.note_name[:-1] if e.note_name[-1].isdigit() else e.note_name[:-2]
            octave = int("".join(filter(str.isdigit, e.note_name)) or 4)
            aligned.append(AlignedNote(
                word=w.text, start=e.start, end=e.end,
                note_name=pitch_class, octave=octave,
            ))
    return aligned


# ---------------------------------------------------------------------------
# 6. Fretboard optimization — dynamic programming over string choice
# ---------------------------------------------------------------------------

STRINGS = [
    {"name": "E", "open_midi": 40},  # 6th, low E2
    {"name": "A", "open_midi": 45},
    {"name": "D", "open_midi": 50},
    {"name": "G", "open_midi": 55},
    {"name": "B", "open_midi": 59},
    {"name": "E", "open_midi": 64},  # 1st, high E4
]
MAX_FRET = 12


def note_to_midi(name: str, octave: int) -> int:
    return octave * 12 + NOTE_NAMES.index(name) + 12  # +12: MIDI octave offset (C-1=0)


@dataclass
class FretPosition:
    string_index: int  # 0 = low E (6th string) ... 5 = high E (1st string)
    fret: int


def optimize_fretting(notes: List[AlignedNote]) -> List[FretPosition]:
    """DP over (note index, string choice) minimizing:
       - impossible positions (fret out of 0..12, or negative) -> excluded
       - hand-position jump distance between consecutive notes (|fret_i - fret_{i-1}|)
       - a small per-note preference for staying near the nut (lower frets),
         used only to break ties among otherwise-equal-cost paths so the
         optimizer doesn't arbitrarily lock onto a high-position fingering
         when an equally-jump-efficient low-position one exists (verified
         bug: without this term, a phrase playable entirely at frets 0-2
         could resolve to frets 5-7 purely by tie-breaking accident on the
         first note, since all its candidates start at cost 0)
    This keeps fingerings physically playable in one hand position as long
    as possible, instead of a greedy pick that jumps all over the neck.
    """
    FRET_HEIGHT_WEIGHT = 0.15  # small: breaks ties toward the nut without overriding real jump minimization
    n = len(notes)
    if n == 0:
        return []

    # candidates[i] = list of (string_index, fret) playable for notes[i]
    candidates: List[List[tuple]] = []
    for note in notes:
        target_midi = note_to_midi(note.note_name, note.octave)
        opts = []
        for si, s in enumerate(STRINGS):
            fret = target_midi - s["open_midi"]
            if 0 <= fret <= MAX_FRET:
                opts.append((si, fret))
        if not opts:
            # transpose up an octave if nothing on the neck reaches it (e.g.
            # detected note was below the guitar's open-string range)
            target_midi += 12
            for si, s in enumerate(STRINGS):
                fret = target_midi - s["open_midi"]
                if 0 <= fret <= MAX_FRET:
                    opts.append((si, fret))
        candidates.append(opts or [(0, 0)])  # last-resort fallback

    INF = float("inf")
    dp = [{} for _ in range(n)]
    parent = [{} for _ in range(n)]
    for si, fret in candidates[0]:
        dp[0][(si, fret)] = fret * FRET_HEIGHT_WEIGHT

    for i in range(1, n):
        for si, fret in candidates[i]:
            best_cost, best_prev = INF, None
            for psi, pfret in candidates[i - 1]:
                prev_cost = dp[i - 1].get((psi, pfret), INF)
                if prev_cost == INF:
                    continue
                jump_cost = abs(fret - pfret)
                same_string_bonus = -1 if si == psi else 0  # slight preference to stay put
                cost = prev_cost + jump_cost + same_string_bonus + fret * FRET_HEIGHT_WEIGHT
                if cost < best_cost:
                    best_cost, best_prev = cost, (psi, pfret)
            dp[i][(si, fret)] = best_cost
            parent[i][(si, fret)] = best_prev

    # backtrack from the cheapest final state
    last_state = min(dp[n - 1], key=lambda k: dp[n - 1][k])
    path = [last_state]
    for i in range(n - 1, 0, -1):
        last_state = parent[i][last_state]
        path.append(last_state)
    path.reverse()

    return [FretPosition(string_index=si, fret=fret) for si, fret in path]


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def process_song_from_audio(wav_path: str, language: Optional[str] = None) -> dict:
    """Analyze an already-downloaded or uploaded audio file (WAV) directly."""
    with tempfile.TemporaryDirectory() as tmp:
        vocals_path = separate_vocals(wav_path, tmp)
        pitch_events = detect_pitch(vocals_path)
        words = transcribe_lyrics(vocals_path, language=language)
        aligned = align(words, pitch_events)
        frets = optimize_fretting(aligned)

        notes_out = []
        for note, pos in zip(aligned, frets):
            notes_out.append({
                "word": note.word,
                "start": note.start,
                "end": note.end,
                "pitch": note.note_name,
                "string": pos.string_index,   # 0=low E ... 5=high E
                "fret": pos.fret,
            })
        return {"source_wav": wav_path, "notes": notes_out}


def process_song(youtube_url_or_query: str, language: Optional[str] = None, target_audio_path: Optional[str] = None) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        wav_path = retrieve_audio(youtube_url_or_query, tmp)
        if target_audio_path:
            shutil.copy(wav_path, target_audio_path)
            persisted_wav = target_audio_path
        else:
            persisted_wav = wav_path

        return process_song_from_audio(persisted_wav, language=language)

