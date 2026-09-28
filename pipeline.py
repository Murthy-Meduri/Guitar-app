"""
Sargam Strings — audio-to-guitar-tab pipeline.
"""

import os
import math
import subprocess
import tempfile
import requests
from dataclasses import dataclass
from typing import List, Optional


# ---------------------------------------------------------------------------
# 1. Media retrieval
# ---------------------------------------------------------------------------

def get_youtube_url_via_api(query: str, api_key: str) -> Optional[str]:
    """Uses YouTube Data API v3 to convert search terms to an exact URL."""
    try:
        search_url = "https://www.googleapis.com/youtube/v3/search"
        params = {
            "part": "snippet",
            "q": query,
            "type": "video",
            "maxResults": 1,
            "key": api_key,
        }
        res = requests.get(search_url, params=params, timeout=5)
        if res.status_code == 200:
            items = res.json().get("items", [])
            if items:
                video_id = items[0]["id"]["videoId"]
                return f"https://www.youtube.com/watch?v={video_id}"
    except Exception as e:
        print(f"YouTube API lookup failed, falling back to ytsearch: {e}")
    return None


def retrieve_audio(youtube_url_or_query: str, out_dir: str) -> str:
    """Download best-quality audio via yt-dlp using YouTube API, local/env cookies, or fallback args."""
    target = youtube_url_or_query

    # Clean YouTube tracking parameters like ?si=... or &si=... from input URLs
    if target.startswith("http"):
        target = target.split("?si=")[0].split("&si=")[0]

    # If it's a search query, resolve to video URL via YouTube Data API v3
    if not target.startswith("http"):
        api_key = os.getenv("YOUTUBE_API_KEY", "AIzaSyC8FCz8lLbeYzq8UrME24FI8RZoqeZNzKc")
        if api_key:
            resolved_url = get_youtube_url_via_api(target, api_key)
            if resolved_url:
                target = resolved_url
            else:
                target = f"ytsearch1:{target}"
        else:
            target = f"ytsearch1:{target}"

    out_template = os.path.join(out_dir, "source.%(ext)s")

    cmd = [
        'yt-dlp',
        '-x',
        '--audio-format', 'wav',
        '--audio-quality', '0',
        '--no-check-certificates',
        '--no-playlist',
    ]

    # Handle cookies via repo file, Render Secret file, or environment variable
    repo_cookie_file = os.path.join(os.path.dirname(__file__), "youtube_cookies.txt")
    secret_cookie_file = "/etc/secrets/youtube_cookies.txt"
    cookies_env = os.getenv("YOUTUBE_COOKIES")

    cookie_file_path = None

    if os.path.exists(repo_cookie_file):
        cmd.extend(['--cookies', repo_cookie_file])
    elif os.path.exists(secret_cookie_file):
        cmd.extend(['--cookies', secret_cookie_file])
    elif cookies_env:
        cookie_file_path = os.path.join(out_dir, "youtube_cookies.txt")
        with open(cookie_file_path, "w", encoding="utf-8") as f:
            f.write(cookies_env)
        cmd.extend(['--cookies', cookie_file_path])
    else:
        # Improved client rotation flags when cookies are absent
        cmd.extend([
            '--extractor-args', 'youtube:player_client=ios,android,web_embedded',
            '--user-agent', 'Mozilla/5.0 (iPhone; CPU iPhone OS 17_5_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.5 Mobile/15E148 Safari/604.1'
        ])

    cmd.extend(['-o', out_template, target])

    # Execute subprocess and capture stderr output for detailed error messages
    result = subprocess.run(cmd, capture_output=True, text=True)

    if cookie_file_path and os.path.exists(cookie_file_path):
        os.remove(cookie_file_path)

    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp failed with exit code {result.returncode}.\nError logs:\n{result.stderr}")

    wav_path = os.path.join(out_dir, "source.wav")
    if not os.path.exists(wav_path):
        raise FileNotFoundError("yt-dlp did not produce the expected wav file")
    return wav_path


# ---------------------------------------------------------------------------
# 2. Vocal / lead-melody separation
# ---------------------------------------------------------------------------

def separate_vocals(wav_path: str, out_dir: str) -> str:
    cmd = ["demucs", "-n", "htdemucs", "--two-stems", "vocals",
           "-o", out_dir, wav_path]
    subprocess.run(cmd, check=True, capture_output=True)
    stem_name = os.path.splitext(os.path.basename(wav_path))[0]
    vocals_path = os.path.join(out_dir, "htdemucs", stem_name, "vocals.wav")
    if not os.path.exists(vocals_path):
        raise FileNotFoundError("Demucs did not produce a vocals stem")
    return vocals_path


# ---------------------------------------------------------------------------
# 3. Pitch detection
# ---------------------------------------------------------------------------

@dataclass
class PitchEvent:
    start: float
    end: float
    freq_hz: float
    note_name: str
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
# 4. Lyric transcription
# ---------------------------------------------------------------------------

@dataclass
class Word:
    text: str
    start: float
    end: float


def transcribe_lyrics(vocals_wav_path: str, language: Optional[str] = None) -> List[Word]:
    import whisper

    model = whisper.load_model("small")
    result = model.transcribe(vocals_wav_path, language=language, word_timestamps=True)

    words: List[Word] = []
    for segment in result["segments"]:
        for w in segment.get("words", []):
            words.append(Word(text=w["word"].strip(), start=w["start"], end=w["end"]))
    return words


# ---------------------------------------------------------------------------
# 5. Alignment
# ---------------------------------------------------------------------------

@dataclass
class AlignedNote:
    word: str
    start: float
    end: float
    note_name: str
    octave: int


def align(words: List[Word], pitch_events: List[PitchEvent]) -> List[AlignedNote]:
    aligned: List[AlignedNote] = []
    for w in words:
        overlapping = [e for e in pitch_events if e.start < w.end and e.end > w.start]
        if not overlapping:
            nearest = min(pitch_events, key=lambda e: abs(e.start - w.start), default=None)
            overlapping = [nearest] if nearest else []
        if not overlapping:
            continue

        overlapping.sort(key=lambda e: e.start)
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
# 6. Fretboard optimization
# ---------------------------------------------------------------------------

STRINGS = [
    {"name": "E", "open_midi": 40},
    {"name": "A", "open_midi": 45},
    {"name": "D", "open_midi": 50},
    {"name": "G", "open_midi": 55},
    {"name": "B", "open_midi": 59},
    {"name": "E", "open_midi": 64},
]
MAX_FRET = 12


def note_to_midi(name: str, octave: int) -> int:
    return octave * 12 + NOTE_NAMES.index(name) + 12


@dataclass
class FretPosition:
    string_index: int
    fret: int


def optimize_fretting(notes: List[AlignedNote]) -> List[FretPosition]:
    FRET_HEIGHT_WEIGHT = 0.15
    n = len(notes)
    if n == 0:
        return []

    candidates: List[List[tuple]] = []
    for note in notes:
        target_midi = note_to_midi(note.note_name, note.octave)
        opts = []
        for si, s in enumerate(STRINGS):
            fret = target_midi - s["open_midi"]
            if 0 <= fret <= MAX_FRET:
                opts.append((si, fret))
        if not opts:
            target_midi += 12
            for si, s in enumerate(STRINGS):
                fret = target_midi - s["open_midi"]
                if 0 <= fret <= MAX_FRET:
                    opts.append((si, fret))
        candidates.append(opts or [(0, 0)])

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
                same_string_bonus = -1 if si == psi else 0
                cost = prev_cost + jump_cost + same_string_bonus + fret * FRET_HEIGHT_WEIGHT
                if cost < best_cost:
                    best_cost, best_prev = cost, (psi, pfret)
            dp[i][(si, fret)] = best_cost
            parent[i][(si, fret)] = best_prev

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

def process_song(youtube_url_or_query: str, language: Optional[str] = None) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        wav_path = retrieve_audio(youtube_url_or_query, tmp)
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
                "string": pos.string_index,
                "fret": pos.fret,
            })
        return {"source_wav": wav_path, "notes": notes_out}
