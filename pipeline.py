"""
Sargam Strings — audio-to-guitar-tab pipeline.

Stages:
  1. retrieve_audio()   -> download/locate source audio (yt-dlp)
  2. separate_vocals()  -> isolate lead melody from the mix (Demucs)
  3. detect_pitch()     -> per-frame f0 + note onsets (Basic Pitch / CREPE)
  4. transcribe_lyrics()-> word-level timestamps (Whisper)
  5. align()            -> merge pitch + lyric timing into per-syllable notes
  6. optimize_fretting() -> DP fingering optimizer -> string/fret per note
"""

import os
import math
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# 1. Media retrieval
# ---------------------------------------------------------------------------

def retrieve_audio(youtube_url_or_query: str, out_dir: str) -> str:
    """Download best-quality audio via yt-dlp. Accepts a URL or a search query
    (falls back to `ytsearch1:` for plain text queries)."""
    target = youtube_url_or_query
    if not target.startswith("http"):
        target = f"ytsearch1:{target}"

    out_template = os.path.join(out_dir, "source.%(ext)s")

    cmd = [
        'yt-dlp',
        '-x',
        '--audio-format', 'wav',
        '--audio-quality', '0',
        '--extractor-args', 'youtube:player_client=mweb,web',
        '--user-agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
        '--no-check-certificates',
        '-o', out_template,
        target
    ]
    
    subprocess.run(cmd, check=True, capture_output=True)
    wav_path = os.path.join(out_dir, "source.wav")
    if not os.path.exists(wav_path):
        raise FileNotFoundError("yt-dlp did not produce the expected wav file")
    return wav_path


# ---------------------------------------------------------------------------
# 2. Vocal / lead-melody separation
# ---------------------------------------------------------------------------

def separate_vocals(wav_path: str, out_dir: str) -> str:
    """Run Demucs (htdemucs model) and return the path to the isolated
    vocals/lead stem, which pitch detection runs on instead of the full mix."""
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
    import whisper

    model = whisper.load_model("small")
    result = model.transcribe(vocals_wav_path, language=language, word_timestamps=True)

    words: List[Word] = []
    for segment in result["segments"]:
        for w in segment.get("words", []):
            words.append(Word(text=w["word"].strip(), start=w["start"], end=w["end"]))
    return words


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
        # while keeping genuinely distinct pitches as separate notes.
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
       - a small per-note preference for staying near the nut (lower frets)
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
            # transpose up an octave if nothing on the neck reaches it
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
                "string": pos.string_index,   # 0=low E ... 5=high E
                "fret": pos.fret,
            })
        return {"source_wav": wav_path, "notes": notes_out}
