# Sargam Strings — Requirements, Handshake & Hand-off

**Status as of this document:** Frontend is complete and runnable today. The
algorithmic core of the backend (fretting optimizer, pitch/note math, lyric-
to-pitch alignment) **has now been executed and verified** — see §6.5. The
I/O-heavy stages (yt-dlp download, Demucs separation, Basic Pitch detection,
Whisper transcription) and the FastAPI service itself remain **unexecuted** —
this authoring environment has no network access and no `fastapi`/`pydantic`
available to install, so nothing that touches YouTube, a GPU model, or an
actual HTTP server has run end-to-end. Treat that portion as integration
testing, not a plug-and-play launch.

---

## 1. Final Scope (current, authoritative)

The product is a **YouTube-pipeline-only** guitar letter-note trainer, plus one
built-in offline demo song. There is no manual song-creation tool — that was
built, then explicitly removed at the product owner's request (see §5, History).

**In scope:**
- Paste a YouTube URL or search text → backend downloads audio, separates the
  lead vocal, detects pitch, transcribes lyrics, aligns the two, computes
  ergonomic guitar fingering, returns it all to the frontend.
- Frontend displays it as a 3-layer notebook sheet (lyric / pitch name /
  string-fret), a real 6-line ASCII tab view, and a chords-only view.
- Real-time synced playback: audio element drives a fretboard highlight and
  active-note highlight via timestamps.
- Click-to-play any note through a synthesized plucked-guitar tone (Web Audio),
  independent of the source recording.
- Loop A/B practice looping — works on both AI-loaded audio and the offline
  demo (synth-driven) playback.
- Metronome, chord diagrams (24 shapes), playback speed (0.5x/0.75x/1x),
  Verse/Chorus/Stanza section labels.
- One hard-coded offline demo: "Le Padha Padhaa" (D Major).

**Explicitly out of scope (removed):**
- Manual lyric entry / pitch assignment UI ("Song Creator")
- Mic-based hum-to-detect-pitch (per-syllable or whole-line)
- Browser-local song saving, JSON export/import
- Any workflow that requires the user to supply lyrics or pitches by hand

---

## 2. Functional Requirements (as specified across the project)

### 2.1 Media retrieval
- Accept a YouTube URL **or** a free-text search query.
- Resolve to a high-quality audio stream server-side.

### 2.2 Audio processing pipeline
- Separate lead vocal/melody from the full mix before pitch detection.
- Detect discrete note events (pitch + onset/offset + confidence), not just a
  raw pitch curve.
- Transcribe lyrics with **word-level timestamps**.
- Align lyric words to the pitch event(s) under their time span; a word held
  across two genuinely distinct pitches becomes two notes sharing that word
  (melisma handling), not one averaged/wrong note.

### 2.3 Fretboard optimization
- Map each detected pitch to a **playable** position on standard tuning
  (E2 A2 D3 G3 B3 E4), frets 0–12.
- Minimize hand-position jumps across the whole phrase (dynamic programming
  over string choice per note, not a greedy per-note pick) so fingerings stay
  ergonomic for a beginner.
- Output format: capital letter = string name, superscript number = fret
  (e.g. `E²`, `B⁰`).

### 2.4 Display
- Three stacked layers per line: lyric syllable, pitch letter name, string/fret.
- A genuine 6-line TAB view (one row per string, fret numbers positioned per
  note column) — not a layer-hiding trick.
- A chords-only view.
- Section labels (Verse/Chorus/Stanza) shown per line when present.

### 2.5 Audio & interaction
- HTML5/Web Audio playback of the real source recording once AI-processed,
  with play/pause, seek, and loop-section (A/B) tooling.
- Playback speed control: 0.5x / 0.75x / 1.0x, applied correctly (0.5x must
  play *slower*, not faster — this was a shipped bug, now fixed).
- Clicking any note/syllable triggers its own synthesized guitar pluck
  immediately, independent of source-audio playback.
- Real-time fretboard highlight follows whichever note is currently sounding,
  whether that's AI-audio-driven or synth-click-driven.

### 2.6 Fretboard UI
- Vector (SVG) 6-string neck, frets 0–12, nut, fret wires, inlay dots at
  3/5/7/9/12.
- Only the active string+fret is highlighted — no note-name clutter across
  the whole board at once.

### 2.7 Metadata & utilities
- Key, tempo (BPM), strumming pattern shown per song.
- Chord reference row with real finger-position diagrams (not a placeholder).
- Metronome with adjustable BPM, independent of song BPM.
- Notation toggle: Notebook / 6-Line TAB / Chords Only.

### 2.8 Demo data
- "Le Padha Padhaa" (*M.S. Dhoni: The Untold Story*) preloaded — D Major,
  D-DU-UDU strum, full lyrics/notes/frets, Verse/Verse/Chorus sections.

---

## 3. System Architecture

```
┌─────────────────────────┐         HTTPS/JSON          ┌──────────────────────────┐
│   Frontend (1 HTML file) │ ───────────────────────────▶│  Backend (FastAPI)       │
│  guitar-notes-app.html   │                              │  main.py + pipeline.py   │
│                          │◀─────────────────────────── │                          │
│  - Notation renderer     │      notes[] + audio_url     │  yt-dlp → Demucs →       │
│  - SVG fretboard         │                              │  Basic Pitch → Whisper → │
│  - Web Audio synth       │                              │  align() → DP fretting   │
│  - Chord diagrams        │                              │                          │
│  - Metronome             │                              │  In-memory job queue     │
└─────────────────────────┘                              └──────────────────────────┘
```

The frontend is a **fully self-contained single HTML file** — no build step, no
server required to run it standalone against the offline demo song. It becomes
"AI-powered" only once `API_BASE_URL` (top of its `<script>` block) points at a
running instance of the backend.

The backend is a **separate deployable service** (`/backend`), intended for a
host with real compute (GPU strongly preferred — Demucs + Whisper are slow on
CPU). It has no frontend of its own; it exists purely to serve the contract
below.

---

## 4. The Handshake — Frontend ⇄ Backend API Contract

This is the exact contract the frontend's `fetchBtn` handler and `wireAudioSync`
function expect. If you change the backend, keep this shape or update the
frontend to match.

### 4.1 Start a job

```
POST {API_BASE_URL}/process
Content-Type: application/json
X-API-Key: <value>            (only required if backend's API_KEY env is set)

Body:
{ "query": "<YouTube URL or free-text search>", "language": "en" }   // language optional

Response 200:
{ "job_id": "<uuid>", "status": "queued" }

Response 400: query missing/empty
Response 401: missing/invalid X-API-Key (only if API_KEY is configured server-side)
Response 429: rate limit exceeded (default 10/hour/IP, see RATE_LIMIT_PER_HOUR)
```

### 4.2 Poll job status

```
GET {API_BASE_URL}/status/{job_id}

Response 200, while running:
{ "status": "queued" | "running", "result": null, "error": null }

Response 200, on success:
{
  "status": "done",
  "result": {
    "notes": [
      { "word": "Le", "start": 12.04, "end": 12.31, "pitch": "F#", "string": 5, "fret": 2 },
      ...
    ],
    "audio_url": "/audio/<job_id>"
  },
  "error": null
}

Response 200, on failure:
{ "status": "error", "result": null, "error": "<message>" }

Response 404: unknown job_id
```

The frontend polls this every 2 seconds for up to 4 minutes (120 tries), then
gives up with a timeout message. Adjust both sides together if pipeline runs
routinely take longer.

**Field notes on a `notes[]` entry:**
- `string`: 0 = low E (6th string) … 5 = high E (1st string) — matches the
  backend's `STRINGS` array index, and the frontend's `STRINGS` array is
  defined identically so indices line up with no translation layer.
- `fret`: 0–12, already hand-position-optimized by the DP pass — the frontend
  does **not** re-run its own optimizer on AI-sourced notes, it trusts these
  values as-is (`notesToSong()` copies them straight through).
- `start`/`end`: seconds, relative to the returned audio file — used directly
  by `realAudio.currentTime` comparisons in `wireAudioSync()`.

### 4.3 Fetch the processed audio

```
GET {API_BASE_URL}/audio/{job_id}
→ audio/wav stream

Response 404: job not found or not finished yet
```

The frontend sets this directly as `<audio>` element's `src`; no auth header
is attached to this request in the current frontend code — if you require
`X-API-Key` on this route too, you'll need to add header support to the
`<audio>` tag's request (not natively supported — would require fetching the
blob via `fetch()` and using an object URL instead of a direct `src`).

### 4.4 Song persistence (built, not yet wired into the frontend)

```
POST {API_BASE_URL}/songs        { "title": "...", "data": {...} }  → { "song_id": "<uuid>" }
GET  {API_BASE_URL}/songs/{id}   → { "title": "...", "data": {...} }
```

These exist on the backend for future cross-device song sync but **the
frontend does not currently call them** — there's no "Save to cloud" button
since the Song Creator (and its save flow) was removed. Wire these up only if
a future requirement brings back some form of persisted custom song.

### 4.5 Health check

```
GET {API_BASE_URL}/health → { "status": "ok" }
```

Not called by the frontend; useful for your own uptime monitoring.

### 4.6 Auth & rate limiting

- Set `API_KEY` in the backend's environment to require `X-API-Key` on
  `/process` and `/songs`. Unset = open (dev-mode only, do not deploy publicly
  like this).
- `ALLOWED_ORIGINS` (comma-separated) restricts CORS — must include whatever
  origin the frontend HTML is actually served/published from.
- `RATE_LIMIT_PER_HOUR` caps `/process` calls per IP, in-memory (single
  instance only — see Known Limitations).

---

## 5. Requirement History (for context — not current scope)

The product went through these stages in order; included so a future
maintainer understands *why* certain code paths were built and then removed,
rather than assuming it was an oversight:

1. **v1** — fully static/manual: paste lyrics, manually assign a pitch letter
   per syllable, app auto-calculates fret positions.
2. **v2** — added mic-based pitch detection (hum a syllable → auto-fill the
   pitch dropdown) so manual entry didn't require music-theory knowledge.
3. **v3** — added the full AI backend (yt-dlp → Demucs → Basic Pitch → Whisper
   → alignment → DP fretting) so *no* manual input is needed for any song on
   YouTube.
4. **v4** — added an "Edit / correct" bridge so AI mistakes could be
   fixed by hand rather than forcing a full manual re-entry.
5. **v5 (current)** — product owner determined the manual Song Creator was
   redundant now that the AI pipeline exists, and requested its complete
   removal. All manual-entry, mic-detection, and correction-bridge code was
   deleted. **The trade-off:** there is currently no way to load or fix a song
   without a running backend — if the backend is down or not yet deployed,
   the app only shows the one offline demo song.

---

## 6. Hand-off Guide

### 6.1 What's already done
- `guitar-notes-app.html` — complete, runnable standalone right now (offline
  demo works with zero setup).
- `backend/main.py`, `backend/pipeline.py` — complete FastAPI service +
  processing pipeline, written against the documented APIs of yt-dlp, Demucs,
  Basic Pitch, and Whisper.
- `backend/requirements.txt`, `.env.example`, `Dockerfile`, `README.md` —
  setup/deploy instructions included.

### 6.2 What was actually executed and verified (this session)

This environment has no network access (so `yt-dlp`, `Demucs`, `Basic Pitch`,
and `Whisper` can't be installed or run against real audio) and can't install
`fastapi`/`pydantic` either (same reason), so the HTTP service itself
couldn't be started. What **could** run — pure Python/JS with no external
calls — was executed for real, and two genuine bugs were found and fixed as
a direct result:

- **DP fretting optimizer** (`optimize_fretting` in `pipeline.py`, mirrored
  in the frontend's `optimizeFrettingDP`) — run against the demo song's real
  note sequence. **Bug found:** the cost function had no preference for
  low frets, so it could arbitrarily lock onto a high-position fingering
  (e.g. frets 5–7) for a phrase that was fully playable in open position
  (frets 0–2), purely because the first note's candidates all tied at cost
  zero. **Fixed** by adding a small per-note fret-height term that breaks
  ties toward the nut without overriding real jump-minimization. Re-run
  confirmed the phrase now opens in the natural low position.
- **Cross-language agreement** — the fixed algorithm was extracted from the
  published HTML and run directly under Node, then diffed byte-for-byte
  against the Python output for the same input. **Confirmed identical**
  (same string/fret at every note, same total jump distance) — this is the
  first real evidence, not just code inspection, that the frontend and
  backend fretting logic agree.
- **`align()` melisma/jitter handling** — run against five constructed
  cases (simple 1:1, genuine melisma split, jitter-blip merge, real-gap
  re-attack, no-overlap fallback). **Bug found:** the jitter-merge check
  compared the wrong event's duration, so a short spurious blip immediately
  followed by the real sustained note failed to merge into one note. **Fixed**
  by checking the gap between consecutive events instead of either event's
  own duration. All five cases pass after the fix, including confirming a
  genuine same-pitch re-attack (with a real gap) correctly stays as two
  separate notes rather than over-merging.
- **`freq_to_note` / `note_to_midi`** — round-tripped against known
  reference pitches (A4=440Hz, C4≈261.63Hz, etc.). All correct.
- **Rate limiter and API-key check logic** (`main.py`'s `check_rate_limit`/
  `check_api_key`) — re-implemented in isolation with the identical logic
  and exercised directly: confirmed per-IP bucketing, the hourly cutoff, and
  key match/mismatch/missing all behave as intended. This validates the
  logic; it does not validate `main.py` running as an actual FastAPI service,
  which still requires `pip install fastapi` on a networked machine.
- **Both `pipeline.py` and `main.py`** — statically compiled
  (`python3 -m py_compile`) with no syntax errors.

**What this does *not* cover:** anything requiring yt-dlp, Demucs, Basic
Pitch, Whisper, or an actual running HTTP server. Those remain genuinely
untested — first deployment is still where you'll find out whether e.g.
Basic Pitch's `predict()` return shape still matches what `pipeline.py`
expects on whatever version actually gets installed.

### 6.3 What the next engineer must do before this is "live"
1. **Actually run the backend once, on real hardware.** Nothing in
   `pipeline.py` has executed end-to-end. Expect at least one library-version
   mismatch (Basic Pitch's `predict()` return shape and Demucs's CLI output
   path convention are the most likely to have drifted).
2. **Deploy it somewhere with a GPU** if you want processing times under a
   few minutes per song. CPU-only will work but will be slow.
3. **Set the three env vars** (`ALLOWED_ORIGINS`, `API_KEY`,
   `RATE_LIMIT_PER_HOUR`) — defaults are dev-only and unsafe to expose
   publicly as-is.
4. **Point `API_BASE_URL`** at the top of the frontend's `<script>` block to
   the deployed backend's address, then republish the artifact.
5. **Replace in-memory job/rate-limit state** with Redis or a database if you
   ever run more than one backend instance — current implementation loses all
   job state on restart and doesn't share state across instances.
6. **Add a job queue (Celery/RQ)** if you expect concurrent users — the
   current version processes one job at a time via FastAPI's
   `BackgroundTasks`, which is fine for a single demo user, not for real
   concurrent load.

### 6.4 Known limitations (carry these forward, don't silently "fix" by guessing)
- Whisper's word-level timestamps are approximate on melismatic/held
  syllables — expect manual correction needs on slower songs (and remember:
  there's currently no in-app way to correct them, per §5).
- No caching — identical song requests are fully reprocessed every time.
- Chord diagram library covers 24 common shapes; anything else shows "no
  diagram yet."
- The SVG fretboard uses even fret spacing for phone-screen legibility, not
  true logarithmic fret spacing.
- Downloading YouTube audio via yt-dlp sits in a legal gray zone under
  YouTube's ToS — fine for personal/research use, not for redistributing
  others' copyrighted songs without a license.
- The `FRET_HEIGHT_WEIGHT` tie-breaking constant (0.15, in both
  `pipeline.py` and the frontend's `optimizeFrettingDP`) was chosen to bias
  toward open position without overriding real jump-minimization, verified
  against one test phrase. It hasn't been tuned against a broader set of
  real songs — revisit if fretting choices feel off on material very
  different from the demo song's range.

### 6.5 File manifest

| File | Purpose |
|---|---|
| `guitar-notes-app.html` | Entire frontend; publish as-is via the Artifact tool |
| `backend/main.py` | FastAPI service: job queue, auth, rate limiting, song storage |
| `backend/pipeline.py` | Retrieval → separation → pitch detection → transcription → alignment → fretting |
| `backend/requirements.txt` | Python dependencies, pinned |
| `backend/.env.example` | Required environment variables, copy to `.env` |
| `backend/Dockerfile` | Container build for deployment |
| `backend/README.md` | Setup, architecture diagram, security defaults, limitations |
