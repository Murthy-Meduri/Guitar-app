# Sargam Strings — backend pipeline

Turns a YouTube URL or search query into time-aligned (lyric, pitch, string/fret)
note data for the frontend player.

## What's real here, and what to expect on first run

This code was written against the documented/standard APIs of yt-dlp, Demucs,
Basic Pitch, and Whisper as of early 2026, but it has **not been executed** —
this authoring environment has no network access or GPU, so nothing here has
been run end-to-end. Treat the first run as integration testing:
- Library APIs (especially `basic_pitch.inference.predict`'s return shape and
  Demucs's CLI output path convention) occasionally change between versions —
  pin the versions in `requirements.txt` if a call breaks.
- Demucs and Whisper are slow on CPU. A GPU (even a modest consumer one)
  turns a multi-minute wait into single-digit seconds.

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # edit ALLOWED_ORIGINS / API_KEY / RATE_LIMIT_PER_HOUR
# ffmpeg must be installed on the system separately (apt install ffmpeg / brew install ffmpeg)
uvicorn main:app --host 0.0.0.0 --port 8000
```

Or with Docker: `docker build -t sargam-backend . && docker run -p 8000:8000 --env-file .env sargam-backend`

Then point the frontend's `API_BASE_URL` at this server.

## API shape (non-blocking)

`/process` is now fire-and-forget: it returns `{job_id, status:"queued"}` immediately
instead of holding the connection open for the minutes Demucs/Whisper can take.
Poll `GET /status/{job_id}` until `status` is `"done"` (result includes `notes`
and `audio_url`) or `"error"`. The frontend's `fetchBtn` handler already does
this polling loop for you.

`POST /songs` / `GET /songs/{id}` give simple cross-device song storage (a
JSON file per song) — an alternative to the frontend's browser-only
`localStorage`, for when you want a saved song to follow the user to a
different device.

## Security defaults

- `ALLOWED_ORIGINS` restricts CORS — set it to your actual frontend origin
  before deploying anywhere reachable by others; don't leave it as `*`.
- `API_KEY`, if set, requires every request to send `X-API-Key: <value>`.
- `RATE_LIMIT_PER_HOUR` caps `/process` calls per IP (in-memory — fine for a
  single instance; swap for a shared store like Redis if you run more than one).

## Legal note

Downloading audio from YouTube via yt-dlp sits in a legal gray zone under
YouTube's Terms of Service. This is fine for personal practice/research use;
don't build a product on top of it that redistributes other people's
copyrighted songs without a license.

## Architecture

```
YouTube URL/query
   -> yt-dlp            (retrieve_audio)
   -> Demucs             (separate_vocals — isolates lead melody from the mix)
   -> Basic Pitch         (detect_pitch — note events with onset/offset + confidence)
   -> Whisper             (transcribe_lyrics — word-level timestamps)
   -> align()             (merge lyric words with the pitch event under their time span)
   -> optimize_fretting() (DP: minimizes hand-position jumps across the neck)
   -> JSON note list + audio URL, served to the frontend
```

## Known limitations to fix before production use

- No caching — the same song gets fully reprocessed every time it's requested.
- Whisper's word-level timestamps are approximate on melismatic (sung/held)
  syllables; expect some manual correction needed on slower ballads.
- The in-memory `JOBS` dict and rate-limit log are lost on restart and don't
  share state across multiple instances — fine for one box, not for a
  horizontally-scaled deploy (move both to Redis/a database at that point).
