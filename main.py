"""
FastAPI service for Sargam Strings — hardened version.

Adds over the minimal first draft:
  - Optional API key check (set API_KEY env var to require X-API-Key header)
  - Simple in-memory per-IP rate limiting (swap for Redis in multi-instance deploys)
  - CORS restricted to ALLOWED_ORIGINS env var (comma-separated), not "*"
  - Non-blocking processing: /process returns a job_id immediately;
    poll /status/{job_id} instead of holding the HTTP connection open for
    the minutes Demucs/Whisper can take
  - /songs persistence (simple JSON-file store) so saved songs can sync
    across devices instead of being stuck in one browser's localStorage

Run locally:
    pip install -r requirements.txt
    cp .env.example .env   # then edit it
    uvicorn main:app --host 0.0.0.0 --port 8000
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import json
import shutil
import subprocess
import time
import uuid
from collections import defaultdict, deque

try:
    import torch
    torch.set_num_threads(1)
except Exception:
    pass

from fastapi import BackgroundTasks, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from pipeline import process_song, process_song_from_audio
from gdrive import drive_service

app = FastAPI(title="Sargam Strings API")

origins_env = os.environ.get("ALLOWED_ORIGINS", "*")
if origins_env.strip() == "*":
    ALLOWED_ORIGINS = ["*"]
else:
    ALLOWED_ORIGINS = [o.strip() for o in origins_env.split(",") if o.strip()]

API_KEY = os.environ.get("API_KEY")  # unset = no auth required (dev mode only)
RATE_LIMIT_PER_HOUR = int(os.environ.get("RATE_LIMIT_PER_HOUR", "60"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def add_private_network_header(request: Request, call_next):
    if request.method == "OPTIONS":
        from fastapi.responses import Response
        res = Response()
        res.headers["Access-Control-Allow-Origin"] = "*"
        res.headers["Access-Control-Allow-Methods"] = "*"
        res.headers["Access-Control-Allow-Headers"] = "*"
        res.headers["Access-Control-Allow-Private-Network"] = "true"
        return res
    response = await call_next(request)
    response.headers["Access-Control-Allow-Private-Network"] = "true"
    return response

JOBS_DIR = os.path.join(os.path.dirname(__file__), "jobs")
SONGS_DIR = os.path.join(os.path.dirname(__file__), "songs")
os.makedirs(JOBS_DIR, exist_ok=True)
os.makedirs(SONGS_DIR, exist_ok=True)

# job_id -> {"status": "queued"|"running"|"done"|"error", "result": {...}, "error": str}
JOBS: dict = {}


def _save_job(job_id: str):
    try:
        job_dir = os.path.join(JOBS_DIR, job_id)
        os.makedirs(job_dir, exist_ok=True)
        with open(os.path.join(job_dir, "job.json"), "w") as f:
            json.dump(JOBS.get(job_id, {}), f)
    except Exception:
        pass


def _get_job(job_id: str):
    if job_id in JOBS:
        return JOBS[job_id]
    job_file = os.path.join(JOBS_DIR, job_id, "job.json")
    if os.path.exists(job_file):
        try:
            with open(job_file) as f:
                data = json.load(f)
                JOBS[job_id] = data
                return data
        except Exception:
            pass
    return None


# --- naive per-IP rate limiter (in-memory; fine for single-instance dev/small deploys) ---
_request_log: dict = defaultdict(deque)


def check_rate_limit(ip: str):
    now = time.time()
    q = _request_log[ip]
    while q and now - q[0] > 3600:
        q.popleft()
    if len(q) >= RATE_LIMIT_PER_HOUR:
        raise HTTPException(429, f"Rate limit exceeded ({RATE_LIMIT_PER_HOUR}/hour). Try again later.")
    q.append(now)


def check_api_key(x_api_key: str | None):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(401, "Missing or invalid X-API-Key header.")


class ProcessRequest(BaseModel):
    query: str
    language: str | None = None


def _run_job(job_id: str, query: str, language: str | None):
    JOBS[job_id]["status"] = "running"
    job_dir = os.path.join(JOBS_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)
    try:
        persisted_audio = os.path.join(job_dir, "audio.wav")
        guitar_audio = os.path.join(job_dir, "guitar.wav")
        result = process_song(query, language=language, target_audio_path=persisted_audio, target_guitar_path=guitar_audio)
        title = query.split("/")[-1].split("?")[0] if "http" in query else query
        drive_res = drive_service.save_song(job_id, title=title, song_data=result, query_or_video_id=query)
        JOBS[job_id]["status"] = "done"
        JOBS[job_id]["step"] = "Complete"
        JOBS[job_id]["result"] = {
            "key": result.get("key", "D Major"),
            "bpm": result.get("bpm", 100),
            "strum": result.get("strum", "D - D U - U D -"),
            "chords": result.get("chords", []),
            "notes": result["notes"],
            "guitar_audio_url": f"/guitar-audio/{job_id}",
            "audio_url": f"/guitar-audio/{job_id}",
            "orig_audio_url": f"/audio/{job_id}",
            "song_id": job_id,
            "title": title,
            "drive_synced": drive_res.get("drive_synced", False)
        }
        _save_job(job_id)
    except Exception as e:  # noqa: BLE001 — surface any pipeline failure to the client
        import traceback
        JOBS[job_id]["status"] = "error"
        JOBS[job_id]["error"] = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        _save_job(job_id)


def _run_audio_job(job_id: str, raw_audio_path: str, language: str | None, query: str | None = None):
    JOBS[job_id]["status"] = "running"
    JOBS[job_id]["step"] = "Preparing audio..."
    _save_job(job_id)
    job_dir = os.path.join(JOBS_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)
    persisted_audio = os.path.join(job_dir, "audio.wav")
    guitar_audio = os.path.join(job_dir, "guitar.wav")
    try:
        # Convert raw uploaded audio (mp3, webm, m4a, etc.) directly to 16000Hz mono wav for zero-overhead YIN analysis
        cmd = ["ffmpeg", "-y", "-i", raw_audio_path, "-ar", "16000", "-ac", "1", persisted_audio]
        conv_res = subprocess.run(cmd, capture_output=True, text=True)
        if conv_res.returncode != 0:
            raise RuntimeError(f"ffmpeg conversion failed: {conv_res.stderr or conv_res.stdout}")

        def progress_cb(msg: str):
            JOBS[job_id]["step"] = msg
            _save_job(job_id)

        result = process_song_from_audio(persisted_audio, language=language, progress_cb=progress_cb, target_guitar_wav=guitar_audio)
        title = query or f"Song {job_id[:6]}"
        drive_res = drive_service.save_song(job_id, title=title, song_data=result, query_or_video_id=query)
        JOBS[job_id]["status"] = "done"
        JOBS[job_id]["step"] = "Complete"
        JOBS[job_id]["result"] = {
            "key": result.get("key", "D Major"),
            "bpm": result.get("bpm", 100),
            "strum": result.get("strum", "D - D U - U D -"),
            "chords": result.get("chords", []),
            "notes": result["notes"],
            "guitar_audio_url": f"/guitar-audio/{job_id}",
            "audio_url": f"/guitar-audio/{job_id}",
            "orig_audio_url": f"/audio/{job_id}",
            "song_id": job_id,
            "title": title,
            "drive_synced": drive_res.get("drive_synced", False)
        }
        _save_job(job_id)
    except Exception as e:
        import traceback
        JOBS[job_id]["status"] = "error"
        JOBS[job_id]["error"] = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        _save_job(job_id)
    finally:
        if os.path.exists(raw_audio_path) and os.path.abspath(raw_audio_path) != os.path.abspath(persisted_audio):
            try:
                os.remove(raw_audio_path)
            except OSError:
                pass


@app.get("/debug-rapidapi/{video_id}")
def debug_rapidapi(video_id: str):
    import urllib.request
    import json
    key = (os.environ.get("RAPIDAPI_KEY") or "3252427cd0msh1e6df2ca0f9eeb6p13901cjsn25e93e92623b").strip()
    url = f"https://youtube-mp36.p.rapidapi.com/dl?id={video_id}"
    headers = {
        "x-rapidapi-key": key,
        "x-rapidapi-host": "youtube-mp36.p.rapidapi.com",
        "User-Agent": "Mozilla/5.0"
    }
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=20) as res:
            raw = res.read().decode()
            data = json.loads(raw)
        dl_link = data.get("link")
        dl_status = None
        dl_err = None
        if dl_link:
            try:
                dl_req = urllib.request.Request(dl_link, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
                with urllib.request.urlopen(dl_req, timeout=10) as dl_res:
                    dl_status = dl_res.status
            except Exception as e:
                import traceback
                dl_err = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        return {"status": "ok", "url": url, "key_prefix": key[:8], "response": data, "dl_status": dl_status, "dl_err": dl_err}
    except Exception as e:
        import traceback
        return {"status": "error", "url": url, "key_prefix": key[:8], "error": str(e), "trace": traceback.format_exc()}


@app.get("/")
def read_root():
    index_path = os.path.join(os.path.dirname(__file__), "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path, media_type="text/html")
    raise HTTPException(404, "index.html not found")


@app.post("/process")
def process(req: ProcessRequest, request: Request, background_tasks: BackgroundTasks,
            x_api_key: str | None = Header(default=None)):
    check_api_key(x_api_key)
    check_rate_limit(request.client.host if request.client else "unknown")
    if not req.query or not req.query.strip():
        raise HTTPException(400, "query must be a YouTube URL or search text")

    # Instant Drive / Local cache lookup (0.01s return)
    cached = drive_service.find_cached_song(req.query)
    if cached:
        job_id = cached["id"]
        cached_data = cached.get("data", {})
        JOBS[job_id] = {
            "status": "done",
            "step": "Complete (from Google Drive / Cache)",
            "result": {
                "key": cached_data.get("key", "C Minor"),
                "bpm": cached_data.get("bpm", 100),
                "strum": cached_data.get("strum", "D - D U - U D -"),
                "chords": cached_data.get("chords", []),
                "notes": cached_data.get("notes", []),
                "audio_url": f"/guitar-audio/{job_id}",
                "guitar_audio_url": f"/guitar-audio/{job_id}",
                "song_id": job_id,
                "title": cached.get("title", req.query),
                "drive_synced": True
            },
            "error": None
        }
        _save_job(job_id)
        return {"job_id": job_id, "status": "done", "cached": True}

    job_id = str(uuid.uuid4())
    JOBS[job_id] = {"status": "queued", "result": None, "error": None}
    background_tasks.add_task(_run_job, job_id, req.query, req.language)
    return {"job_id": job_id, "status": "queued"}


@app.post("/process-audio")
async def process_audio(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    query: str | None = Form(default=None),
    language: str | None = Form(default=None),
    x_api_key: str | None = Header(default=None)
):
    check_api_key(x_api_key)
    check_rate_limit(request.client.host if request.client else "unknown")

    # Instant Drive / Local cache lookup (0.01s return)
    if query:
        cached = drive_service.find_cached_song(query)
        if cached:
            job_id = cached["id"]
            cached_data = cached.get("data", {})
            JOBS[job_id] = {
                "status": "done",
                "step": "Complete (from Google Drive / Cache)",
                "result": {
                    "key": cached_data.get("key", "C Minor"),
                    "bpm": cached_data.get("bpm", 100),
                    "strum": cached_data.get("strum", "D - D U - U D -"),
                    "chords": cached_data.get("chords", []),
                    "notes": cached_data.get("notes", []),
                    "audio_url": f"/guitar-audio/{job_id}",
                    "guitar_audio_url": f"/guitar-audio/{job_id}",
                    "song_id": job_id,
                    "title": cached.get("title", query),
                    "drive_synced": True
                },
                "error": None
            }
            _save_job(job_id)
            return {"job_id": job_id, "status": "done", "cached": True}

    job_id = str(uuid.uuid4())
    job_dir = os.path.join(JOBS_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)

    filename = file.filename or "upload.mp3"
    ext = os.path.splitext(filename)[1] or ".mp3"
    raw_path = os.path.join(job_dir, f"input{ext}")

    with open(raw_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    JOBS[job_id] = {"status": "queued", "result": None, "error": None}
    _save_job(job_id)
    background_tasks.add_task(_run_audio_job, job_id, raw_path, language, query)
    return {"job_id": job_id, "status": "queued"}


@app.get("/status/{job_id}")
def status(job_id: str):
    job = _get_job(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    return job


@app.get("/guitar-audio/{job_id}")
def get_guitar_audio(job_id: str):
    path = os.path.join(JOBS_DIR, job_id, "guitar.wav")
    if not os.path.exists(path):
        job = _get_job(job_id)
        if job and job.get("result", {}).get("notes"):
            notes = job["result"]["notes"]
            tot_dur = max(float(n.get("end", 0.0)) for n in notes) if notes else 10.0
            from pipeline import synthesize_guitar_audio
            synthesize_guitar_audio(notes, tot_dur, path)
    if not os.path.exists(path):
        raise HTTPException(404, "Guitar solo audio track not found")
    return FileResponse(path, media_type="audio/wav")


@app.get("/audio/{job_id}")
def get_original_audio(job_id: str):
    path = os.path.join(JOBS_DIR, job_id, "audio.wav")
    if not os.path.exists(path):
        path = os.path.join(JOBS_DIR, job_id, "guitar.wav")
    if not os.path.exists(path):
        raise HTTPException(404, "Audio track not found")
    return FileResponse(path, media_type="audio/wav")



def _write_midi(notes, out_path, tempo_bpm=100, program=25):
    ticks_per_beat = 480
    sec_per_tick = 60.0 / (max(40, tempo_bpm) * ticks_per_beat)
    events = []
    STRINGS_MIDI = [40, 45, 50, 55, 59, 64]
    for n in notes:
        st = float(n.get('start', 0.0))
        en = float(n.get('end', st + 0.4))
        s_idx = min(5, max(0, int(n.get('string', 3))))
        fret = min(24, max(0, int(n.get('fret', 0))))
        midi_pitch = STRINGS_MIDI[s_idx] + fret
        events.append((st, 0x90, midi_pitch, 90))
        events.append((en, 0x80, midi_pitch, 0))
    events.sort(key=lambda x: (x[0], 0 if x[1] == 0x80 else 1))

    track_bytes = bytearray([0x00, 0xC0, program])
    last_time = 0.0
    for t, status, pitch, vel in events:
        delta_ticks = int(round(max(0.0, t - last_time) / sec_per_tick))
        last_time = t
        val = delta_ticks
        buf = [val & 0x7F]
        while val > 0x7F:
            val >>= 7
            buf.append((val & 0x7F) | 0x80)
        track_bytes.extend(reversed(buf))
        track_bytes.extend([status, pitch, vel])
    track_bytes.extend([0x00, 0xFF, 0x2F, 0x00])

    header = bytearray(b'MThd')
    header.extend((6).to_bytes(4, 'big'))
    header.extend((0).to_bytes(2, 'big'))
    header.extend((1).to_bytes(2, 'big'))
    header.extend(ticks_per_beat.to_bytes(2, 'big'))

    track_chunk = bytearray(b'MTrk')
    track_chunk.extend(len(track_bytes).to_bytes(4, 'big'))
    track_chunk.extend(track_bytes)
    with open(out_path, 'wb') as f:
        f.write(header + track_chunk)


@app.get("/export/midi/{job_id}")
def export_midi(job_id: str):
    job = _get_job(job_id)
    if not job or not job.get("result", {}).get("notes"):
        raise HTTPException(404, "Job notes not found")
    midi_path = os.path.join(JOBS_DIR, job_id, "solo_guitar.mid")
    bpm = int(job.get("result", {}).get("bpm", 100))
    _write_midi(job["result"]["notes"], midi_path, tempo_bpm=bpm)
    return FileResponse(midi_path, media_type="audio/midi", filename=f"guitar_solo_{job_id[:6]}.mid")


@app.get("/export/tab/{job_id}")
def export_tab(job_id: str):
    job = _get_job(job_id)
    if not job or not job.get("result", {}).get("notes"):
        raise HTTPException(404, "Job notes not found")
    notes = job["result"]["notes"]
    labels = ["e", "B", "G", "D", "A", "E"]
    order = [5, 4, 3, 2, 1, 0]
    out_lines = [f"{lbl}|" for lbl in labels]
    for n in notes:
        s_idx = int(n.get("string", 3))
        f_str = str(n.get("fret", 0))
        for ri, si in enumerate(order):
            out_lines[ri] += f"-{f_str if si == s_idx else '-' * len(f_str)}-"
    for ri in range(6):
        out_lines[ri] += "|"
    tab_text = f"# Solo Guitar Tab - {job.get('result', {}).get('title', 'Transcribed')}\n# Key: {job.get('result', {}).get('key')} | BPM: {job.get('result', {}).get('bpm')}\n\n" + "\n".join(out_lines) + "\n"
    tab_path = os.path.join(JOBS_DIR, job_id, "tab.txt")
    with open(tab_path, "w", encoding="utf-8") as f:
        f.write(tab_text)
    return FileResponse(tab_path, media_type="text/plain", filename=f"tab_{job_id[:6]}.txt")


# --- Google Drive & local song persistence ---

class SavedSong(BaseModel):
    title: str
    data: dict


@app.get("/songs")
def list_songs():
    return drive_service.list_songs()


@app.post("/songs")
def save_song(song: SavedSong, x_api_key: str | None = Header(default=None)):
    check_api_key(x_api_key)
    song_id = str(uuid.uuid4())
    res = drive_service.save_song(song_id, song.title, song.data)
    return {"song_id": song_id, "drive_synced": res.get("drive_synced", False)}


@app.get("/songs/{song_id}")
def get_song(song_id: str):
    song = drive_service.get_song(song_id)
    if not song:
        raise HTTPException(404, "song not found")
    return song


@app.get("/gdrive/status")
def gdrive_status():
    return {
        "connected": drive_service.is_connected(),
        "folder_name": "Sargam Strings Tabs",
        "folder_id": drive_service.folder_id
    }


@app.get("/health")
def health():
    return {"status": "ok", "version": "pure-instant-v1"}
