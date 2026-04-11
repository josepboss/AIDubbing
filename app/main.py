import os
import uuid
import json
import logging
import threading
import subprocess
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.config import load_settings, save_settings
from app import downloader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent
UPLOADS_DIR = BASE_DIR / "app" / "uploads"
TRANSCRIPTS_DIR = BASE_DIR / "app" / "transcripts"
AUDIO_DIR = BASE_DIR / "app" / "audio"
OUTPUT_DIR = BASE_DIR / "app" / "output"
JOBS_DIR = BASE_DIR / "app" / "jobs"
STATIC_DIR = BASE_DIR / "static"

for d in [UPLOADS_DIR, TRANSCRIPTS_DIR, AUDIO_DIR, OUTPUT_DIR, JOBS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="AIDubbing", version="1.0.0", root_path="/dubbing")

PIPELINE_STEPS = [
    "upload",
    "extract_audio",
    "transcribe",
    "detect_speakers",
    "translate",
    "generate_tts",
    "assemble_audio",
    "merge_video"
]


def job_path(job_id: str) -> Path:
    return JOBS_DIR / f"{job_id}.json"


def read_job(job_id: str) -> dict:
    p = job_path(job_id)
    if not p.exists():
        raise HTTPException(status_code=404, detail="Job not found")
    with open(p) as f:
        return json.load(f)


def write_job(job_id: str, data: dict):
    with open(job_path(job_id), "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def update_job(job_id: str, **kwargs):
    data = read_job(job_id)
    data.update(kwargs)
    write_job(job_id, data)


# --- Download job helpers (stored in same JOBS_DIR with dl_ prefix) ---

def save_dl_job(job_id: str, data: dict):
    path = JOBS_DIR / f"dl_{job_id}.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_dl_job(job_id: str) -> dict | None:
    path = JOBS_DIR / f"dl_{job_id}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def run_pipeline(job_id: str):
    try:
        job = read_job(job_id)
        video_path = job["video_path"]
        settings = load_settings()

        update_job(job_id, status="running", current_step="extract_audio",
                   step_index=1, message="Extracting audio from video...")
        audio_path = str(AUDIO_DIR / f"{job_id}.wav")
        subprocess.run([
            "ffmpeg", "-y", "-i", video_path,
            "-ac", "1", "-ar", "16000",
            audio_path
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        update_job(job_id, current_step="transcribe", step_index=2,
                   message="Transcribing speech with Whisper...")
        from app.transcribe import transcribe_video
        segments = transcribe_video(video_path, settings.get("whisper_model", "base"))

        transcript_path = str(TRANSCRIPTS_DIR / f"{job_id}.json")
        with open(transcript_path, "w") as f:
            json.dump(segments, f, indent=2, ensure_ascii=False)

        update_job(job_id, current_step="detect_speakers", step_index=3,
                   message="Detecting speakers and gender...")
        from app.speaker import detect_speakers
        hf_token = settings.get("hf_token", "")
        segments = detect_speakers(video_path, segments, hf_token=hf_token)

        update_job(job_id, current_step="translate", step_index=4,
                   message=f"Translating to {settings.get('target_language', 'Arabic')}...")
        if not settings.get("openrouter_api_key"):
            raise ValueError("OpenRouter API key is not set. Please configure it in Settings.")
        from app.translate import translate_segments
        segments = translate_segments(
            segments,
            settings["openrouter_api_key"],
            settings.get("openrouter_model", "google/gemini-2.0-flash-lite-001"),
            settings.get("target_language", "Arabic")
        )

        with open(transcript_path, "w") as f:
            json.dump(segments, f, indent=2, ensure_ascii=False)

        update_job(job_id, current_step="generate_tts", step_index=5,
                   message="Generating dubbed audio with AI voices...")
        from app.dub import create_dubbed_audio, get_video_duration
        dubbed_audio_path = str(AUDIO_DIR / f"{job_id}_dubbed.mp3")
        duration = get_video_duration(video_path)
        create_dubbed_audio(segments, settings, dubbed_audio_path, duration)

        update_job(job_id, current_step="assemble_audio", step_index=6,
                   message="Assembling dubbed audio track...")

        update_job(job_id, current_step="merge_video", step_index=7,
                   message="Merging dubbed audio with video...")
        from app.dub import merge_video_with_dubbed_audio
        output_filename = f"{job_id}_dubbed.mp4"
        output_path = str(OUTPUT_DIR / output_filename)
        merge_video_with_dubbed_audio(video_path, dubbed_audio_path, output_path)

        update_job(job_id,
                   status="completed",
                   current_step="done",
                   step_index=8,
                   message="Dubbing complete!",
                   output_filename=output_filename)

    except Exception as e:
        logger.exception(f"Pipeline failed for job {job_id}")
        try:
            update_job(job_id, status="error", message=str(e))
        except Exception:
            pass


@app.get("/api/settings")
def get_settings():
    return load_settings()


@app.post("/api/settings")
async def post_settings(body: dict):
    save_settings(body)
    return {"success": True}


@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    ext = Path(file.filename).suffix.lower()
    if ext not in [".mp4", ".mkv", ".avi", ".mov", ".webm"]:
        raise HTTPException(status_code=400, detail="Unsupported file format. Use MP4, MKV, AVI, MOV, or WebM.")

    job_id = str(uuid.uuid4())
    video_path = str(UPLOADS_DIR / f"{job_id}{ext}")

    content = await file.read()
    with open(video_path, "wb") as f:
        f.write(content)

    job_data = {
        "job_id": job_id,
        "status": "uploaded",
        "current_step": "upload",
        "step_index": 0,
        "message": "Video uploaded. Ready to process.",
        "original_filename": file.filename,
        "video_path": video_path,
        "output_filename": None,
        "steps": PIPELINE_STEPS
    }
    write_job(job_id, job_data)

    logger.info(f"Uploaded job {job_id}: {file.filename}")
    return {"job_id": job_id, "filename": file.filename}


@app.post("/api/process/{job_id}")
def process_job(job_id: str):
    job = read_job(job_id)
    if job["status"] == "running":
        raise HTTPException(status_code=409, detail="Job is already running")

    thread = threading.Thread(target=run_pipeline, args=(job_id,), daemon=True)
    thread.start()

    return {"job_id": job_id, "status": "started"}


@app.get("/api/status/{job_id}")
def get_status(job_id: str):
    return read_job(job_id)


@app.get("/api/download/{job_id}")
def download_video(job_id: str):
    job = read_job(job_id)
    if job["status"] != "completed" or not job.get("output_filename"):
        raise HTTPException(status_code=404, detail="Output not ready yet")

    output_path = OUTPUT_DIR / job["output_filename"]
    if not output_path.exists():
        raise HTTPException(status_code=404, detail="Output file not found")

    original = Path(job.get("original_filename", "video.mp4")).stem
    return FileResponse(
        str(output_path),
        media_type="video/mp4",
        filename=f"{original}_arabic_dubbed.mp4"
    )


@app.post("/api/download/info")
async def get_download_info(body: dict):
    url = body.get("url", "").strip()
    if not url:
        return {"success": False, "error": "URL is required"}
    return downloader.get_video_info(url)


@app.post("/api/download/start")
async def start_download(body: dict, background_tasks: BackgroundTasks):
    url = body.get("url", "").strip()
    quality = body.get("quality", "720p")
    if not url:
        return {"success": False, "error": "URL is required"}

    job_id = str(uuid.uuid4())[:8]
    job = {
        "job_id": job_id,
        "status": "downloading",
        "progress": 0,
        "current_step": "Starting download...",
        "source_url": url,
        "video_path": None,
        "error_message": None
    }
    save_dl_job(job_id, job)
    background_tasks.add_task(_run_download, job_id, url, quality)
    return {"success": True, "job_id": job_id}


def _run_download(job_id: str, url: str, quality: str):
    def progress_cb(pct):
        job = load_dl_job(job_id)
        if job:
            job["progress"] = int(pct)
            job["current_step"] = f"Downloading... {pct:.1f}%"
            save_dl_job(job_id, job)

    result = downloader.download_video(url, job_id, quality, progress_cb)
    job = load_dl_job(job_id) or {}

    if result["success"]:
        job["status"] = "ready"
        job["progress"] = 100
        job["current_step"] = f"Downloaded ({result['file_size_mb']} MB) — Ready to dub"
        job["video_path"] = result["video_path"]
    else:
        job["status"] = "failed"
        job["error_message"] = result["error"]
        job["current_step"] = "Download failed"

    save_dl_job(job_id, job)


@app.get("/api/download/status/{job_id}")
async def download_job_status(job_id: str):
    job = load_dl_job(job_id)
    if not job:
        return {"success": False, "error": "Job not found"}
    return job


@app.post("/api/dub-from-download/{download_job_id}")
async def dub_from_download(download_job_id: str):
    """Create a dubbing job from an already-downloaded video"""
    dl_job = load_dl_job(download_job_id)
    if not dl_job:
        raise HTTPException(status_code=404, detail="Download job not found")
    if dl_job.get("status") != "ready":
        raise HTTPException(status_code=400, detail="Download not complete yet")

    video_path = dl_job.get("video_path")
    if not video_path or not Path(video_path).exists():
        raise HTTPException(status_code=404, detail="Downloaded video file not found")

    dub_job_id = str(uuid.uuid4())
    job_data = {
        "job_id": dub_job_id,
        "status": "uploaded",
        "current_step": "upload",
        "step_index": 0,
        "message": "Video loaded from downloader. Ready to process.",
        "original_filename": Path(video_path).name,
        "video_path": video_path,
        "output_filename": None,
        "steps": PIPELINE_STEPS
    }
    write_job(dub_job_id, job_data)
    logger.info(f"Created dub job {dub_job_id} from download {download_job_id}")
    return {"job_id": dub_job_id}


app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
