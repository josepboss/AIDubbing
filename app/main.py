import os
import uuid
import json
import logging
import threading
import subprocess
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.config import load_settings, save_settings

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
        segments = detect_speakers(video_path, segments)

        update_job(job_id, current_step="translate", step_index=4,
                   message=f"Translating to {settings.get('target_language', 'Arabic')}...")
        if not settings.get("openrouter_api_key"):
            raise ValueError("OpenRouter API key is not set. Please configure it in Settings.")
        from app.translate import translate_segments
        segments = translate_segments(
            segments,
            settings["openrouter_api_key"],
            settings.get("openrouter_model", "google/gemini-2.0-flash-lite"),
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


app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
