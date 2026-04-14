import os
import uuid
import json
import logging
import threading
import subprocess
import time
import math
from datetime import datetime, timezone
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Request
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

# Install Arabic/Unicode fonts for subtitle rendering (best-effort, silent)
subprocess.run(
    ["apt-get", "install", "-y", "fonts-arabic-extra", "fonts-noto-color-emoji"],
    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
)

app = FastAPI(title="AIDubbing", version="1.0.0", root_path="/dubbing")

PIPELINE_STEPS = [
    "upload",
    "extract_audio",
    "transcribe",
    "detect_speakers",
    "translate",
    "generate_tts",
    "assemble_audio",
    "merge_video",
    "metadata"
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


RESUMABLE_STEPS = [
    "extract_audio",   # 0
    "transcribe",      # 1
    "detect_speakers", # 2
    "translate",       # 3
    "generate_tts",    # 4
    "assemble_audio",  # 5
    "merge_video",     # 6
    "metadata",        # 7
]


def _ckpt(job_id: str, name: str) -> str:
    """Checkpoint file path for a given step name."""
    return str(TRANSCRIPTS_DIR / f"{job_id}_{name}.json")


def _load_ckpt(job_id: str, name: str):
    p = _ckpt(job_id, name)
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return None


def _save_ckpt(job_id: str, name: str, data):
    with open(_ckpt(job_id, name), "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def run_pipeline(job_id: str, resume_from: str = None):
    try:
        job = read_job(job_id)
        video_path = job["video_path"]
        settings = load_settings()

        audio_path = str(AUDIO_DIR / f"{job_id}.wav")
        dubbed_audio_path = str(AUDIO_DIR / f"{job_id}_dubbed.mp3")

        resume_idx = RESUMABLE_STEPS.index(resume_from) if resume_from in RESUMABLE_STEPS else 0

        # Step timing tracker — records actual elapsed seconds per step
        step_timings = job.get("step_timings", {})

        def start_step(name):
            step_timings[name] = {"start": time.time(), "done": False}
            update_job(job_id, step_timings=step_timings)

        def finish_step(name):
            if name in step_timings:
                step_timings[name]["elapsed"] = round(time.time() - step_timings[name]["start"])
                step_timings[name]["done"] = True
            update_job(job_id, step_timings=step_timings)

        # Load segments from the most recent checkpoint when resuming
        segments = None
        if resume_idx >= 4:
            segments = _load_ckpt(job_id, "translated") or _load_ckpt(job_id, "speakers")
        elif resume_idx >= 3:
            segments = _load_ckpt(job_id, "speakers") or _load_ckpt(job_id, "transcribed")
        elif resume_idx >= 2:
            segments = _load_ckpt(job_id, "transcribed")

        update_job(job_id, status="running")

        # ── Step 1: Extract audio ──────────────────────────────────────────────
        if resume_idx <= 0:
            start_step("extract_audio")
            update_job(job_id, current_step="extract_audio", step_index=1,
                       message="Extracting audio from video...")
            subprocess.run([
                "ffmpeg", "-y", "-i", video_path,
                "-ac", "1", "-ar", "16000", audio_path
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            finish_step("extract_audio")

        # ── Step 2: Transcribe ─────────────────────────────────────────────────
        if resume_idx <= 1:
            start_step("transcribe")
            update_job(job_id, current_step="transcribe", step_index=2,
                       message="Transcribing speech with Whisper...")
            from app.transcribe import transcribe_video
            segments = transcribe_video(video_path, settings.get("whisper_model", "base"))
            _save_ckpt(job_id, "transcribed", segments)
            finish_step("transcribe")

        # ── Step 3: Speaker detection ──────────────────────────────────────────
        if resume_idx <= 2:
            start_step("detect_speakers")
            update_job(job_id, current_step="detect_speakers", step_index=3,
                       message="Detecting speakers and gender...")
            from app.speaker import detect_speakers
            segments, speaker_method = detect_speakers(
                video_path, segments, hf_token=settings.get("hf_token", "")
            )
            _save_ckpt(job_id, "speakers", segments)
            update_job(job_id, speaker_method=speaker_method,
                       message=f"Speakers detected via {speaker_method}")
            finish_step("detect_speakers")

        # ── Step 4: Translate ──────────────────────────────────────────────────
        if resume_idx <= 3:
            start_step("translate")
            update_job(job_id, current_step="translate", step_index=4,
                       message=f"Translating to {settings.get('target_language', 'Arabic')}...")
            if not settings.get("openrouter_api_key"):
                raise ValueError("OpenRouter API key is not set. Please configure it in Settings.")
            from app.translate import translate_segments
            segments = translate_segments(
                segments,
                settings["openrouter_api_key"],
                settings.get("translation_model", "qwen/qwen-2.5-72b-instruct"),
                settings.get("target_language", "Arabic")
            )
            _save_ckpt(job_id, "translated", segments)

            # Save SRT for the translated segments
            from app.dub import generate_srt
            srt_dir = OUTPUT_DIR / job_id
            srt_dir.mkdir(parents=True, exist_ok=True)
            srt_path = str(srt_dir / "subtitles.srt")
            with open(srt_path, "w", encoding="utf-8") as f:
                f.write(generate_srt(segments))
            update_job(job_id, srt_path=srt_path)
            logger.info(f"SRT saved: {srt_path}")
            finish_step("translate")

        # ── Step 5: Generate TTS ───────────────────────────────────────────────
        if resume_idx <= 4:
            start_step("generate_tts")
            update_job(job_id, current_step="generate_tts", step_index=5,
                       message="Generating dubbed audio with AI voices...")
            # Merge per-job voice overrides into settings (highest priority)
            for key in ("narrator_voice", "narrator_gender", "male_voice", "female_voice"):
                if job.get(key):
                    settings[key] = job[key]
            from app.dub import create_dubbed_audio, get_video_duration
            duration = get_video_duration(video_path)
            create_dubbed_audio(segments, settings, dubbed_audio_path, duration)
            finish_step("generate_tts")

        # ── Step 6: Assemble audio ─────────────────────────────────────────────
        if resume_idx <= 5:
            start_step("assemble_audio")
            update_job(job_id, current_step="assemble_audio", step_index=6,
                       message="Assembling dubbed audio track...")
            finish_step("assemble_audio")

        # ── Step 7: Merge video ────────────────────────────────────────────────
        start_step("merge_video")
        cpu_count = os.cpu_count() or 1
        update_job(job_id, current_step="merge_video", step_index=7,
                   message=f"Starting demucs vocal separation on {cpu_count} CPUs...",
                   cpu_count=cpu_count)
        from app.dub import merge_video_with_dubbed_audio

        # Progress callback — updates job JSON on every chunk so the UI can show live progress
        def demucs_progress(chunk_num, total_chunks, remaining, avg_sec, eta_min, cpus):
            update_job(job_id,
                       message=f"Demucs vocal separation: chunk {chunk_num}/{total_chunks} "
                               f"({remaining} remaining, ~{eta_min:.0f} min ETA)",
                       demucs_progress={
                           "chunk": chunk_num,
                           "total": total_chunks,
                           "remaining": remaining,
                           "eta_min": round(eta_min, 1),
                           "avg_sec_per_chunk": round(avg_sec, 1),
                           "cpu_count": cpus,
                       })

        output_filename = f"{job_id}_dubbed.mp4"
        output_path = str(OUTPUT_DIR / output_filename)
        current_srt_path = read_job(job_id).get("srt_path")
        logger.info(f"Merge: srt_path={current_srt_path}")
        merge_video_with_dubbed_audio(video_path, dubbed_audio_path, output_path,
                                      job_id=job_id, settings=settings,
                                      srt_path=current_srt_path,
                                      progress_fn=demucs_progress)
        finish_step("merge_video")

        # Warn in job status if demucs failed (no_vocals.wav missing or tiny)
        no_vocals_wav = AUDIO_DIR / job_id / "no_vocals.wav"
        use_demucs = settings.get("vocal_removal", "demucs") == "demucs"
        if use_demucs:
            if not no_vocals_wav.exists() or no_vocals_wav.stat().st_size < 1_000_000:
                logger.warning("Demucs produced no output — video merged with original audio at 15% volume")
                update_job(job_id,
                           message="⚠️ Demucs vocal removal failed (see server logs for reason). "
                                   "Video merged using original audio at 15% volume instead. "
                                   "Continuing to subtitle burn...")

        # ── Step 8: Translate title + embed metadata + generate thumbnail ────────
        update_job(job_id, current_step="metadata", step_index=8,
                   message="Generating title translation and thumbnail...")
        from app import metadata as meta_mod
        # Prefer the stored video_title (from YouTube download info) over
        # the filename stem, which is just "video" when using the fixed downloader
        original_title = (
            job.get("video_title")
            or Path(job.get("original_filename", "video.mp4")).stem
        )
        translated_title = meta_mod.translate_title(
            original_title,
            settings.get("target_language", "Arabic"),
            settings.get("openrouter_api_key", ""),
            settings.get("openrouter_model", "google/gemini-2.0-flash-lite-001")
        )

        meta_output = str(OUTPUT_DIR / f"{job_id}_final.mp4")
        meta_mod.embed_metadata(
            output_path,
            meta_output,
            title=translated_title,
            description=f"Dubbed in {settings.get('target_language','Arabic')} by AIDubbing",
            language="ara" if settings.get("target_language", "Arabic") == "Arabic" else "und"
        )
        if os.path.exists(meta_output) and os.path.getsize(meta_output) > 0:
            output_filename = f"{job_id}_final.mp4"
        else:
            meta_output = output_path

        thumb_style = settings.get("thumbnail_style", "title_overlay")
        thumbnail_path = meta_mod.generate_thumbnail(
            meta_output, str(OUTPUT_DIR / job_id), translated_title, style=thumb_style
        )

        update_job(job_id, status="completed", current_step="done", step_index=9,
                   message="Dubbing complete!", output_filename=output_filename,
                   translated_title=translated_title,
                   thumbnail_path=thumbnail_path)

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
        "steps": PIPELINE_STEPS,
        "created_at": datetime.now(timezone.utc).isoformat()
    }
    write_job(job_id, job_data)

    logger.info(f"Uploaded job {job_id}: {file.filename}")
    return {"job_id": job_id, "filename": file.filename}


@app.post("/api/process/{job_id}")
async def process_job(job_id: str, request: Request):
    job = read_job(job_id)
    if job["status"] == "running":
        raise HTTPException(status_code=409, detail="Job is already running")

    # Accept optional voice settings in request body
    try:
        body = await request.json()
    except Exception:
        body = {}

    voice_keys = ("narrator_voice", "narrator_gender", "male_voice", "female_voice")
    updates = {k: body[k] for k in voice_keys if k in body}
    if updates:
        update_job(job_id, **updates)
        logger.info(f"Job {job_id} voice settings: {updates}")

    thread = threading.Thread(target=run_pipeline, args=(job_id,), daemon=True)
    thread.start()
    return {"job_id": job_id, "status": "started"}


@app.post("/api/resume/{job_id}")
def resume_job(job_id: str):
    """Resume a failed job from the step it failed at, reusing all saved checkpoints."""
    job = read_job(job_id)
    if job["status"] == "running":
        raise HTTPException(status_code=409, detail="Job is already running")
    if job["status"] == "completed":
        raise HTTPException(status_code=400, detail="Job already completed")

    failed_step = job.get("current_step", "extract_audio")
    # If the step is not resumable (e.g. "upload", "done") start fresh
    if failed_step not in RESUMABLE_STEPS:
        failed_step = "extract_audio"

    logger.info(f"Resuming job {job_id} from step: {failed_step}")
    thread = threading.Thread(target=run_pipeline, args=(job_id, failed_step), daemon=True)
    thread.start()
    return {"job_id": job_id, "status": "resumed", "from_step": failed_step}


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
        # Carry the real video title from the download job so the metadata step
        # can translate something meaningful instead of just "video"
        "video_title": dl_job.get("title", ""),
        "video_path": video_path,
        "output_filename": None,
        "steps": PIPELINE_STEPS
    }
    write_job(dub_job_id, job_data)
    logger.info(f"Created dub job {dub_job_id} from download {download_job_id}")
    return {"job_id": dub_job_id}


def _get_video_duration(video_path: str) -> float:
    """Return video duration in seconds via ffprobe."""
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", video_path],
        capture_output=True, text=True, timeout=30,
    )
    info = json.loads(result.stdout)
    return float(info["format"]["duration"])


def _calc_clip_count(duration_sec: float) -> int:
    """Return the optimal number of clips so each is between 10–15 min."""
    duration_min = duration_sec / 60
    if duration_min <= 15:
        return 1
    n = math.ceil(duration_min / 15)   # min clips to keep each ≤ 15 min
    clip_min = duration_min / n
    # If clips are suspiciously short, reduce n (shouldn't happen but be safe)
    while n > 1 and clip_min < 10:
        n -= 1
        clip_min = duration_min / n
    return n


def _run_split(dl_job_id: str, video_path: str, n_clips: int):
    dl_job = load_dl_job(dl_job_id) or {}
    dl_job["split_status"] = "splitting"
    dl_job["split_clips"] = []
    save_dl_job(dl_job_id, dl_job)

    clips_dir = Path(video_path).parent / "clips"
    clips_dir.mkdir(exist_ok=True)

    try:
        duration_sec = _get_video_duration(video_path)
        segment_sec = math.ceil(duration_sec / n_clips)

        clip_pattern = str(clips_dir / "clip_%03d.mp4")
        cmd = [
            "ffmpeg", "-y", "-i", video_path,
            "-c", "copy", "-map", "0",
            "-segment_time", str(segment_sec),
            "-f", "segment",
            "-reset_timestamps", "1",
            clip_pattern,
        ]
        logger.info(f"Split: {n_clips} clips × ~{segment_sec}s  →  {clips_dir}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)

        clip_files = sorted(clips_dir.glob("clip_*.mp4"))
        dl_job = load_dl_job(dl_job_id) or {}
        if result.returncode == 0 and clip_files:
            dl_job["split_status"] = "done"
            dl_job["split_clips"] = [
                {
                    "index": i,
                    "filename": f.name,
                    "size_mb": round(f.stat().st_size / 1024 / 1024, 1),
                }
                for i, f in enumerate(clip_files)
            ]
            logger.info(f"Split done: {len(clip_files)} clips")
        else:
            dl_job["split_status"] = "failed"
            dl_job["split_error"] = result.stderr[-400:] if result.stderr else "Unknown error"
            logger.error(f"Split failed: {dl_job['split_error']}")
    except Exception as e:
        dl_job = load_dl_job(dl_job_id) or {}
        dl_job["split_status"] = "failed"
        dl_job["split_error"] = str(e)
        logger.error(f"Split exception: {e}")

    save_dl_job(dl_job_id, dl_job)


@app.get("/api/download/split-info/{dl_job_id}")
async def split_info(dl_job_id: str):
    """Return suggested clip count and per-clip duration."""
    dl_job = load_dl_job(dl_job_id)
    if not dl_job:
        raise HTTPException(status_code=404, detail="Download job not found")
    video_path = dl_job.get("video_path")
    if not video_path or not Path(video_path).exists():
        raise HTTPException(status_code=404, detail="Video file not found")
    try:
        duration_sec = _get_video_duration(video_path)
        n = _calc_clip_count(duration_sec)
        return {
            "duration_sec": round(duration_sec),
            "duration_min": round(duration_sec / 60, 1),
            "suggested_clips": n,
            "clip_duration_min": round(duration_sec / 60 / n, 1),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/download/split/{dl_job_id}")
async def start_split(dl_job_id: str, body: dict, background_tasks: BackgroundTasks):
    dl_job = load_dl_job(dl_job_id)
    if not dl_job:
        raise HTTPException(status_code=404, detail="Download job not found")
    video_path = dl_job.get("video_path")
    if not video_path or not Path(video_path).exists():
        raise HTTPException(status_code=404, detail="Video file not found")
    n_clips = int(body.get("n_clips", 1))
    if n_clips < 1:
        n_clips = 1
    background_tasks.add_task(_run_split, dl_job_id, video_path, n_clips)
    return {"success": True}


@app.get("/api/download/split-status/{dl_job_id}")
async def get_split_status(dl_job_id: str):
    dl_job = load_dl_job(dl_job_id)
    if not dl_job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {
        "split_status": dl_job.get("split_status"),
        "split_clips": dl_job.get("split_clips", []),
        "split_error": dl_job.get("split_error"),
    }


@app.get("/api/download/split-file/{dl_job_id}/{clip_index}")
async def download_split_file(dl_job_id: str, clip_index: int):
    dl_job = load_dl_job(dl_job_id)
    if not dl_job:
        raise HTTPException(status_code=404, detail="Job not found")
    clips = dl_job.get("split_clips", [])
    if clip_index < 0 or clip_index >= len(clips):
        raise HTTPException(status_code=404, detail="Clip index out of range")
    video_path = dl_job.get("video_path")
    clip_file = Path(video_path).parent / "clips" / clips[clip_index]["filename"]
    if not clip_file.exists():
        raise HTTPException(status_code=404, detail="Clip file not found on disk")
    return FileResponse(str(clip_file), media_type="video/mp4", filename=clips[clip_index]["filename"])


@app.get("/api/download/file/{dl_job_id}")
async def download_source_file(dl_job_id: str):
    """Download the raw video file that was fetched by yt-dlp."""
    dl_job = load_dl_job(dl_job_id)
    if not dl_job:
        raise HTTPException(status_code=404, detail="Download job not found")
    video_path = dl_job.get("video_path")
    if not video_path or not Path(video_path).exists():
        raise HTTPException(status_code=404, detail="Video file not found on server")
    filename = Path(video_path).name
    return FileResponse(video_path, media_type="video/mp4", filename=filename)


@app.get("/api/download-srt/{job_id}")
async def download_srt(job_id: str):
    """Download the generated SRT subtitle file for a job."""
    job = read_job(job_id)
    srt_path = job.get("srt_path", str(OUTPUT_DIR / job_id / "subtitles.srt"))
    if not Path(srt_path).exists():
        raise HTTPException(status_code=404, detail="SRT file not found. Run dubbing first.")
    return FileResponse(
        srt_path,
        media_type="text/plain; charset=utf-8",
        filename=f"subtitles_{job_id[:8]}.srt"
    )


@app.get("/api/thumbnail/{job_id}")
def get_thumbnail(job_id: str):
    """Return the generated thumbnail image for a completed job."""
    job = read_job(job_id)
    thumb = job.get("thumbnail_path", str(OUTPUT_DIR / job_id / "thumbnail.jpg"))
    if not Path(thumb).exists():
        raise HTTPException(status_code=404, detail="Thumbnail not found")
    return FileResponse(str(thumb), media_type="image/jpeg",
                        filename=f"thumbnail_{job_id[:8]}.jpg")


@app.get("/api/history")
def get_history():
    """Return all dubbing jobs sorted by most recent first."""
    jobs = []
    for p in JOBS_DIR.glob("*.json"):
        if p.name.startswith("dl_"):
            continue
        try:
            with open(p) as f:
                job = json.load(f)
            # Use stored created_at, or fall back to file mtime
            if "created_at" not in job:
                mtime = p.stat().st_mtime
                job["created_at"] = datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()
            jobs.append(job)
        except Exception:
            continue
    jobs.sort(key=lambda j: j.get("created_at", ""), reverse=True)
    return jobs[:50]


app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
