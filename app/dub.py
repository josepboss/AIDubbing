import subprocess
import os
import logging
from pydub import AudioSegment
import io

logger = logging.getLogger(__name__)


def create_dubbed_audio(segments: list, settings: dict, output_path: str,
                        original_duration: float) -> str:
    """Create full dubbed audio track from translated segments"""
    from app.tts import generate_segment_audio

    dubbed = AudioSegment.silent(duration=int(original_duration * 1000))

    # Pre-compile label pattern for TTS safety guard
    import re as _re
    _LABEL_ONLY = _re.compile(r'^\[?(?:SPEAKER_\d+|NARRATOR)\]?\s*:?\s*$', _re.IGNORECASE)

    for i, seg in enumerate(segments):
        text = seg.get("translated", "").strip()
        if not text or _LABEL_ONLY.match(text):
            # Skip segments with no translation or bare speaker labels
            if text:
                logger.warning(f"Segment {i}: skipping label-only translation: {text!r}")
            continue

        try:
            audio_bytes = generate_segment_audio(
                seg["translated"],
                seg.get("gender", "male"),
                settings,
                is_narrator=seg.get("is_narrator", False)
            )

            segment_audio = AudioSegment.from_mp3(io.BytesIO(audio_bytes))

            target_ms = int((seg["end"] - seg["start"]) * 1000)
            actual_ms = len(segment_audio)

            if actual_ms > target_ms * 1.2:
                speed_factor = actual_ms / target_ms
                speed_factor = min(speed_factor, 1.5)
                segment_audio = _change_speed(segment_audio, speed_factor)

            position_ms = int(seg["start"] * 1000)
            dubbed = dubbed.overlay(segment_audio, position=position_ms)

            logger.info(f"Segment {i + 1}/{len(segments)} dubbed")

        except Exception as e:
            logger.error(f"Failed to dub segment {i}: {e}")
            continue

    dubbed.export(output_path, format="mp3")
    return output_path


def _change_speed(audio: AudioSegment, speed: float) -> AudioSegment:
    """Speed up audio using ffmpeg"""
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        tmp_in = f.name
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        tmp_out = f.name

    audio.export(tmp_in, format="mp3")
    subprocess.run([
        "ffmpeg", "-y", "-i", tmp_in,
        "-filter:a", f"atempo={speed}",
        tmp_out
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    result = AudioSegment.from_mp3(tmp_out)
    os.unlink(tmp_in)
    os.unlink(tmp_out)
    return result


def separate_background_audio(video_path: str, job_id: str,
                              progress_fn=None) -> str | None:
    """
    Extract background audio (music + effects) without vocals using demucs.
    Processes audio in 60-second chunks to avoid OOM on long videos — a 2-hour
    video at 44.1kHz stereo float32 is ~2.5 GB if loaded all at once.
    Returns path to no_vocals.wav, or None if separation fails.
    """
    try:
        import torch
        import numpy as np
        import soundfile as sf
        from demucs.pretrained import get_model
        from demucs.apply import apply_model

        output_dir = f"app/audio/{job_id}"
        os.makedirs(output_dir, exist_ok=True)

        # ── 0. Reuse if already completed successfully (valid file > 1 MB) ──
        no_vocals_path = os.path.join(output_dir, "no_vocals.wav")
        if os.path.exists(no_vocals_path) and os.path.getsize(no_vocals_path) > 1_000_000:
            logger.info(f"Demucs: reusing existing no_vocals.wav ({os.path.getsize(no_vocals_path)//1024//1024}MB)")
            return no_vocals_path

        chunks_dir = os.path.join(output_dir, "demucs_chunks")
        os.makedirs(chunks_dir, exist_ok=True)

        # ── 1. Extract stereo 44.1 kHz WAV with ffmpeg ──────────────────────
        audio_path = os.path.join(output_dir, "original.wav")
        if os.path.exists(audio_path) and os.path.getsize(audio_path) > 1_000_000:
            logger.info(f"Demucs: reusing existing original.wav ({os.path.getsize(audio_path)//1024//1024}MB)")
        else:
            ret = subprocess.run([
                "ffmpeg", "-y", "-i", video_path,
                "-ac", "2", "-ar", "44100", "-vn", audio_path
            ], capture_output=True, text=True)

            if ret.returncode != 0 or not os.path.exists(audio_path):
                logger.error(
                    f"DEMUCS FAILED — step: audio extraction\n"
                    f"  Reason: ffmpeg returned code {ret.returncode}\n"
                    f"  ffmpeg stderr: {ret.stderr.strip()[-600:]}"
                )
                return None
            logger.info(f"Audio extracted for demucs: {os.path.getsize(audio_path)//1024//1024}MB at {audio_path}")

        # ── 2. Load model (cache in workspace so it survives restarts) ──────
        #      ~/.cache/torch is wiped on every container restart; the workspace
        #      directory persists, so we redirect TORCH_HOME there.
        workspace_cache = os.path.abspath("app/model_cache")
        os.makedirs(workspace_cache, exist_ok=True)
        os.environ["TORCH_HOME"] = workspace_cache
        import torch as _torch_hub_setup
        _torch_hub_setup.hub.set_dir(workspace_cache)

        existing_cache = [f for f in os.listdir(workspace_cache) if f.endswith(".th") or f.endswith(".yaml")]
        if existing_cache:
            logger.info(f"Demucs: loading model from workspace cache ({existing_cache})")
        else:
            logger.info("Demucs: no cached model found — downloading htdemucs (~300MB, one-time only)...")
        model = get_model("htdemucs")
        model.eval()
        stem_names = model.sources          # ['drums', 'bass', 'other', 'vocals']
        vocal_idx = stem_names.index("vocals")
        logger.info(f"Demucs model loaded. Stems: {stem_names}, vocal_idx={vocal_idx}")

        # ── 3. Process audio in 60-second chunks, saving each to disk ────────
        #      Chunk files let us resume after a container restart without
        #      re-processing already-completed chunks.
        info = sf.info(audio_path)
        sr = info.samplerate
        total_dur = info.frames / sr
        chunk_samples = sr * 60             # 60 seconds per chunk
        total_chunks = int(total_dur / 60) + 1
        logger.info(
            f"Demucs: {total_dur/60:.1f} min audio → ~{total_chunks} chunks of 60s. "
            f"Chunk files saved to {chunks_dir} (resumable on restart)"
        )

        import time as _time
        cpu_count = os.cpu_count() or 1
        chunk_times = []   # seconds taken per new chunk
        skipped = 0

        with sf.SoundFile(audio_path, "r") as f:
            chunk_num = 0
            while True:
                data = f.read(chunk_samples)
                if len(data) == 0:
                    break
                chunk_num += 1
                chunk_file = os.path.join(chunks_dir, f"chunk_{chunk_num:04d}.wav")

                # Skip already-processed chunks (resume support)
                if os.path.exists(chunk_file) and os.path.getsize(chunk_file) > 1000:
                    skipped += 1
                    logger.info(f"Demucs: chunk {chunk_num}/{total_chunks} already done — skipping")
                    continue

                t0 = _time.time()
                if data.ndim == 1:
                    data = np.stack([data, data], axis=1)

                waveform = torch.tensor(data.T, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    sources = apply_model(model, waveform, device="cpu", progress=False)

                no_vocals_stereo = sum(
                    sources[0, i] for i in range(len(stem_names)) if i != vocal_idx
                )

                sf.write(chunk_file, no_vocals_stereo.numpy().T, sr)
                del waveform, sources, no_vocals_stereo

                elapsed = _time.time() - t0
                chunk_times.append(elapsed)
                avg_sec = sum(chunk_times) / len(chunk_times)
                remaining = total_chunks - chunk_num
                eta_min = (remaining * avg_sec) / 60

                logger.info(
                    f"Demucs: chunk {chunk_num}/{total_chunks} saved "
                    f"({elapsed:.0f}s, ETA ~{eta_min:.0f} min, {cpu_count} CPUs)"
                )

                if progress_fn:
                    progress_fn(chunk_num, total_chunks, remaining, avg_sec, eta_min, cpu_count)

        # ── 4. Stitch all chunk files into no_vocals.wav ──────────────────────
        chunk_files = sorted(f for f in os.listdir(chunks_dir) if f.startswith("chunk_") and f.endswith(".wav"))
        if not chunk_files:
            logger.error(
                "DEMUCS FAILED — step: chunk processing\n"
                "  Reason: no chunk files found — audio may be empty or unreadable"
            )
            return None

        logger.info(f"Demucs: stitching {len(chunk_files)} chunk files into {no_vocals_path}...")
        with sf.SoundFile(no_vocals_path, "w", samplerate=sr, channels=2, subtype="PCM_16") as out_f:
            for cf in chunk_files:
                chunk_data, _ = sf.read(os.path.join(chunks_dir, cf))
                out_f.write(chunk_data)

        result_size_mb = os.path.getsize(no_vocals_path) / 1_000_000
        logger.info(f"Demucs complete: {no_vocals_path} ({result_size_mb:.0f}MB) from {len(chunk_files)} chunks")
        return no_vocals_path

    except Exception as e:
        import traceback
        logger.error(
            f"DEMUCS FAILED — unhandled exception\n"
            f"  Error type: {type(e).__name__}\n"
            f"  Error: {e}\n"
            f"  Traceback:\n{traceback.format_exc()}"
        )
        return None


def generate_srt(segments: list) -> str:
    """Generate SRT subtitle content from translated segments."""
    srt_lines = []
    counter = 1
    for seg in segments:
        text = seg.get("translated", "").strip()
        if not text:
            continue
        start = _format_srt_time(seg["start"])
        end = _format_srt_time(seg["end"])
        srt_lines.append(f"{counter}\n{start} --> {end}\n{text}\n")
        counter += 1
    return "\n".join(srt_lines)


def _format_srt_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def merge_video_with_dubbed_audio(video_path: str, dubbed_audio_path: str,
                                   output_path: str, job_id: str = "",
                                   settings: dict = None,
                                   srt_path: str = None,
                                   progress_fn=None) -> str:
    """
    Merge video with dubbed audio and optionally burn subtitles.
    Split into two checkpointed phases so a restart can skip the first phase:
      Phase 1 — mix video + dubbed audio (+ optional background music) → _merged_audio.mp4
      Phase 2 — burn subtitles onto _merged_audio.mp4 → final output_path

    If vocal_removal is enabled (default), uses demucs to strip original
    vocals and keep only background music/effects at 40%.
    Falls back to original audio at 15% if demucs fails or is disabled.
    """
    if settings is None:
        settings = {}

    # Intermediate checkpoint: video+audio merged, no subtitles yet
    merged_audio_path = output_path.replace(".mp4", "_merged_audio.mp4")

    # ── PHASE 1: video + dubbed audio merge (skipped if checkpoint exists) ────
    if os.path.exists(merged_audio_path) and os.path.getsize(merged_audio_path) > 1_000_000:
        logger.info(f"Phase 1 checkpoint found ({os.path.getsize(merged_audio_path)//1_000_000}MB) — skipping audio merge")
    else:
        use_demucs = settings.get("vocal_removal", "demucs") == "demucs"
        background_path = None

        if use_demucs and job_id:
            background_path = separate_background_audio(video_path, job_id,
                                                        progress_fn=progress_fn)

        if background_path and os.path.exists(background_path):
            logger.info("Using separated background audio (vocals removed)")
            extra_audio_inputs = ["-i", background_path]
            audio_filter = (
                "[2:a]volume=0.4[bg];"
                "[1:a]volume=1.0[dub];"
                "[bg][dub]amix=inputs=2:duration=longest[aout]"
            )
        else:
            if use_demucs:
                logger.warning("Demucs failed — falling back to original audio at 15%")
            else:
                logger.info("Vocal removal disabled — using original audio at 15%")
            extra_audio_inputs = []
            audio_filter = (
                "[0:a]volume=0.15[original];"
                "[1:a]volume=1.0[dubbed];"
                "[original][dubbed]amix=inputs=2:duration=longest[aout]"
            )

        cmd_phase1 = (
            ["ffmpeg", "-y", "-i", video_path, "-i", dubbed_audio_path]
            + extra_audio_inputs
            + ["-filter_complex", audio_filter]
            + ["-map", "0:v:0", "-map", "[aout]"]
            + ["-c:v", "copy"]
            + ["-shortest", merged_audio_path]
        )
        logger.info("Phase 1: merging video + dubbed audio...")
        result1 = subprocess.run(cmd_phase1, check=False,
                                 stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        if result1.returncode != 0:
            err = result1.stderr.decode(errors="replace").strip()
            raise subprocess.CalledProcessError(result1.returncode, cmd_phase1,
                                                stderr=result1.stderr)
        logger.info(f"Phase 1 complete: {merged_audio_path}")

    # ── PHASE 2: burn subtitles (or just rename if no SRT) ────────────────────
    has_srt = srt_path and os.path.exists(srt_path)
    if has_srt:
        escaped = srt_path.replace("\\", "\\\\").replace("'", "\\'").replace(":", "\\:")
        subtitle_filter = (
            f"subtitles='{escaped}'"
            f":force_style='FontName=DejaVu Sans,FontSize=20,"
            f"PrimaryColour=&HFFFFFF,OutlineColour=&H000000,"
            f"Outline=2,Shadow=1,Alignment=2,MarginV=30'"
        )
        cmd_phase2 = [
            "ffmpeg", "-y", "-i", merged_audio_path,
            "-vf", subtitle_filter,
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-c:a", "copy",
            output_path
        ]
        logger.info(f"Phase 2: burning subtitles from {srt_path}...")
        result2 = subprocess.run(cmd_phase2, check=False,
                                 stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        if result2.returncode != 0:
            err = result2.stderr.decode(errors="replace").strip()
            logger.error(f"Subtitle burn failed (using merged_audio as final): {err[-800:]}")
            import shutil
            shutil.copy2(merged_audio_path, output_path)
        else:
            logger.info("Phase 2 complete: subtitles burned")
    else:
        import shutil
        shutil.copy2(merged_audio_path, output_path)
        logger.info("No SRT — copied merged_audio as final output")

    return output_path


def get_video_duration(video_path: str) -> float:
    """Get video duration in seconds using ffprobe"""
    result = subprocess.run([
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        video_path
    ], capture_output=True, text=True)

    import json
    data = json.loads(result.stdout)
    return float(data["format"]["duration"])
