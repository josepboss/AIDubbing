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


def separate_background_audio(video_path: str, job_id: str) -> str | None:
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

        # ── 1. Extract stereo 44.1 kHz WAV with ffmpeg ──────────────────────
        audio_path = os.path.join(output_dir, "original.wav")
        ret = subprocess.run([
            "ffmpeg", "-y", "-i", video_path,
            "-ac", "2", "-ar", "44100", "-vn", audio_path
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        if ret.returncode != 0 or not os.path.exists(audio_path):
            logger.error("Failed to extract audio for demucs")
            return None

        # ── 2. Load model once (shared across all chunks) ────────────────────
        logger.info("Loading demucs model...")
        model = get_model("htdemucs")
        model.eval()
        stem_names = model.sources          # ['drums', 'bass', 'other', 'vocals']
        vocal_idx = stem_names.index("vocals")

        # ── 3. Process audio in 60-second chunks to bound RAM usage ──────────
        logger.info("Separating vocals from background audio in chunks...")
        info = sf.info(audio_path)
        sr = info.samplerate
        chunk_samples = sr * 60             # 60 seconds per chunk
        no_vocals_chunks = []

        with sf.SoundFile(audio_path, "r") as f:
            chunk_num = 0
            while True:
                data = f.read(chunk_samples)   # (samples, 2) or empty
                if len(data) == 0:
                    break
                chunk_num += 1
                if data.ndim == 1:
                    data = np.stack([data, data], axis=1)   # mono → fake stereo

                # (samples, 2) → (2, samples) → (1, 2, samples)
                waveform = torch.tensor(data.T, dtype=torch.float32).unsqueeze(0)

                with torch.no_grad():
                    # No segment= here — our 60-second outer chunks already
                    # bound memory. Internal sub-segmentation caused reshape
                    # errors when chunk length wasn't evenly divisible.
                    sources = apply_model(
                        model, waveform, device="cpu", progress=False
                    )
                # sources: (1, n_stems, 2, samples)
                # Sum all non-vocal stems into one stereo track
                no_vocals_stereo = sum(
                    sources[0, i] for i in range(len(stem_names)) if i != vocal_idx
                )  # shape: (2, samples)

                no_vocals_chunks.append(no_vocals_stereo.numpy().T)  # (samples, 2)

                # Free tensors immediately to keep RAM low
                del waveform, sources, no_vocals_stereo
                logger.info(f"Demucs: processed chunk {chunk_num}")

        if not no_vocals_chunks:
            logger.error("No audio chunks were processed by demucs")
            return None

        # ── 4. Concatenate all chunks and save ───────────────────────────────
        all_no_vocals = np.concatenate(no_vocals_chunks, axis=0)  # (total_samples, 2)
        no_vocals_path = os.path.join(output_dir, "no_vocals.wav")
        sf.write(no_vocals_path, all_no_vocals, sr)

        logger.info(f"Background audio extracted: {no_vocals_path}")
        return no_vocals_path

    except Exception as e:
        logger.error(f"Demucs separation failed: {e}")
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
                                   srt_path: str = None) -> str:
    """
    Merge video with dubbed audio and optionally burn subtitles.
    If vocal_removal is enabled (default), uses demucs to strip original
    vocals and keep only background music/effects at 40%.
    Falls back to original audio at 15% if demucs fails or is disabled.
    """
    if settings is None:
        settings = {}

    use_demucs = settings.get("vocal_removal", "demucs") == "demucs"
    background_path = None

    if use_demucs and job_id:
        background_path = separate_background_audio(video_path, job_id)

    # ── Audio filter ──────────────────────────────────────────────────────────
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

    # ── Subtitle / video filter ───────────────────────────────────────────────
    has_srt = srt_path and os.path.exists(srt_path)
    if has_srt:
        # Escape single quotes and colons for the ffmpeg subtitles filter
        escaped = srt_path.replace("\\", "\\\\").replace("'", "\\'").replace(":", "\\:")
        subtitle_filter = (
            f"subtitles='{escaped}'"
            f":force_style='FontName=DejaVu Sans,FontSize=20,"
            f"PrimaryColour=&HFFFFFF,OutlineColour=&H000000,"
            f"Outline=2,Shadow=1,Alignment=2,MarginV=30'"
        )
        video_filter = ["-vf", subtitle_filter]
        video_codec = ["-c:v", "libx264", "-preset", "fast", "-crf", "23"]
        logger.info(f"Burning subtitles from {srt_path}")
    else:
        video_filter = []
        video_codec = ["-c:v", "copy"]

    cmd = (
        ["ffmpeg", "-y", "-i", video_path, "-i", dubbed_audio_path]
        + extra_audio_inputs
        + ["-filter_complex", audio_filter]
        + video_filter
        + ["-map", "0:v:0", "-map", "[aout]"]
        + video_codec
        + ["-shortest", output_path]
    )

    result = subprocess.run(cmd, check=False,
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    if result.returncode != 0:
        err = result.stderr.decode(errors="replace").strip()
        if has_srt:
            # Subtitle burning failed — retry without subtitles so the job still completes
            logger.error(f"ffmpeg subtitle burn failed (retrying without subtitles): {err[-800:]}")
            cmd_no_sub = (
                ["ffmpeg", "-y", "-i", video_path, "-i", dubbed_audio_path]
                + extra_audio_inputs
                + ["-filter_complex", audio_filter]
                + ["-map", "0:v:0", "-map", "[aout]"]
                + ["-c:v", "copy"]
                + ["-shortest", output_path]
            )
            subprocess.run(cmd_no_sub, check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            raise subprocess.CalledProcessError(result.returncode, cmd, stderr=result.stderr)
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
