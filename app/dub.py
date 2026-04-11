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

    for i, seg in enumerate(segments):
        if not seg.get("translated"):
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
    Extract background audio (music + effects) without vocals
    using demucs audio separation.
    Returns path to no_vocals.wav, or None if separation fails.
    """
    output_dir = f"app/audio/{job_id}"
    os.makedirs(output_dir, exist_ok=True)

    audio_path = f"{output_dir}/original.wav"
    ret = subprocess.run([
        "ffmpeg", "-y", "-i", video_path,
        "-ac", "2", "-ar", "44100",
        "-vn", audio_path
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    if ret.returncode != 0 or not os.path.exists(audio_path):
        logger.error("Failed to extract audio for demucs")
        return None

    logger.info("Separating vocals from background audio with demucs...")

    result = subprocess.run([
        "python3", "-m", "demucs",
        "--two-stems", "vocals",
        "--out", output_dir,
        audio_path
    ], capture_output=True, text=True)

    if result.returncode != 0:
        logger.error(f"Demucs failed: {result.stderr[-500:]}")
        return None

    # Demucs outputs to: {output_dir}/htdemucs/{stem_name}/no_vocals.wav
    no_vocals_path = None
    for root, dirs, files in os.walk(output_dir):
        for f in files:
            if "no_vocals" in f:
                no_vocals_path = os.path.join(root, f)
                break
        if no_vocals_path:
            break

    if no_vocals_path and os.path.exists(no_vocals_path):
        logger.info(f"Background audio extracted: {no_vocals_path}")
        return no_vocals_path

    logger.warning("Could not find no_vocals output from demucs")
    return None


def merge_video_with_dubbed_audio(video_path: str, dubbed_audio_path: str,
                                   output_path: str, job_id: str = "",
                                   settings: dict = None) -> str:
    """
    Merge video with dubbed audio.
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

    if background_path and os.path.exists(background_path):
        logger.info("Using separated background audio (vocals removed)")
        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-i", dubbed_audio_path,
            "-i", background_path,
            "-filter_complex",
            "[2:a]volume=0.4[bg];"
            "[1:a]volume=1.0[dub];"
            "[bg][dub]amix=inputs=2:duration=longest[aout]",
            "-map", "0:v:0",
            "-map", "[aout]",
            "-c:v", "copy",
            "-shortest",
            output_path
        ]
    else:
        if use_demucs:
            logger.warning("Demucs failed — falling back to original audio at 15%")
        else:
            logger.info("Vocal removal disabled — using original audio at 15%")
        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-i", dubbed_audio_path,
            "-filter_complex",
            "[0:a]volume=0.15[original];"
            "[1:a]volume=1.0[dubbed];"
            "[original][dubbed]amix=inputs=2:duration=longest[aout]",
            "-map", "0:v:0",
            "-map", "[aout]",
            "-c:v", "copy",
            "-shortest",
            output_path
        ]

    subprocess.run(cmd, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
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
