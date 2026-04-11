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


def merge_video_with_dubbed_audio(video_path: str, dubbed_audio_path: str,
                                   output_path: str) -> str:
    """Mix dubbed voices (100%) with original audio at 15% to keep background music/effects."""
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
