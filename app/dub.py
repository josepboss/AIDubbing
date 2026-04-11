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
    Extract background audio (music + effects) without vocals using demucs.
    Uses the Python API directly (not the CLI subprocess) and feeds demucs a
    pre-loaded torch tensor so that torchcodec is never invoked — bypassing
    the missing libnppicc.so.13 that makes the CLI crash on this server.
    Returns path to no_vocals.wav, or None if separation fails.
    """
    try:
        import torch
        import numpy as np
        import librosa
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

        # ── 2. Load WAV as torch tensor (shape: [channels, samples]) ─────────
        logger.info("Loading audio for demucs separation...")
        data, sr = sf.read(audio_path, always_2d=True)   # (samples, channels)
        waveform = torch.tensor(data.T, dtype=torch.float32)  # (channels, samples)
        waveform = waveform.unsqueeze(0)                       # (1, channels, samples)

        # ── 3. Run demucs htdemucs model (two-stems: vocals / no_vocals) ─────
        logger.info("Separating vocals from background audio with demucs...")
        model = get_model("htdemucs")
        model.eval()

        with torch.no_grad():
            sources = apply_model(model, waveform, device="cpu", progress=False)
        # sources shape: (batch, stems, channels, samples)
        # stem order: drums, bass, other, vocals  (htdemucs 4-stem)
        # no_vocals = all stems except vocals
        stem_names = model.sources   # e.g. ['drums', 'bass', 'other', 'vocals']
        vocal_idx = stem_names.index("vocals")
        no_vocals = torch.cat(
            [sources[0, i] for i in range(len(stem_names)) if i != vocal_idx],
            dim=0
        )
        # Sum stereo pairs: result shape (2, samples)
        no_vocals_stereo = no_vocals.reshape(-1, 2, no_vocals.shape[-1]).sum(dim=0)

        # ── 4. Save no_vocals to WAV ─────────────────────────────────────────
        no_vocals_path = os.path.join(output_dir, "no_vocals.wav")
        sf.write(no_vocals_path, no_vocals_stereo.numpy().T, sr)

        logger.info(f"Background audio extracted: {no_vocals_path}")
        return no_vocals_path

    except Exception as e:
        logger.error(f"Demucs separation failed: {e}")
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
