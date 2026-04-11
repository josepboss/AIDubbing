import logging
import subprocess
import os
import tempfile

logger = logging.getLogger(__name__)


def detect_speakers(video_path: str, segments: list, hf_token: str = "") -> list:
    """
    Uses pyannote.audio to detect male/female speakers.
    Falls back to pitch-based gender detection if pyannote fails.
    """
    try:
        from pyannote.audio import Pipeline

        if not hf_token:
            raise ValueError("No HuggingFace token — using fallback")

        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token
        )

        diarization = pipeline(video_path)

        for seg in segments:
            mid = (seg["start"] + seg["end"]) / 2
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                if turn.start <= mid <= turn.end:
                    seg["speaker"] = speaker
                    break

        speakers = list(set(s["speaker"] for s in segments))
        gender_map = _detect_gender_pyannote(video_path, speakers, diarization)

        for seg in segments:
            seg["gender"] = gender_map.get(seg["speaker"], "male")

        return segments

    except Exception as e:
        logger.warning(f"Speaker detection failed: {e} — using pitch-based fallback")
        return _fallback_with_pitch(video_path, segments)


def _detect_gender_pyannote(video_path, speakers, diarization):
    """Pitch-based gender detection using pyannote diarization segments."""
    try:
        import librosa
        import numpy as np

        gender_map = {}
        for speaker in speakers:
            speaker_turns = [
                turn for turn, _, spk in diarization.itertracks(yield_label=True)
                if spk == speaker
            ]

            if not speaker_turns:
                gender_map[speaker] = "male"
                continue

            turn = speaker_turns[0]
            tmp_wav = tempfile.mktemp(suffix=".wav")
            try:
                subprocess.run([
                    "ffmpeg", "-y", "-i", video_path,
                    "-ss", str(turn.start), "-t", "10",
                    "-ac", "1", "-ar", "16000", tmp_wav
                ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

                y, sr = librosa.load(tmp_wav, sr=16000)
                f0, voiced, _ = librosa.pyin(y, fmin=50, fmax=500, sr=sr)
                voiced_f0 = f0[voiced] if f0 is not None else []
                import numpy as np
                valid = [v for v in voiced_f0 if v and not (v != v)]
                mean_pitch = float(np.mean(valid)) if valid else 150.0
            finally:
                if os.path.exists(tmp_wav):
                    os.remove(tmp_wav)

            gender_map[speaker] = "female" if mean_pitch > 165 else "male"
            logger.info(f"Speaker {speaker}: F0={mean_pitch:.0f}Hz → {gender_map[speaker]}")

        return gender_map

    except Exception as e:
        logger.warning(f"Gender detection failed: {e}")
        return {spk: "male" for spk in speakers}


def _fallback_with_pitch(video_path: str, segments: list) -> list:
    """
    No pyannote available. Try pitch analysis per speaker using ffmpeg + librosa.
    Falls back to alternating male/female only if pitch analysis also fails.
    """
    speakers = list(dict.fromkeys(s["speaker"] for s in segments))
    gender_map = _pitch_gender_for_speakers(video_path, segments, speakers)

    if not gender_map:
        # Last resort: alternate by speaker order
        logger.info("Using alternating gender fallback")
        gender_map = {spk: ("male" if i % 2 == 0 else "female")
                      for i, spk in enumerate(speakers)}

    for seg in segments:
        seg["gender"] = gender_map.get(seg["speaker"], "male")

    return segments


def _pitch_gender_for_speakers(video_path: str, segments: list, speakers: list):
    """
    Sample up to 3 segments per speaker, extract audio, and estimate
    fundamental frequency (F0) to determine gender.
    Female voices typically have F0 > 165 Hz, male < 165 Hz.
    Returns a gender_map dict or None if analysis failed entirely.
    """
    try:
        import librosa
        import numpy as np

        gender_map = {}

        for speaker in speakers:
            speaker_segs = [s for s in segments if s["speaker"] == speaker][:3]
            all_f0 = []

            for seg in speaker_segs:
                duration = min(seg["end"] - seg["start"], 5.0)
                if duration < 0.3:
                    continue

                tmp_wav = tempfile.mktemp(suffix=".wav")
                try:
                    ret = subprocess.run([
                        "ffmpeg", "-y", "-i", video_path,
                        "-ss", str(seg["start"]), "-t", str(duration),
                        "-ac", "1", "-ar", "16000", tmp_wav
                    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

                    if ret.returncode != 0 or not os.path.exists(tmp_wav):
                        continue

                    y, sr = librosa.load(tmp_wav, sr=16000)
                    f0, voiced, _ = librosa.pyin(y, fmin=50, fmax=500, sr=sr)
                    if f0 is not None and voiced is not None:
                        valid = f0[voiced]
                        valid = valid[~np.isnan(valid)]
                        all_f0.extend(valid.tolist())
                finally:
                    if os.path.exists(tmp_wav):
                        os.remove(tmp_wav)

            if all_f0:
                mean_f0 = float(np.mean(all_f0))
                gender = "female" if mean_f0 > 165 else "male"
                gender_map[speaker] = gender
                logger.info(f"Pitch fallback: {speaker} F0={mean_f0:.0f}Hz → {gender}")
            else:
                # No pitch data for this speaker — mark unknown, fill below
                gender_map[speaker] = None

        # Fill any unknowns by alternating within the unknowns
        unknowns = [s for s, g in gender_map.items() if g is None]
        for i, spk in enumerate(unknowns):
            gender_map[spk] = "male" if i % 2 == 0 else "female"

        return gender_map

    except Exception as e:
        logger.warning(f"Pitch gender analysis failed: {e}")
        return None
