import logging
import subprocess
import os
import tempfile

logger = logging.getLogger(__name__)


def detect_speakers(video_path: str, segments: list, hf_token: str = "") -> tuple:
    """
    Uses pyannote.audio to detect male/female speakers.
    Falls back to pitch-based gender detection if pyannote fails.
    Returns (segments, method_name) where method_name is one of:
      'pyannote' | 'pitch_analysis' | 'alternating'
    """
    try:
        from pyannote.audio import Pipeline
        import torch

        if not hf_token:
            raise ValueError("No HuggingFace token — using fallback")

        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            token=hf_token
        )

        # torchcodec cannot decode video/audio directly in this environment
        # (missing libnppicc.so). Pre-extract mono 16 kHz WAV with ffmpeg and
        # pass pyannote a waveform dict to bypass the broken torchcodec path.
        tmp_wav = tempfile.mktemp(suffix=".wav")
        try:
            subprocess.run([
                "ffmpeg", "-y", "-i", video_path,
                "-ac", "1", "-ar", "16000", tmp_wav
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            import librosa
            data, sample_rate = librosa.load(tmp_wav, sr=16000, mono=True)
            waveform = torch.tensor(data, dtype=torch.float32).unsqueeze(0)
            audio_input = {"waveform": waveform, "sample_rate": sample_rate}
        finally:
            if os.path.exists(tmp_wav):
                os.remove(tmp_wav)

        diarization = pipeline(audio_input)

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

        logger.info("Speaker detection: pyannote/speaker-diarization-3.1")
        return segments, "pyannote"

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


def _fallback_with_pitch(video_path: str, segments: list) -> tuple:
    """
    No pyannote available. Analyse pitch per segment individually so that
    even when Whisper labels every segment 'SPEAKER_00' each one gets the
    correct gender based on its own audio content.
    Falls back to alternating male/female only if pitch analysis fails entirely.
    Returns (segments, method_name).
    """
    success = _pitch_gender_per_segment(video_path, segments)

    if success:
        method = "pitch_analysis"
        logger.info("Speaker gender: per-segment pitch analysis (F0 estimation)")
    else:
        method = "alternating"
        logger.info("Speaker gender: alternating fallback")
        # Alternate gender across unique speakers as a last resort
        speakers = list(dict.fromkeys(s["speaker"] for s in segments))
        gender_map = {spk: ("male" if i % 2 == 0 else "female")
                      for i, spk in enumerate(speakers)}
        for seg in segments:
            seg["gender"] = gender_map.get(seg["speaker"], "male")

    return segments, method


def _pitch_gender_per_segment(video_path: str, segments: list) -> bool:
    """
    Extract audio for EACH segment individually and estimate F0.
    Sets seg['gender'] directly on each segment.
    Female voices typically have F0 > 165 Hz, male < 165 Hz.
    Returns True if at least one segment was successfully analysed.
    """
    try:
        import librosa
        import numpy as np

        any_success = False

        for seg in segments:
            duration = min(seg["end"] - seg["start"], 5.0)
            if duration < 0.3:
                seg.setdefault("gender", "male")
                continue

            tmp_wav = tempfile.mktemp(suffix=".wav")
            try:
                ret = subprocess.run([
                    "ffmpeg", "-y", "-i", video_path,
                    "-ss", str(seg["start"]), "-t", str(duration),
                    "-ac", "1", "-ar", "16000", tmp_wav
                ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

                if ret.returncode != 0 or not os.path.exists(tmp_wav):
                    seg.setdefault("gender", "male")
                    continue

                y, sr = librosa.load(tmp_wav, sr=16000)
                f0, voiced, _ = librosa.pyin(y, fmin=50, fmax=500, sr=sr)

                if f0 is not None and voiced is not None:
                    valid = f0[voiced]
                    valid = valid[~np.isnan(valid)]
                    if len(valid) > 0:
                        mean_f0 = float(np.mean(valid))
                        seg["gender"] = "female" if mean_f0 > 165 else "male"
                        logger.info(
                            f"Seg [{seg['start']:.1f}-{seg['end']:.1f}s]: "
                            f"F0={mean_f0:.0f}Hz → {seg['gender']}"
                        )
                        any_success = True
                        continue

                seg.setdefault("gender", "male")

            finally:
                if os.path.exists(tmp_wav):
                    os.remove(tmp_wav)

        return any_success

    except Exception as e:
        logger.warning(f"Per-segment pitch analysis failed: {e}")
        return False
