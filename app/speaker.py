import logging
import subprocess
import os

logger = logging.getLogger(__name__)


def detect_speakers(video_path: str, segments: list, hf_token: str = "") -> list:
    """
    Uses pyannote.audio to detect male/female speakers.
    Falls back to alternating male/female if pyannote fails.
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
        gender_map = _detect_gender(video_path, speakers, diarization)

        for seg in segments:
            seg["gender"] = gender_map.get(seg["speaker"], "male")

        return segments

    except Exception as e:
        logger.warning(f"Speaker detection failed: {e} — using alternating voices")
        return _fallback_alternating(segments)


def _detect_gender(video_path, speakers, diarization):
    """Simple pitch-based gender detection"""
    try:
        import librosa
        import numpy as np

        gender_map = {}
        for speaker in speakers:
            segments_for_speaker = [
                turn for turn, _, spk in diarization.itertracks(yield_label=True)
                if spk == speaker
            ]

            if not segments_for_speaker:
                gender_map[speaker] = "male"
                continue

            turn = segments_for_speaker[0]
            cmd = [
                "ffmpeg", "-y",
                "-i", video_path,
                "-ss", str(turn.start),
                "-t", "10",
                "-ac", "1", "-ar", "16000",
                "/tmp/speaker_sample.wav"
            ]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            y, sr = librosa.load("/tmp/speaker_sample.wav", sr=16000)
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
            pitch_values = pitches[magnitudes > np.percentile(magnitudes, 75)]
            mean_pitch = np.mean(pitch_values[pitch_values > 0]) if len(pitch_values) > 0 else 150

            gender_map[speaker] = "female" if mean_pitch > 165 else "male"
            logger.info(f"Speaker {speaker}: pitch={mean_pitch:.0f}Hz → {gender_map[speaker]}")

        return gender_map

    except Exception as e:
        logger.warning(f"Gender detection failed: {e}")
        return {spk: "male" for spk in speakers}


def _fallback_alternating(segments):
    """Alternate male/female when detection fails"""
    speakers = list(set(s["speaker"] for s in segments))
    gender_map = {spk: ("male" if i % 2 == 0 else "female")
                  for i, spk in enumerate(speakers)}
    for seg in segments:
        seg["gender"] = gender_map.get(seg["speaker"], "male")
    return segments
