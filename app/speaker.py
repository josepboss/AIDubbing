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

        raw = pipeline(audio_input)

        # pyannote ≥ 4.x returns a DiarizeOutput dataclass; extract the
        # Annotation from .speaker_diarization.  Older versions return the
        # Annotation directly (has .itertracks).
        if hasattr(raw, "speaker_diarization"):
            diarization = raw.speaker_diarization
        elif hasattr(raw, "itertracks"):
            diarization = raw
        else:
            raise ValueError(f"Unrecognised pyannote output type: {type(raw)}")

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
    Detect gender per speech turn (consecutive segments with no gap > 0.5 s).
    Analysing whole turns (instead of individual short segments) is far more
    robust — a single voiced segment can be too short to produce a stable F0.

    Sets seg['gender'] on every segment in-place.
    Returns True if at least one turn was successfully analysed.
    """
    try:
        import librosa
        import numpy as np

        # ── 1. Group segments into speech turns ───────────────────────────────
        TURN_GAP = 0.5   # seconds; a gap larger than this starts a new turn

        turns: list[list[int]] = []   # list of lists of segment indices
        current: list[int] = []
        for i, seg in enumerate(segments):
            if not current:
                current.append(i)
            else:
                prev_end = segments[current[-1]]["end"]
                if seg["start"] - prev_end > TURN_GAP:
                    turns.append(current)
                    current = [i]
                else:
                    current.append(i)
        if current:
            turns.append(current)

        logger.info(f"Speaker: grouped {len(segments)} segments into {len(turns)} speech turns")

        MIN_ANALYSIS_DURATION = 2.0   # extend short turns to at least this many seconds
        turn_genders: list[str | None] = []
        any_success = False

        for turn_idx, indices in enumerate(turns):
            turn_start = segments[indices[0]]["start"]
            turn_end   = segments[indices[-1]]["end"]
            raw_dur    = turn_end - turn_start

            # For very short turns, extend the window around the turn so we get
            # more voiced frames and a more reliable F0 estimate.
            if raw_dur < MIN_ANALYSIS_DURATION:
                pad = (MIN_ANALYSIS_DURATION - raw_dur) / 2
                analysis_start = max(0.0, turn_start - pad)
            else:
                analysis_start = turn_start

            duration = min(max(raw_dur, MIN_ANALYSIS_DURATION), 10.0)

            if raw_dur < 0.15:
                turn_genders.append(None)   # too short — fill in during smoothing
                continue

            tmp_wav = tempfile.mktemp(suffix=".wav")
            try:
                ret = subprocess.run([
                    "ffmpeg", "-y", "-i", video_path,
                    "-ss", str(analysis_start), "-t", str(duration),
                    "-ac", "1", "-ar", "16000", tmp_wav
                ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

                if ret.returncode != 0 or not os.path.exists(tmp_wav):
                    turn_genders.append(None)
                    continue

                y, sr = librosa.load(tmp_wav, sr=16000)
                f0, voiced, _ = librosa.pyin(y, fmin=50, fmax=500, sr=sr)

                if f0 is not None and voiced is not None:
                    valid = f0[voiced]
                    valid = valid[~np.isnan(valid)]
                    if len(valid) > 0:
                        mean_f0 = float(np.mean(valid))
                        gender  = "female" if mean_f0 > 165 else "male"
                        turn_genders.append(gender)
                        logger.info(
                            f"Turn {turn_idx+1}/{len(turns)} "
                            f"[{turn_start:.1f}-{turn_end:.1f}s, {len(indices)} segs]: "
                            f"F0={mean_f0:.0f}Hz → {gender}"
                        )
                        any_success = True
                        continue

                turn_genders.append(None)

            finally:
                if os.path.exists(tmp_wav):
                    os.remove(tmp_wav)

        # ── 2. Smoothing pass ─────────────────────────────────────────────────
        # Fill None entries by propagating from neighbours, then fix isolated
        # anomalies (a single turn that differs from both neighbours is flipped).
        # This corrects short-turn mislabels caused by ambiguous F0 estimates.
        resolved = list(turn_genders)

        # Forward-fill then backward-fill any None entries
        last = "male"
        for i, g in enumerate(resolved):
            if g is not None:
                last = g
            else:
                resolved[i] = last
        last = resolved[-1]
        for i in range(len(resolved) - 1, -1, -1):
            if turn_genders[i] is None:
                resolved[i] = last
            else:
                last = resolved[i]

        # Flip isolated anomalies: if turn[i] differs from both turn[i-1] and
        # turn[i+1], replace it with its neighbours' value.
        smoothed = list(resolved)
        for i in range(1, len(smoothed) - 1):
            if smoothed[i] != smoothed[i - 1] and smoothed[i] != smoothed[i + 1]:
                smoothed[i] = smoothed[i - 1]
                logger.info(
                    f"Turn {i+1}: smoothed isolated gender anomaly "
                    f"({resolved[i]} → {smoothed[i]})"
                )

        # ── 3. Assign smoothed genders back to segments ───────────────────────
        for turn_idx, indices in enumerate(turns):
            gender = smoothed[turn_idx] if turn_idx < len(smoothed) else "male"
            for i in indices:
                segments[i]["gender"] = gender

        return any_success

    except Exception as e:
        logger.warning(f"Per-segment pitch analysis failed: {e}")
        return False
