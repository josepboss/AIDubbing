import whisper
import json
import logging

logger = logging.getLogger(__name__)

_model = None


def get_model(size="base"):
    global _model
    if _model is None:
        logger.info(f"Loading Whisper model: {size}")
        _model = whisper.load_model(size)
    return _model


def transcribe_video(video_path: str, model_size: str = "base") -> list:
    """
    Returns list of segments:
    [{"start": 0.0, "end": 2.5, "text": "Hello", "speaker": "SPEAKER_00"}]
    """
    model = get_model(model_size)
    logger.info(f"Transcribing: {video_path}")

    result = model.transcribe(
        video_path,
        word_timestamps=True,
        verbose=False
    )

    segments = []
    for seg in result["segments"]:
        segments.append({
            "start": seg["start"],
            "end": seg["end"],
            "text": seg["text"].strip(),
            "speaker": "SPEAKER_00"
        })

    logger.info(f"Transcribed {len(segments)} segments")
    return segments
