import requests
import logging

logger = logging.getLogger(__name__)

MAX_SEGMENTS_PER_BATCH = 30


def translate_batch(batch: list, api_key: str, model: str, target_language: str) -> list:
    full_text = "\n".join([
        f"[{seg['speaker']}]: {seg['text']}"
        for seg in batch
    ])

    system_prompt = f"""You are a professional dubbing translator.
Translate the following dialogue to {target_language}.
Rules:
- Maintain the same speaker labels
- Keep translations natural and speakable (not literal)
- Match the emotional tone of each line
- Keep similar length to original when possible
- Return ONLY the translated dialogue in the same format
- Format: [SPEAKER_XX]: translated text"""

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": full_text}
        ],
        "max_tokens": 2000
    }

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://aidubbing.replit.app",
            "X-Title": "AIDubbing"
        },
        json=payload,
        timeout=120
    )

    if not response.ok:
        try:
            err_body = response.json()
        except Exception:
            err_body = response.text
        logger.error(f"OpenRouter {response.status_code}: {err_body}")
        raise RuntimeError(
            f"OpenRouter error {response.status_code}: "
            f"{err_body.get('error', {}).get('message', str(err_body)) if isinstance(err_body, dict) else err_body}"
        )

    translated_text = response.json()["choices"][0]["message"]["content"]
    lines = translated_text.strip().split("\n")

    for i, seg in enumerate(batch):
        if i < len(lines):
            line = lines[i]
            if "]: " in line:
                seg["translated"] = line.split("]: ", 1)[1].strip()
            else:
                seg["translated"] = line.strip()
        else:
            seg["translated"] = seg["text"]

    return batch


def translate_segments(segments: list, api_key: str, model: str,
                       target_language: str = "Arabic") -> list:
    """Translate all segments in batches to avoid token limits."""
    logger.info(f"Translating {len(segments)} segments using model: {model}")

    result = []
    for i in range(0, len(segments), MAX_SEGMENTS_PER_BATCH):
        batch = segments[i: i + MAX_SEGMENTS_PER_BATCH]
        logger.info(f"Translating batch {i // MAX_SEGMENTS_PER_BATCH + 1} "
                    f"({len(batch)} segments)...")
        translated = translate_batch(batch, api_key, model, target_language)
        result.extend(translated)

    logger.info("Translation complete.")
    return result
