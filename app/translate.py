import requests
import logging

logger = logging.getLogger(__name__)

MAX_SEGMENTS_PER_BATCH = 30


def _build_system_prompt(target_language: str) -> str:
    return f"""You are a professional dubbing translator specializing in Chinese drama localization.

Target language: {target_language}

Rules:
- Translate for DUBBING not subtitles — text must sound natural when spoken aloud
- Match the emotional intensity of each line (angry = strong words, sad = soft words)
- Keep sentence length similar to original so it fits the audio timing
- Use colloquial natural {target_language} not formal/literal translation
- Preserve character personality in their speech style
- Narrator lines [NARRATOR] should be dramatic and descriptive
- Never translate names — keep Chinese character names as-is
- For Arabic: use Modern Standard Arabic (فصحى مبسطة) not dialect
- Identify narrator lines (third-person descriptions, scene-setting, commentary not spoken by a character) and prefix them with [NARRATOR]
- All other character dialogue lines keep their original [SPEAKER_XX] label

Return ONLY the translated dialogue in exact same format as input.
Each line must start with the same speaker label."""


def _translate_single(text: str, target_language: str, api_key: str, model: str) -> str:
    """Translate a single very short segment without batching context."""
    try:
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        f"You are a dubbing translator. Translate the following short phrase to "
                        f"{target_language}. Return ONLY the translated text, nothing else."
                    )
                },
                {"role": "user", "content": text}
            ],
            "max_tokens": 60
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
            timeout=30
        )
        if response.ok:
            return response.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        logger.warning(f"_translate_single failed for '{text}': {e}")
    return text


def translate_batch(batch: list, api_key: str, model: str, target_language: str) -> list:
    full_text = "\n".join([
        f"[{seg['speaker']}]: {seg['text']}"
        for seg in batch
    ])

    system_prompt = _build_system_prompt(target_language)

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
    lines = [l for l in translated_text.strip().split("\n") if l.strip()]

    for i, seg in enumerate(batch):
        if i < len(lines):
            line = lines[i].strip()
            if line.startswith("[NARRATOR]"):
                seg["is_narrator"] = True
                text = line[len("[NARRATOR]"):].strip()
                seg["translated"] = text.lstrip(":").strip()
            elif "]: " in line:
                seg["is_narrator"] = False
                seg["translated"] = line.split("]: ", 1)[1].strip()
            else:
                seg["is_narrator"] = False
                seg["translated"] = line.strip()
        else:
            seg["is_narrator"] = False
            seg["translated"] = seg["text"]

    return batch


def translate_segments(segments: list, api_key: str, model: str,
                       target_language: str = "Arabic") -> list:
    """Translate all segments in batches, routing very short ones to a fast single call."""
    logger.info(f"Translating {len(segments)} segments using model: {model}")

    # Split segments: short ones (< 3 words) go to _translate_single,
    # the rest are batched together for context-aware translation.
    short_indices = []
    batch_segments = []

    for idx, seg in enumerate(segments):
        word_count = len(seg.get("text", "").split())
        if word_count < 3:
            short_indices.append(idx)
        else:
            batch_segments.append(seg)

    # Translate short segments individually (fast, no context needed)
    for idx in short_indices:
        seg = segments[idx]
        logger.info(f"Short segment ({seg['text']!r}) — direct translation")
        seg["translated"] = _translate_single(seg["text"], target_language, api_key, model)
        seg.setdefault("is_narrator", False)

    # Translate the rest in batches of MAX_SEGMENTS_PER_BATCH
    result_batches = []
    for i in range(0, len(batch_segments), MAX_SEGMENTS_PER_BATCH):
        batch = batch_segments[i: i + MAX_SEGMENTS_PER_BATCH]
        logger.info(f"Translating batch {i // MAX_SEGMENTS_PER_BATCH + 1} "
                    f"({len(batch)} segments)...")
        translated = translate_batch(batch, api_key, model, target_language)
        result_batches.extend(translated)

    logger.info("Translation complete.")
    return segments
