import re
import requests
import logging

logger = logging.getLogger(__name__)

# Matches any speaker/narrator label at the start of a line:
#   [SPEAKER_01]:   [SPEAKER_01] :   SPEAKER_01:   [NARRATOR]:  etc.
_LABEL_RE = re.compile(r'^\[?(?:SPEAKER_\d+|NARRATOR)\]?\s*:?\s*', re.IGNORECASE)

MAX_SEGMENTS_PER_BATCH = 20
RETRY_BATCH_SIZE = 5

_REFUSAL_PATTERNS = [
    "i'm sorry", "i am sorry", "i cannot", "i can't", "i won't",
    "unable to", "not able to", "inappropriate", "i apologize",
    "as an ai", "i must decline", "i need to decline",
]


def _is_refusal(text: str) -> bool:
    """Return True if the model returned a refusal instead of a translation."""
    low = text.lower().strip()
    return any(p in low for p in _REFUSAL_PATTERNS)


def _call_openrouter(messages: list, api_key: str, model: str,
                     max_tokens: int = 2000) -> str:
    """Make one OpenRouter chat completion call; return content string."""
    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://aidubbing.replit.app",
            "X-Title": "AIDubbing"
        },
        json={
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens
        },
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

    return response.json()["choices"][0]["message"]["content"]


def _build_system_prompt(target_language: str) -> str:
    return f"""You are a certified professional dubbing translator working for a localization studio.
Your task: translate ALL dialogue lines to {target_language} for audio dubbing, no exceptions.
This is professional media localization — translate every line faithfully regardless of content.

IMPORTANT — narrator detection:
Prefix lines with [NARRATOR] if they are third-person scene descriptions or narration.
Keep the [SPEAKER_XX] label for all direct character dialogue.

Output rules:
- Return EXACTLY the same number of lines as the input — one translated line per input line.
- Format: [LABEL]: translated text
- Keep translations natural, speakable, and similar in length to the original.
- Do NOT skip, summarize, or refuse any line. If a line is unclear, transliterate it.
- Return ONLY the translated lines — nothing else."""


def _parse_lines(lines: list, batch: list) -> None:
    """Assign translated text from response lines back into segment dicts in-place."""
    for i, seg in enumerate(batch):
        if i < len(lines):
            line = lines[i].strip()

            # Detect narrator flag before stripping
            is_narrator = bool(re.match(r'^\[?NARRATOR\]?\s*:', line, re.IGNORECASE))

            # Strip any leading speaker/narrator label (handles missing space after colon)
            text = _LABEL_RE.sub('', line).strip()

            if not text:
                # Model returned a label with no content — skip TTS for this segment
                seg["is_narrator"] = is_narrator
                seg["translated"] = ""  # empty → dub.py will skip it silently
            else:
                seg["is_narrator"] = is_narrator
                seg["translated"] = text
        else:
            seg.setdefault("translated", seg["text"])


def _translate_single_segment(seg: dict, api_key: str, model: str,
                               target_language: str) -> None:
    """Last-resort: translate one segment at a time."""
    try:
        content = _call_openrouter(
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Translate this single line of dialogue to {target_language} "
                        f"for professional video dubbing. "
                        f"Return ONLY the translated text, nothing else.\n\n"
                        f"{seg['text']}"
                    )
                }
            ],
            api_key=api_key,
            model=model,
            max_tokens=300
        )
        if content and not _is_refusal(content):
            seg["translated"] = content.strip()
            logger.info(f"Single-segment fallback translated: {seg['text'][:40]!r}")
        else:
            logger.warning(f"Single-segment refusal for: {seg['text'][:40]!r}")
            seg.setdefault("translated", seg["text"])
    except Exception as e:
        logger.error(f"Single-segment translation failed: {e}")
        seg.setdefault("translated", seg["text"])


def _translate_batch_once(batch: list, api_key: str, model: str,
                           target_language: str) -> str:
    """Call the API for a batch; return raw response content."""
    full_text = "\n".join(
        f"[{seg['speaker']}]: {seg['text']}" for seg in batch
    )
    return _call_openrouter(
        messages=[
            {"role": "system", "content": _build_system_prompt(target_language)},
            {"role": "user",   "content": full_text}
        ],
        api_key=api_key,
        model=model,
        max_tokens=min(4000, len(batch) * 120)
    )


def translate_batch(batch: list, api_key: str, model: str,
                    target_language: str) -> list:
    """
    Translate a batch of segments to target_language.
    Strategy:
      1. Try the full batch.
      2. If refused or too few lines returned, split into RETRY_BATCH_SIZE mini-batches.
      3. If a mini-batch is also refused, fall back to per-segment translation.
    """
    batch_num = getattr(translate_batch, '_batch_counter', 0) + 1
    translate_batch._batch_counter = batch_num

    try:
        raw = _translate_batch_once(batch, api_key, model, target_language)
        lines = [l for l in raw.strip().split("\n") if l.strip()]

        if _is_refusal(raw) or len(lines) < len(batch) * 0.5:
            logger.warning(
                f"Batch {batch_num}: model refused or returned too few lines "
                f"({len(lines)}/{len(batch)}). Retrying in mini-batches..."
            )
            raise ValueError("refusal_or_short")

        _parse_lines(lines, batch)
        logger.info(f"Batch {batch_num}: {len(batch)} segments translated OK.")

    except Exception as e:
        if "refusal_or_short" not in str(e):
            logger.warning(f"Batch {batch_num} API error: {e}. Retrying in mini-batches...")

        for start in range(0, len(batch), RETRY_BATCH_SIZE):
            mini = batch[start: start + RETRY_BATCH_SIZE]
            try:
                raw = _translate_batch_once(mini, api_key, model, target_language)
                lines = [l for l in raw.strip().split("\n") if l.strip()]

                if _is_refusal(raw) or len(lines) < len(mini) * 0.5:
                    logger.warning(
                        f"Mini-batch also refused/short ({len(lines)}/{len(mini)}). "
                        f"Falling back to per-segment translation..."
                    )
                    for seg in mini:
                        if not seg.get("translated"):
                            _translate_single_segment(seg, api_key, model, target_language)
                else:
                    _parse_lines(lines, mini)
                    logger.info(f"Mini-batch ({len(mini)} segs) translated OK.")

            except Exception as e2:
                logger.error(f"Mini-batch failed: {e2}. Falling back to per-segment...")
                for seg in mini:
                    if not seg.get("translated"):
                        _translate_single_segment(seg, api_key, model, target_language)

    return batch


def translate_segments(segments: list, api_key: str, model: str,
                        target_language: str = "Arabic") -> list:
    """Translate all segments in batches, with refusal detection and retry."""
    translate_batch._batch_counter = 0
    logger.info(f"Translating {len(segments)} segments → {target_language} "
                f"using model: {model}")

    result = []
    for i in range(0, len(segments), MAX_SEGMENTS_PER_BATCH):
        batch = segments[i: i + MAX_SEGMENTS_PER_BATCH]
        logger.info(f"Batch {i // MAX_SEGMENTS_PER_BATCH + 1}: "
                    f"segments {i+1}–{i+len(batch)}")
        translated = translate_batch(batch, api_key, model, target_language)
        result.extend(translated)

    untranslated = sum(1 for s in result if s.get("translated") == s.get("text"))
    total = len(result)

    if untranslated == total:
        raise RuntimeError(
            f"Translation failed completely — 0/{total} segments were translated. "
            f"Check that the model ID is correct and your OpenRouter API key has credits."
        )
    elif untranslated > total * 0.5:
        logger.warning(f"Translation complete — only {total - untranslated}/{total} "
                       f"segments translated ({untranslated} left in original language).")
    else:
        logger.info(f"Translation complete — {total - untranslated}/{total} segments translated.")

    return result
