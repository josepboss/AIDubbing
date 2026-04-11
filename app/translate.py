import requests
import logging

logger = logging.getLogger(__name__)


def translate_segments(segments: list, api_key: str, model: str,
                       target_language: str = "Arabic") -> list:
    """Translate all segments maintaining speaker context"""

    full_text = "\n".join([
        f"[{seg['speaker']}]: {seg['text']}"
        for seg in segments
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

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        },
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": full_text}
            ],
            "max_tokens": 4000
        },
        timeout=60
    )
    response.raise_for_status()

    translated_text = response.json()["choices"][0]["message"]["content"]

    lines = translated_text.strip().split("\n")
    for i, seg in enumerate(segments):
        if i < len(lines):
            line = lines[i]
            if "]: " in line:
                seg["translated"] = line.split("]: ", 1)[1].strip()
            else:
                seg["translated"] = line.strip()
        else:
            seg["translated"] = seg["text"]

    return segments
