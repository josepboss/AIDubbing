import json
import os

SETTINGS_PATH = os.path.join(os.path.dirname(__file__), "..", "settings.json")

DEFAULT_SETTINGS = {
    "openrouter_api_key": "",
    "openrouter_model": "google/gemini-2.0-flash-lite-001",
    "translation_model": "qwen/qwen-2.5-72b-instruct",
    "tts_provider": "azure",
    "azure_tts_key": "",
    "azure_tts_region": "eastus",
    "male_voice": "ar-EG-ShakirNeural",
    "female_voice": "ar-EG-SalmaNeural",
    "openai_api_key": "",
    "elevenlabs_api_key": "",
    "elevenlabs_male_voice_id": "",
    "elevenlabs_female_voice_id": "",
    "target_language": "Arabic",
    "whisper_model": "base",
    "hf_token": ""
}


def load_settings() -> dict:
    path = os.path.abspath(SETTINGS_PATH)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        merged = {**DEFAULT_SETTINGS, **data}
        return merged
    return dict(DEFAULT_SETTINGS)


def save_settings(settings: dict) -> None:
    path = os.path.abspath(SETTINGS_PATH)
    merged = {**DEFAULT_SETTINGS, **settings}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)
