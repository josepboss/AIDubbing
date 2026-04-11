import requests
import logging

logger = logging.getLogger(__name__)


def generate_segment_audio(text: str, gender: str, settings: dict,
                           is_narrator: bool = False) -> bytes:
    """Generate TTS for a single segment with gender-appropriate voice.
    Narrator lines use the dedicated narrator voice regardless of gender."""
    provider = settings.get("tts_provider", "azure")

    if provider == "azure":
        if is_narrator:
            voice = settings.get("narrator_voice", "ar-EG-SalmaNeural")
        elif gender == "male":
            voice = settings.get("male_voice", "ar-EG-ShakirNeural")
        else:
            voice = settings.get("female_voice", "ar-EG-SalmaNeural")
        return _azure_tts(text, settings["azure_tts_key"],
                          settings["azure_tts_region"], voice)
    elif provider == "elevenlabs":
        if is_narrator:
            voice_id = settings.get("elevenlabs_narrator_voice_id") or \
                       settings.get("elevenlabs_female_voice_id")
        elif gender == "male":
            voice_id = settings.get("elevenlabs_male_voice_id")
        else:
            voice_id = settings.get("elevenlabs_female_voice_id")
        return _elevenlabs_tts(text, settings["elevenlabs_api_key"], voice_id)
    else:
        if is_narrator:
            narrator_gender = settings.get("narrator_gender", "female")
            openai_voice = "nova" if narrator_gender == "female" else "onyx"
        else:
            openai_voice = "onyx" if gender == "male" else "nova"
        return _openai_tts(text, settings["openai_api_key"], openai_voice)


def _azure_tts(text, api_key, region, voice_name):
    token = requests.post(
        f"https://{region}.api.cognitive.microsoft.com/sts/v1.0/issueToken",
        headers={"Ocp-Apim-Subscription-Key": api_key},
        timeout=10
    ).text

    ssml = f"""<speak version='1.0' xml:lang='ar-EG'>
        <voice name='{voice_name}'>{text}</voice>
    </speak>"""

    response = requests.post(
        f"https://{region}.tts.speech.microsoft.com/cognitiveservices/v1",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/ssml+xml",
            "X-Microsoft-OutputFormat": "audio-48khz-192kbitrate-mono-mp3"
        },
        data=ssml.encode("utf-8"),
        timeout=30
    )
    response.raise_for_status()
    return response.content


def _elevenlabs_tts(text, api_key, voice_id):
    response = requests.post(
        f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
        headers={"xi-api-key": api_key, "Content-Type": "application/json"},
        json={
            "text": text,
            "model_id": "eleven_multilingual_v2",
            "voice_settings": {"stability": 0.5, "similarity_boost": 0.75}
        },
        timeout=60
    )
    response.raise_for_status()
    return response.content


def _openai_tts(text, api_key, voice):
    response = requests.post(
        "https://api.openai.com/v1/audio/speech",
        headers={"Authorization": f"Bearer {api_key}"},
        json={"model": "tts-1", "input": text, "voice": voice},
        timeout=60
    )
    response.raise_for_status()
    return response.content
