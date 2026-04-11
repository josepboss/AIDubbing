import requests
import logging

logger = logging.getLogger(__name__)


def generate_segment_audio(text: str, gender: str, settings: dict) -> bytes:
    """Generate TTS for a single segment with gender-appropriate voice"""
    provider = settings.get("tts_provider", "azure")

    if provider == "azure":
        voice = settings.get("male_voice", "ar-EG-ShakirNeural") \
                if gender == "male" \
                else settings.get("female_voice", "ar-EG-SalmaNeural")
        return _azure_tts(text, settings["azure_tts_key"],
                          settings["azure_tts_region"], voice)
    elif provider == "elevenlabs":
        voice_id = settings.get("elevenlabs_male_voice_id") \
                   if gender == "male" \
                   else settings.get("elevenlabs_female_voice_id")
        return _elevenlabs_tts(text, settings["elevenlabs_api_key"], voice_id)
    else:
        return _openai_tts(text, settings["openai_api_key"],
                           "onyx" if gender == "male" else "nova")


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
