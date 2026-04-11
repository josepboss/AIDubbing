# AIDubbing

AI-powered Arabic video dubbing tool.

## Features
- Upload MP4/MKV video files
- Transcribe speech with Whisper
- Auto-detect speakers and gender
- Translate to Arabic via OpenRouter
- Generate Arabic TTS (Azure, ElevenLabs, or OpenAI)
- Assemble and sync dubbed audio with FFmpeg

## Setup
```bash
pip install -r requirements.txt
python run.py
```

Open http://localhost:5002 and configure your API keys in the **Settings** tab.

## Required API Keys
- OpenRouter API Key (translation)
- Azure TTS Key + Region (default TTS provider)
