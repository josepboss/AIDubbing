import requests
import logging
import subprocess
import os

logger = logging.getLogger(__name__)

_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def translate_title(original_title: str, target_language: str,
                    api_key: str, model: str) -> str:
    """Translate video title to target language via OpenRouter."""
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": model,
                "messages": [{
                    "role": "user",
                    "content": (
                        f"Translate this video title to {target_language}.\n"
                        f"Make it catchy and suitable for YouTube.\n"
                        f"Keep it under 100 characters.\n"
                        f"Return ONLY the translated title, nothing else.\n\n"
                        f"Title: {original_title}"
                    )
                }],
                "max_tokens": 100
            },
            timeout=30
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        logger.error(f"Title translation failed: {e}")
        return original_title


def generate_thumbnail(video_path: str, output_dir: str,
                       title: str, style: str = "title_overlay") -> str | None:
    """
    Generate a thumbnail from the video.
    style='title_overlay' adds a text overlay at the bottom.
    style='frame_only'    extracts just the frame with no text.
    Returns path to thumbnail.jpg, or None on total failure.
    """
    os.makedirs(output_dir, exist_ok=True)
    thumbnail_path = os.path.join(output_dir, "thumbnail.jpg")

    try:
        result = subprocess.run([
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path
        ], capture_output=True, text=True)
        duration = float(result.stdout.strip() or "10")
        timestamp = max(duration * 0.2, 1.0)

        if style == "title_overlay" and title:
            safe_title = title.replace("'", "").replace(":", " ")[:50]
            vf = (
                "scale=1280:720:force_original_aspect_ratio=increase,"
                "crop=1280:720,"
                "drawbox=x=0:y=ih-110:w=iw:h=110:color=black@0.65:t=fill,"
                f"drawtext=text='{safe_title}':"
                f"fontfile={_FONT}:"
                "fontcolor=white:fontsize=30:"
                "x=(w-text_w)/2:y=h-70:"
                "shadowcolor=black:shadowx=2:shadowy=2"
            )
        else:
            vf = "scale=1280:720:force_original_aspect_ratio=increase,crop=1280:720"

        ret = subprocess.run([
            "ffmpeg", "-y",
            "-ss", str(timestamp),
            "-i", video_path,
            "-vframes", "1",
            "-vf", vf,
            "-q:v", "2",
            thumbnail_path
        ], capture_output=True)

        if ret.returncode != 0:
            raise RuntimeError(ret.stderr.decode(errors="replace")[-400:])

        logger.info(f"Thumbnail generated: {thumbnail_path}")
        return thumbnail_path

    except Exception as e:
        logger.error(f"Thumbnail generation failed: {e}")
        try:
            subprocess.run([
                "ffmpeg", "-y",
                "-ss", str(timestamp if 'timestamp' in dir() else 5),
                "-i", video_path,
                "-vframes", "1",
                "-q:v", "2",
                thumbnail_path
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            logger.info(f"Thumbnail fallback (no text) generated: {thumbnail_path}")
            return thumbnail_path
        except Exception as e2:
            logger.error(f"Thumbnail fallback also failed: {e2}")
            return None


def embed_metadata(video_path: str, output_path: str,
                   title: str, description: str = "",
                   language: str = "ara") -> str:
    """Embed title/description/language metadata into MP4 via ffmpeg stream-copy."""
    try:
        subprocess.run([
            "ffmpeg", "-y",
            "-i", video_path,
            "-metadata", f"title={title}",
            "-metadata", f"description={description}",
            "-metadata", f"language={language}",
            "-metadata", "artist=AIDubbing",
            "-c", "copy",
            output_path
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return output_path
    except Exception as e:
        logger.error(f"Metadata embedding failed: {e}")
        return video_path
