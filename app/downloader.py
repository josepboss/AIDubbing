import subprocess
import os
import logging
import json
import re
import tempfile

logger = logging.getLogger(__name__)

SUPPORTED_SITES = [
    "youku.com", "iqiyi.com", "wetv.vip", "youtube.com",
    "youtu.be", "bilibili.com", "vimeo.com", "dailymotion.com", "dai.ly"
]


# ── Public interface ───────────────────────────────────────────────────────────

def get_video_info(url: str) -> dict:
    """Return video metadata dict without downloading."""
    return _ytdlp_video_info(url)


def download_video(url: str, job_id: str, quality: str = "720p",
                   progress_callback=None) -> dict:
    """Download video via yt-dlp (handles all supported sites)."""
    return _ytdlp_download(url, job_id, quality, progress_callback)


# ── yt-dlp implementation ──────────────────────────────────────────────────────

_BILI_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

def _is_bilibili(url: str) -> bool:
    return "bilibili.com" in url or "b23.tv" in url


def _is_dailymotion(url: str) -> bool:
    return "dailymotion.com" in url or "dai.ly" in url


def _dailymotion_extra_args() -> list[str]:
    """Return extra yt-dlp flags for Dailymotion (CDN Referer bypass)."""
    return [
        "--add-headers", "Referer:https://www.dailymotion.com/",
        "--add-headers", "Origin:https://www.dailymotion.com",
    ]


def _get_bilibili_cookies() -> str:
    """Load bilibili_cookies string from settings.json."""
    try:
        from app.config import load_settings
        return load_settings().get("bilibili_cookies", "").strip()
    except Exception:
        return ""


def _bilibili_extra_args(cookies_file: str | None = None) -> list[str]:
    """Return extra yt-dlp flags needed to bypass Bilibili's 412 check."""
    args = [
        "--user-agent", _BILI_UA,
        "--add-headers", "Referer:https://www.bilibili.com",
        "--add-headers", "Origin:https://www.bilibili.com",
    ]
    if cookies_file:
        args += ["--cookies", cookies_file]
    return args


def _write_cookies_tmp(cookies_content: str) -> str | None:
    """Write cookie content to a named temp file, return path (caller must delete)."""
    if not cookies_content:
        return None
    # Auto-add Netscape header if missing
    if not cookies_content.startswith("# Netscape"):
        cookies_content = "# Netscape HTTP Cookie File\n" + cookies_content
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, prefix="bili_cookies_"
    )
    tmp.write(cookies_content)
    tmp.close()
    return tmp.name


def _ytdlp_video_info(url: str) -> dict:
    cookies_file = None
    if _is_bilibili(url):
        cookies_content = _get_bilibili_cookies()
        cookies_file = _write_cookies_tmp(cookies_content)
        extra = _bilibili_extra_args(cookies_file)
    elif _is_dailymotion(url):
        extra = _dailymotion_extra_args()
    else:
        extra = []
    try:
        result = subprocess.run(
            ["yt-dlp", "--dump-json", "--no-download", *extra, url],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            return {"success": False, "error": result.stderr[:500]}
        info = json.loads(result.stdout)
        return {
            "success": True,
            "title": info.get("title", "Unknown"),
            "duration": info.get("duration", 0),
            "thumbnail": info.get("thumbnail", ""),
            "uploader": info.get("uploader", ""),
            "formats": len(info.get("formats", [])),
            "site": info.get("extractor", ""),
        }
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "Timeout — site may be blocking requests"}
    except Exception as e:
        return {"success": False, "error": str(e)}
    finally:
        if cookies_file:
            try:
                os.unlink(cookies_file)
            except Exception:
                pass


def _ytdlp_download(url: str, job_id: str, quality: str = "720p",
                    progress_callback=None) -> dict:
    output_dir = f"app/uploads/{job_id}"
    os.makedirs(output_dir, exist_ok=True)
    output_template = f"{output_dir}/video.%(ext)s"

    cookies_file = None
    if _is_bilibili(url):
        cookies_content = _get_bilibili_cookies()
        cookies_file = _write_cookies_tmp(cookies_content)
        logger.info(f"Bilibili: cookies {'loaded (' + str(len(cookies_content)) + ' bytes)' if cookies_content else 'NOT set — may get 412'}")
        extra = _bilibili_extra_args(cookies_file)
    elif _is_dailymotion(url):
        logger.info("Dailymotion: using yt-dlp with Referer headers")
        extra = _dailymotion_extra_args()
    else:
        extra = []

    cmd = [
        "yt-dlp",
        "--format", _get_format(quality),
        "--output", output_template,
        "--no-playlist",
        "--merge-output-format", "mp4",
        "--no-write-info-json",
        "--newline",
        *extra,
        url,
    ]

    logger.info(f"Downloading: {url}")

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        video_path = None
        for line in process.stdout:
            line = line.strip()
            logger.info(f"yt-dlp: {line}")

            if "[download]" in line and "%" in line:
                match = re.search(r"(\d+\.?\d*)%", line)
                if match and progress_callback:
                    progress_callback(float(match.group(1)))

            if "Destination:" in line:
                match = re.search(r"Destination: (.+\.mp4)", line)
                if match:
                    video_path = match.group(1).strip()

            if "[Merger]" in line:
                match = re.search(r'Merging formats into "(.+\.mp4)"', line)
                if match:
                    video_path = match.group(1).strip()

        process.wait()

        if process.returncode != 0:
            return {"success": False, "error": "Download failed — video may be geo-restricted or require login"}

        if not video_path:
            files = [f for f in os.listdir(output_dir) if f.endswith(".mp4")]
            if files:
                video_path = os.path.join(output_dir, files[0])

        if not video_path or not os.path.exists(video_path):
            return {"success": False, "error": "Download completed but output file not found"}

        file_size = os.path.getsize(video_path) / (1024 * 1024)
        logger.info(f"Downloaded: {video_path} ({file_size:.1f} MB)")
        return {
            "success": True,
            "video_path": os.path.abspath(video_path),
            "file_size_mb": round(file_size, 1),
        }
    except Exception as e:
        logger.error(f"Download error: {e}")
        return {"success": False, "error": str(e)}
    finally:
        if cookies_file:
            try:
                os.unlink(cookies_file)
            except Exception:
                pass


def _get_format(quality: str) -> str:
    formats = {
        "best": "bestvideo[height<=1080]+bestaudio/best[height<=1080]",
        "720p": "bestvideo[height<=720]+bestaudio/best[height<=720]",
        "480p": "bestvideo[height<=480]+bestaudio/best[height<=480]",
        "audio_only": "bestaudio",
    }
    return formats.get(quality, formats["720p"])
