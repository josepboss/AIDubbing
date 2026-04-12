import subprocess
import os
import logging
import json
import re

logger = logging.getLogger(__name__)

SUPPORTED_SITES = [
    "youku.com", "iqiyi.com", "wetv.vip", "youtube.com",
    "youtu.be", "bilibili.com", "vimeo.com", "dailymotion.com"
]


def get_video_info(url: str) -> dict:
    """Get video metadata without downloading"""
    try:
        result = subprocess.run(
            ["yt-dlp", "--dump-json", "--no-download", url],
            capture_output=True,
            text=True,
            timeout=30
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
            "site": info.get("extractor", "")
        }
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "Timeout — site may be blocking requests"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def download_video(url: str, job_id: str, quality: str = "720p",
                   progress_callback=None) -> dict:
    """Download video using yt-dlp"""
    output_dir = f"app/uploads/{job_id}"
    os.makedirs(output_dir, exist_ok=True)
    # Use a safe fixed filename — title comes from get_video_info() separately
    output_template = f"{output_dir}/video.%(ext)s"

    cmd = [
        "yt-dlp",
        "--format", _get_format(quality),
        "--output", output_template,
        "--no-playlist",
        "--merge-output-format", "mp4",
        "--no-write-info-json",   # avoid filename issues with special chars in title
        "--newline",
        url
    ]

    logger.info(f"Downloading: {url}")

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        video_path = None
        for line in process.stdout:
            line = line.strip()
            logger.info(f"yt-dlp: {line}")

            if "[download]" in line and "%" in line:
                match = re.search(r'(\d+\.?\d*)%', line)
                if match and progress_callback:
                    progress_callback(float(match.group(1)))

            if "Destination:" in line:
                match = re.search(r'Destination: (.+\.mp4)', line)
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
            files = [f for f in os.listdir(output_dir) if f.endswith('.mp4')]
            if files:
                video_path = os.path.join(output_dir, files[0])

        if not video_path or not os.path.exists(video_path):
            return {"success": False, "error": "Download completed but output file not found"}

        file_size = os.path.getsize(video_path) / (1024 * 1024)
        logger.info(f"Downloaded: {video_path} ({file_size:.1f} MB)")

        return {
            "success": True,
            "video_path": os.path.abspath(video_path),
            "file_size_mb": round(file_size, 1)
        }

    except Exception as e:
        logger.error(f"Download error: {e}")
        return {"success": False, "error": str(e)}


def _get_format(quality: str) -> str:
    formats = {
        "best": "bestvideo[height<=1080]+bestaudio/best[height<=1080]",
        "720p": "bestvideo[height<=720]+bestaudio/best[height<=720]",
        "480p": "bestvideo[height<=480]+bestaudio/best[height<=480]",
        "audio_only": "bestaudio"
    }
    return formats.get(quality, formats["720p"])
