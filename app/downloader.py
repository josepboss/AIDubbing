import subprocess
import os
import logging
import json
import re
import tempfile
import urllib.request
import urllib.parse
import http.cookiejar
import uuid

logger = logging.getLogger(__name__)

SUPPORTED_SITES = [
    "youku.com", "iqiyi.com", "wetv.vip", "youtube.com",
    "youtu.be", "bilibili.com", "vimeo.com", "dailymotion.com"
]

_DM_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/137.0.0.0 Safari/537.36"
)
_DM_METADATA_URL = (
    "https://www.dailymotion.com/player/metadata/video/{video_id}"
    "?embedder=https%3A%2F%2Fwww.dailymotion.com&locale=en_US&dmV1st={visitor_id}"
)
_DM_OEMBED_URL = "https://www.dailymotion.com/services/oembed?url={url}&format=json"


# ── Dailymotion helpers ────────────────────────────────────────────────────────

def _is_dailymotion(url: str) -> bool:
    return "dailymotion.com" in url or "dai.ly" in url


def _extract_dm_video_id(url: str) -> str | None:
    for pattern in [
        r'dailymotion\.com/video/([a-zA-Z0-9]+)',
        r'dailymotion\.com/embed/video/([a-zA-Z0-9]+)',
        r'dai\.ly/([a-zA-Z0-9]+)',
    ]:
        m = re.search(pattern, url)
        if m:
            return m.group(1)
    return None


def _make_dm_opener() -> tuple[urllib.request.OpenerDirector, http.cookiejar.CookieJar]:
    """Build a urllib opener that tracks cookies, mimicking a browser session."""
    jar = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
    opener.addheaders = [
        ("User-Agent", _DM_UA),
        ("Accept", "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"),
        ("Accept-Language", "en-US,en;q=0.9"),
        ("Accept-Encoding", "identity"),
        ("Connection", "keep-alive"),
    ]
    return opener, jar


def _jar_to_cookie_str(jar: http.cookiejar.CookieJar) -> str:
    """Convert CookieJar contents to a single 'name=val; ...' cookie header value."""
    return "; ".join(f"{c.name}={c.value}" for c in jar)


def _dm_get_m3u8(video_id: str) -> tuple[str, dict, str]:
    """Return (m3u8_url, full_meta, cookie_str) for a Dailymotion video.

    Two-step browser simulation:
    1. Visit video page  → Dailymotion sets dmV1st + other session cookies
    2. Fetch player metadata with those cookies → signed m3u8 URL
    cookie_str must be forwarded to ffmpeg so Dailymotion's CDN accepts the stream.
    """
    opener, jar = _make_dm_opener()
    visitor_id = str(uuid.uuid4())

    # Step 1 – visit video page to pick up session cookies (dmV1st etc.)
    page_url = f"https://www.dailymotion.com/video/{video_id}"
    try:
        req0 = urllib.request.Request(
            page_url,
            headers={"Referer": "https://www.dailymotion.com/"}
        )
        with opener.open(req0, timeout=15) as _:
            pass
        logger.info(f"DM page visit OK — cookies: {_jar_to_cookie_str(jar)[:120]}")
    except Exception as e:
        logger.warning(f"DM page visit failed (continuing): {e}")

    # Step 2 – fetch player metadata with the session cookies
    meta_url = _DM_METADATA_URL.format(video_id=video_id, visitor_id=visitor_id)
    req1 = urllib.request.Request(
        meta_url,
        headers={
            "Referer": page_url,
            "Accept": "application/json, text/javascript, */*; q=0.01",
            "X-Requested-With": "XMLHttpRequest",
        },
    )
    with opener.open(req1, timeout=20) as resp:
        meta = json.loads(resp.read().decode())

    cookie_str = _jar_to_cookie_str(jar)
    logger.info(f"DM metadata OK — cookie_str len={len(cookie_str)}")

    qualities = meta.get("qualities", {})

    # Prefer the adaptive 'auto' HLS stream
    for stream in qualities.get("auto", []):
        if stream.get("type") == "application/x-mpegURL":
            return stream["url"], meta, cookie_str

    # Fall back to the highest numbered quality
    numbered = sorted(
        [k for k in qualities if k != "auto" and k.isdigit()],
        key=int, reverse=True
    )
    for q_key in numbered:
        for stream in qualities[q_key]:
            if stream.get("type") == "application/x-mpegURL":
                return stream["url"], meta, cookie_str

    raise ValueError("No m3u8 stream found in Dailymotion player metadata")


# ── Public interface ───────────────────────────────────────────────────────────

def get_video_info(url: str) -> dict:
    """Return video metadata dict without downloading."""
    if _is_dailymotion(url):
        return _dm_video_info(url)
    return _ytdlp_video_info(url)


def download_video(url: str, job_id: str, quality: str = "720p",
                   progress_callback=None) -> dict:
    """Download video. Uses direct ffmpeg for Dailymotion, yt-dlp for everything else."""
    if _is_dailymotion(url):
        return _dm_download(url, job_id, quality, progress_callback)
    return _ytdlp_download(url, job_id, quality, progress_callback)


# ── Dailymotion implementation ─────────────────────────────────────────────────

def _dm_video_info(url: str) -> dict:
    video_id = _extract_dm_video_id(url)
    if not video_id:
        return {"success": False, "error": "Could not extract Dailymotion video ID from URL"}
    try:
        _m3u8_url, meta, _cookies = _dm_get_m3u8(video_id)

        # Try oEmbed for a proper thumbnail URL
        thumbnail = meta.get("thumbnail_url", "")
        if not thumbnail:
            try:
                opener, _jar = _make_dm_opener()
                oembed_url = _DM_OEMBED_URL.format(url=urllib.parse.quote(url, safe=""))
                with opener.open(urllib.request.Request(oembed_url), timeout=10) as r:
                    oembed = json.loads(r.read().decode())
                thumbnail = oembed.get("thumbnail_url", "")
            except Exception:
                pass

        owner = meta.get("owner") or {}
        uploader = owner.get("screenname", "") if isinstance(owner, dict) else str(owner)

        return {
            "success": True,
            "title": meta.get("title", "Unknown"),
            "duration": meta.get("duration", 0),
            "thumbnail": thumbnail,
            "uploader": uploader,
            "formats": len(meta.get("qualities", {})),
            "site": "dailymotion",
        }
    except Exception as e:
        logger.error(f"Dailymotion metadata error: {e}")
        return {"success": False, "error": str(e)}


def _dm_download(url: str, job_id: str, quality: str = "720p",
                 progress_callback=None) -> dict:
    """Download Dailymotion video via direct m3u8 URL + ffmpeg (no curl_cffi needed)."""
    video_id = _extract_dm_video_id(url)
    if not video_id:
        return {"success": False, "error": "Could not extract Dailymotion video ID from URL"}

    try:
        m3u8_url, meta, cookie_str = _dm_get_m3u8(video_id)
    except Exception as e:
        return {"success": False, "error": f"Failed to fetch Dailymotion stream: {e}"}

    duration = meta.get("duration", 0)
    output_dir = f"app/uploads/{job_id}"
    os.makedirs(output_dir, exist_ok=True)
    output_path = f"{output_dir}/video.mp4"

    logger.info(f"Dailymotion: ffmpeg download — video_id={video_id}, cookies={bool(cookie_str)}")

    # Build the headers string; include Cookie if we captured any session cookies
    headers_str = (
        f"User-Agent: {_DM_UA}\r\n"
        f"Referer: https://www.dailymotion.com/video/{video_id}\r\n"
        f"Origin: https://www.dailymotion.com\r\n"
    )
    if cookie_str:
        headers_str += f"Cookie: {cookie_str}\r\n"

    cmd = [
        "ffmpeg", "-y",
        "-headers", headers_str,
        "-i", m3u8_url,
        "-c", "copy",
        output_path,
    ]

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in process.stdout:
            line = line.strip()
            logger.info(f"ffmpeg: {line}")
            if progress_callback and duration > 0:
                m = re.search(r"time=(\d+):(\d+):(\d+\.?\d*)", line)
                if m:
                    h, mn, s = int(m.group(1)), int(m.group(2)), float(m.group(3))
                    pct = min(100.0, (h * 3600 + mn * 60 + s) / duration * 100)
                    progress_callback(pct)

        process.wait()

        if process.returncode != 0:
            return {"success": False, "error": "ffmpeg failed to download Dailymotion stream"}

        if not os.path.exists(output_path):
            return {"success": False, "error": "Download completed but output file not found"}

        file_size = os.path.getsize(output_path) / (1024 * 1024)
        logger.info(f"Downloaded: {output_path} ({file_size:.1f} MB)")
        return {
            "success": True,
            "video_path": os.path.abspath(output_path),
            "file_size_mb": round(file_size, 1),
        }
    except Exception as e:
        logger.error(f"Dailymotion download error: {e}")
        return {"success": False, "error": str(e)}


# ── yt-dlp implementation (all other sites) ────────────────────────────────────

_BILI_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

def _is_bilibili(url: str) -> bool:
    return "bilibili.com" in url or "b23.tv" in url


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
    extra = _bilibili_extra_args(cookies_file) if _is_bilibili(url) else []
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
    extra = _bilibili_extra_args(cookies_file) if _is_bilibili(url) else []

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
