from urllib.parse import urlparse, parse_qs
import os
import yt_dlp

class CachedYouTubeDownloader:
    def __init__(self, cache_dir="data/cache_videos"):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def _get_video_id(self, url):
        parsed = urlparse(url)
        if parsed.netloc == 'youtu.be':
            return parsed.path.strip('/')
        if parsed.netloc in ['www.youtube.com', 'youtube.com']:
            query = parse_qs(parsed.query)
            if 'v' in query:
                return query['v'][0]
            if '/embed/' in parsed.path:
                return parsed.path.split('/embed/')[1]
            if '/watch' in parsed.path:
                return parse_qs(parsed.query)['v'][0]
        return None

    def download_if_needed(self, url):
        video_id = self._get_video_id(url)
        if not video_id:
            raise ValueError("Could not extract video ID from URL")

        cached_path = os.path.join(self.cache_dir, f"{video_id}.mp4")
        if os.path.exists(cached_path):
            return cached_path 
    
        ydl_opts = {
            "format": "bestvideo*+bestaudio/best",
            "outtmpl": cached_path,
            "quiet": True,
            "no_warnings": True,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            video_path = ydl.prepare_filename(info)
        return video_path
