from abc import ABC, abstractmethod
from moviepy.video.io.VideoFileClip import VideoFileClip
from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse, parse_qs
import os
import uuid
from logs import log 

class BaseSource(ABC):
    @abstractmethod
    def get_transcript(self):
        pass


class YouTubeSource(BaseSource):
    def __init__(self, url: str):
        self.video_id = self._extract_video_id(url)

    def _extract_video_id(self, url):
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

    def get_transcript(self):
        if not self.video_id:
            return None
        try:
            log.info(f"Fetching transcript for video_id={self.video_id}")
            transcript = YouTubeTranscriptApi.get_transcript(self.video_id)
            return {
                'text': " ".join([e['text'] for e in transcript]),
                'segments': [
                    {
                        'text': e['text'],
                        'start_time': e['start'],
                        'duration': e['duration'],
                        'end_time': e['start'] + e['duration']
                    } for e in transcript
                ]
            }
        except Exception as e:
            print(f"Error fetching YouTube transcript: {e}")
            return None


class FileSource(BaseSource):
    def __init__(self, video_path: str, transcriber):
        self.video_path = video_path
        self.transcriber = transcriber

    def get_transcript(self):
        audio_path = self._extract_audio()
        return self.transcriber.transcribe(audio_path)

    def _extract_audio(self):
        clip = VideoFileClip(self.video_path)
        audio_path = f'data/processed/audio_{uuid.uuid4()}.wav'
        clip.audio.write_audiofile(audio_path)
        return audio_path


class UploadSource(BaseSource):
    def __init__(self, uploaded_file, transcriber):
        self.uploaded_file = uploaded_file
        self.transcriber = transcriber

    def get_transcript(self):
        temp_path = f'data/processed/{uuid.uuid4()}_{self.uploaded_file.name}'
        with open(temp_path, 'wb') as f:
            f.write(self.uploaded_file.getvalue())
        return FileSource(temp_path, self.transcriber).get_transcript()
