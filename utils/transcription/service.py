from .source import YouTubeSource, FileSource, UploadSource
from .transcript import WhisperTranscriber

class TranscriptionService:
    def __init__(self, model_size='base'):
        self.transcriber = WhisperTranscriber(model_size)

    def get_transcription(self, source):
        if isinstance(source, str) and ('youtube.com' in source or 'youtu.be' in source):
            return YouTubeSource(source).get_transcript()
        elif isinstance(source, str):
            return FileSource(source, self.transcriber).get_transcript()
        elif hasattr(source, 'name') and hasattr(source, 'getvalue'):
            return UploadSource(source, self.transcriber).get_transcript()
        else:
            raise ValueError("Unsupported source type for transcription")
