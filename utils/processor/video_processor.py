import os
import cv2
import uuid
import shutil
import tempfile
import numpy as np
from pathlib import Path
from functools import lru_cache
import yt_dlp as youtube_dl
from TransNetV2.inference.transnetv2 import TransNetV2
from logs import log
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch
from .cache_video_id import CachedYouTubeDownloader

class VideoProcessor:
    def __init__(self, video_source):
        self.source = video_source
        self.video_path = None
        self.output_dir = "data/keyframes"
        os.makedirs(self.output_dir, exist_ok=True)
        self.temp_dir = tempfile.mkdtemp(prefix='/home/dattruong/Desktop/Job/Technical-Test/Eklipse/video_chatbot_')
        
        # CLIP model for future embedding use
        self.clip_model = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32", 
            device_map="cpu",  
            torch_dtype=torch.float32)
               
        self.clip_processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch32",
            cache_dir=os.path.expanduser("~/.cache/huggingface/clip"))
    
    def load(self):
        if isinstance(self.source, str) and ('youtube.com' in self.source or 'youtu.be' in self.source):
            downloader = CachedYouTubeDownloader()
            self.video_path = downloader.download_if_needed(self.source)
            return self.video_path
        elif hasattr(self.source, 'name') and hasattr(self.source, 'getvalue'):
            return self._process_uploaded_file(self.source)
        else:
            raise ValueError("Unsupported video source format")
        
    def cleanup(self):
        try:
            if self.video_path and os.path.exists(self.video_path):
                os.remove(self.video_path)
            shutil.rmtree(self.temp_dir, ignore_errors=True)
            log.info("Temporary files cleaned up")
        except Exception as e:
            log.error(f"Cleanup error: {e}")

    def _generate_temp_filename(self, ext):
        return os.path.join(self.temp_dir, f"{uuid.uuid4()}.{ext}")

    # @lru_cache(maxsize=2)
    # def _download_youtube_video(self):
    #     log.info(f"Downloading YouTube video: {self.source}")
    #     try:
    #         temp_path = self._generate_temp_filename("mp4")

    #         ydl_opts = {
    #             'format': 'bestvideo*+bestaudio/best',
    #             'outtmpl': temp_path,
    #             'quiet': True,
    #             'no_warnings': True
    #         }
    #         with youtube_dl.YoutubeDL(ydl_opts) as ydl:
    #             info = ydl.extract_info(self.source, download=True)
    #             self.video_path = ydl.prepare_filename(info)

    #         log.info(f"Downloaded to {self.video_path}")
    #         return self.video_path

    #     except Exception as e:
    #         log.error(f"Download failed: {e}")
    #         raise
        
    def _process_uploaded_file(self, uploaded_file):
        ext = Path(uploaded_file.name).suffix.lstrip('.')
        temp_path = self._generate_temp_filename(ext)
        with open(temp_path, 'wb') as f:
            f.write(uploaded_file.getvalue())
        self.video_path = temp_path
        log.info(f"Uploaded file saved to {self.video_path}")
        return self.video_path

    def extract_frames(self, method='interval', interval=5):
        """Extract keyframes from video using TransNetV2 or interval method"""
        log.info("Keyframe extracting....")
        if not self.video_path:
            self.load()
            
        if method == 'transnet':
            log.info("Use Transnet method")

            return self._extract_frames_transnet()
        log.info("Use Interval method")
        return self._extract_frames_interval(interval)

    def _extract_frames_interval(self, interval=3):
        try: 
            log.info("Interval method extracting....")
            cap = cv2.VideoCapture(self.video_path)
            fps = 25
            keyframes = []
            timestamps = []
            
            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % (fps * interval) == int(fps * interval / 2):
                    timestamp = frame_count / fps
                    keyframe_path = os.path.join(self.output_dir, f"frame_{int(frame_count)}.jpg")
                    cv2.imwrite(keyframe_path, frame)
                    keyframes.append(keyframe_path)
                    timestamps.append(timestamp)
                
                frame_count += 1
            
            cap.release()
            log.info("extracted")
            return {"image": keyframes, "timestamp": timestamps}
        
        except Exception as e:
            log.error(f"Frame extraction error: {e}")
            raise e

    def _extract_frames_transnet(self):
        try:
            log.info("Transnet method extracting....")
            model = TransNetV2()
            _, single_frame_predictions, _ = model.predict_video(self.video_path)
            scenes = model.predictions_to_scenes(single_frame_predictions)
            cap = cv2.VideoCapture(self.video_path)
            fps = 25
            keyframes, timestamps = [], []

            for start, end in scenes:
                for rel_pos in [0.1, 0.5, 0.9]:
                    frame_idx = int(start + (end - start) * rel_pos)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ret, frame = cap.read()
                    if not ret:
                        continue
                    ts = frame_idx / fps
                    path = os.path.join(self.output_dir, f"scene_{frame_idx}.jpg")
                    cv2.imwrite(path, frame)
                    keyframes.append(path)
                    timestamps.append(ts)

            cap.release()
            log.info(f"Extracted {len(keyframes)} TransNet-based keyframes (start, mid, end)")
            return {"image": keyframes, "timestamp": timestamps}

        except Exception as e:
            log.error(f"TransNetV2 extraction failed: {e}")
            raise