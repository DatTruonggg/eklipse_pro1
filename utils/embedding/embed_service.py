import os
import cohere
import numpy as np
import base64
import cv2

from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
from logs import log

load_dotenv()

class EmbeddingService:
    def __init__(self, use_cohere: bool = False):
        self.use_cohere = use_cohere
        self.text_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.cohere_client = None

        if use_cohere:
            cohere_key = os.getenv('COHERE_API_KEY')
            if not cohere_key:
                log.warning("COHERE_API_KEY is missing from environment")
                raise ValueError("Missing COHERE_API_KEY in environment")
            self.cohere_client = cohere.Client(cohere_key)

    def embed_text(self, texts: list[str]) -> np.ndarray:
        """Generate embeddings for text (batch)"""
        if not texts:
            return np.array([])

        try:
            if self.use_cohere:
                response = self.cohere_client.embed(
                    texts=texts,
                    model='embed-english-v3.0',
                    input_type='search_document'
                )
                return np.array(response.embeddings)
            else:
                return np.array(self.text_model.embed_documents(texts), dtype='float32')
        except Exception as e:
            log.error(f"Text embedding failed: {e}")
            return np.array([])

    def embed_image(self, frames: list[np.ndarray]) -> np.ndarray:
        """
        Convert OpenCV frames to base64 and embed using Cohere.
        NOTE: Cohere currently doesn't officially support image input, this is hypothetical.
        """
        if not self.use_cohere:
            raise NotImplementedError("Image embedding requires Cohere API")

        base64_frames = [self._frame_to_base64(f) for f in frames]
        try:
            response = self.cohere_client.embed(
                texts=base64_frames,
                model='embed-english-v3.0',
                input_type='search_document'
            )
            return np.array(response.embeddings)
        except Exception as e:
            log.error(f"Image embedding failed: {e}")
            return np.array([])

    def _frame_to_base64(self, frame: np.ndarray) -> str:
        """Convert OpenCV frame (BGR) to base64 string"""
        _, buffer = cv2.imencode('.jpg', frame)
        return base64.b64encode(buffer).decode('utf-8')
