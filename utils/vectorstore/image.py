from .base import BaseVectorStore
from PIL import Image
from typing import List
import faiss
import torch
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
import numpy as np 

class ImageVectorStore(BaseVectorStore):
    def __init__(self, embedder, processor, model):
        super().__init__(embedder=None)
        self.processor = processor  # CLIP image preprocessor
        self.model = model          # CLIP model for image embedding
        self._text_embedder = embedder  # Required only for FAISS wrapper

    def _embed_images(self, paths: List[str]):
        embeddings = []
        for path in paths:
            try:
                image = Image.open(path).convert("RGB")
                inputs = self.processor(images=image, return_tensors="pt")
                with torch.no_grad():
                    features = self.model.get_image_features(**inputs)
                # Normalize to improve distance-based similarity
                normed = features / features.norm(p=2, dim=-1, keepdim=True)
                embeddings.append(normed.squeeze().numpy())
            except Exception as e:
                print(f"Warning: failed to embed image {path} - {e}")
        return np.array(embeddings, dtype='float32')

    def store(self, image_paths: List[str], timestamps: List[float]):
        embeddings = self._embed_images(image_paths)

        documents = [
            Document(page_content="[IMAGE]", metadata={"timestamp": ts})
            for ts in timestamps
        ]

        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)

        # embedding_function is required by FAISS wrapper even if unused
        self.vectorstore = FAISS(
            embedding_function=self._text_embedder,
            index=index,
            documents=documents
        )