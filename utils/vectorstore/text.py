from .base import BaseVectorStore
from langchain.schema import Document
import numpy as np 
from typing import List

class TextVectorStore(BaseVectorStore):
    def store(self, segments: List[dict]):
        texts, metadata = [], []
        for seg in segments:
            if 'text' in seg:
                texts.append(seg['text'])
                metadata.append({'start_time': seg.get('start_time'), 'duration': seg.get('duration')})
        embeddings = np.array(self.embedder.embed_documents(texts), dtype='float32')
        documents = [Document(page_content=texts[i], metadata=metadata[i]) for i in range(len(texts))]
        self._build_faiss(embeddings, documents)