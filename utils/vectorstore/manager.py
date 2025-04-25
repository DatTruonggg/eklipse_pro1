from typing import List
from langchain.schema import Document
from .text import TextVectorStore
from .image import ImageVectorStore

class VectorStoreManager:
    def __init__(self, text_store: TextVectorStore, image_store: ImageVectorStore):
        self.text_store = text_store
        self.image_store = image_store

    def store_text(self, segments: List[dict]):
        if not self.text_store:
            raise ValueError("Text store is not initialized")
        self.text_store.store(segments)

    def store_image(self, image_paths: List[str], timestamps: List[float]):
        if not self.image_store:
            raise ValueError("Image store is not initialized")
        self.image_store.store(image_paths, timestamps)

    def search_text(self, query: str, k=3) -> List[Document]:
        if not self.text_store or not self.text_store.vectorstore:
            raise ValueError("Text vectorstore is not available")
        return self.text_store.vectorstore.similarity_search(query, k=k)

    def search_image(self, image_path: str, k=3) -> List[Document]:
        if not self.image_store or not self.image_store.vectorstore:
            raise ValueError("Image vectorstore is not available")
        return self.image_store.vectorstore.similarity_search(image_path, k=k)
