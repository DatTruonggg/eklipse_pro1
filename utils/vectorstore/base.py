import faiss
import numpy as np
from typing import List
from uuid import uuid4
from langchain.schema import Document
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


class BaseVectorStore:
    def __init__(self, embedder):
        self.embedder = embedder
        self.index = None
        self.vectorstore = None

    def _build_faiss(self, embeddings: np.ndarray, documents: List[Document]):
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        doc_ids = [str(uuid4()) for _ in documents]
        self.vectorstore = FAISS(
            embedding_function=self.embedder,
            index=self.index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={}
        )
        self.vectorstore.add_documents(documents=documents, ids=doc_ids)

