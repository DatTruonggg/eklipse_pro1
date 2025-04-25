from logs import log

class ContextRetriever:
    def __init__(self, text_vectorstore, image_vectorstore, k=10):
        self.text_vectorstore = text_vectorstore
        self.image_vectorstore = image_vectorstore
        self.top_k = k

    def retrieve_text_context(self, query: str) -> list:
        """Retrieve top-k text segments related to the query"""
        try:
            return self.text_vectorstore.similarity_search(query, k=self.top_k)
        except Exception as e:
            log.error(f"Failed to retrieve text context: {e}")
            return []

    def retrieve_image_context(self, query_image) -> list:
        """Retrieve top-k image segments related to the image query"""
        if not self.image_vectorstore:
            log.warning("Image vectorstore not initialized.")
            return []

        try:
            return self.image_vectorstore.similarity_search(query_image, k=self.top_k)
        except Exception as e:
            log.error(f"Failed to retrieve image context: {e}")
            return []

    def format_documents(self, docs: list) -> str:
        return "\n".join([
            f"Content: {doc.page_content}\nMetadata: {doc.metadata}"
            for doc in docs
        ])

    def retrieve_context(self, query: str = None, image=None) -> str:
        """
        Unified interface for retrieving relevant context.
        Can retrieve text-based context or image-based context.
        """
        text_results = self.retrieve_text_context(query) if query else []
        image_results = self.retrieve_image_context(image) if image is not None else []

        context_text = self.format_documents(text_results)
        context_image = self.format_documents(image_results)

        return f"Text Context:\n{context_text}\n\nImage Context:\n{context_image}"
