import numpy as np
from logs import log 

class ContextRetriever:
    def __init__(self, context_vectorstore, image_vectorstore=None, k=5):
        """
        Initializes the ContextRetriever with context and image vectorstores.
        
        Parameters:
        - context_vectorstore: The vectorstore that holds the context (e.g., text embeddings).
        - image_vectorstore: Optional image vectorstore for matching image embeddings (default is None).
        - k: Number of top results to retrieve (default is 5).
        """
        self.context_vectorstore = context_vectorstore
        self.image_vectorstore = image_vectorstore
        self.top_k = k
        log.info("ContextRetriever initialized.")

    def retrieve_context(self, query):
        """
        Retrieve the most relevant context based on embedding similarity.

        Parameters:
        - query: The query string to match against the context vectorstore.

        Returns:
        - A string containing the relevant context and metadata.
        """
        log.info(f"Retrieving context for the query: {query}")

        # Retrieve most relevant text context
        try:
            similarities = self.context_vectorstore.similarity_search(
                query, 
                k=self.top_k,
            )
            log.info(f"Retrieved {len(similarities)} similar text segments.")
        except Exception as e:
            log.error(f"Error during context retrieval: {e}")
            return "Error occurred during context retrieval."

        formatted_context = "\n".join([
            f"Content: {doc.page_content}\nMetadata: {doc.metadata}"
            for doc in similarities
        ])

        if self.image_vectorstore:
            try:
                # Retrieve most relevant image context (if available)
                image_similarities = self.image_vectorstore.similarity_search(
                    query,
                    k=self.top_k,
                )
                log.info(f"Retrieved {len(image_similarities)} similar image segments.")
                
                # Combine the results from text and image embeddings
                image_formatted_context = "\n".join([
                    f"Image Path: {img_doc.page_content}\nMetadata: {img_doc.metadata}"
                    for img_doc in image_similarities
                ])
                formatted_context += f"\n\nImage Context:\n{image_formatted_context}"

            except Exception as e:
                log.error(f"Error during image context retrieval: {e}")
                formatted_context += "\n\nError occurred during image context retrieval."

        # Return the formatted context along with metadata
        return f"Context:\n{formatted_context}\n\nUse this context to answer the user's query."

