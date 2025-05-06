import os
import cohere
import numpy as np
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
import base64
import cv2
from logs import log  # Using your custom log module

# Load environment variables from .env file
load_dotenv()

class EmbeddingService:
    def __init__(self):
        """
        Initializes the EmbeddingService by setting up the Cohere API client 
        using the API key from environment variables.
        """
        # Retrieve the API key from environment variables
        api_key = os.getenv('COHERE_API_KEY')
        
        if not api_key:
            log.error("COHERE_API_KEY environment variable is not set.")
            raise ValueError("COHERE_API_KEY environment variable is not set.")
        
        # Initialize Cohere client with the API key
        self.cohere_client = cohere.Client(api_key)
        log.info("Cohere client initialized successfully.")
    
    def embed_text(self, texts):
        """
        Generate embeddings for the provided texts using HuggingFace's transformer model.

        Parameters:
        - texts (list of str): List of text strings to embed.

        Returns:
        - np.array: Array of text embeddings.
        """
        try:
            # Initialize HuggingFace embedding model
            self.text_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            # Generate embeddings for the provided texts
            log.info("Generating text embeddings...")
            response = self.text_model.embed_documents(texts)[0]
            log.info(f"Generated embeddings for {len(texts)} texts.")
            return np.array(response)
        except Exception as e:
            # Log the error
            log.error(f"Embedding error: {e}")
            return None
    
    def embed_image(self, frames, model='embed-english-v3.0'):
        """
        Convert frames to base64 format and generate embeddings using the Cohere API.

        Parameters:
        - frames (list): List of image frames (numpy arrays).
        - model (str): The Cohere model to use for embedding (default 'embed-english-v3.0').

        Returns:
        - np.array: Array of image embeddings.
        """
        log.info(f"Converting {len(frames)} frames to base64 for embedding.")
        # Convert frames to base64 format
        base64_frames = [self._frame_to_base64(frame) for frame in frames]
        
        try:
            # Request image embeddings from the Cohere API
            log.info("Generating image embeddings using Cohere...")
            response = self.cohere_client.embed(
                texts=base64_frames,
                model=model,
                input_type='search_document'
            )
            log.info(f"Generated embeddings for {len(frames)} frames.")
            return np.array(response.embeddings)
        except Exception as e:
            # Log the error
            log.error(f"Image embedding error: {e}")
            return None
    
    def _frame_to_base64(self, frame):
        """
        Convert an OpenCV frame to base64 format.

        Parameters:
        - frame (numpy.array): The image frame to convert.

        Returns:
        - str: Base64 encoded string of the frame.
        """
        # Encode the frame as a .jpg image
        log.debug("Converting frame to base64 format.")
        _, buffer = cv2.imencode('.jpg', frame)
        # Return the base64 encoded string
        base64_string = base64.b64encode(buffer).decode('utf-8')
        log.debug("Frame successfully converted to base64.")
        return base64_string
