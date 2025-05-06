import os
import cohere
from dotenv import load_dotenv
from logs import log  # Import custom logging module

# Load environment variables from .env file
load_dotenv()

class ChatbotEngine:
    def __init__(self):
        """
        Initialize the chatbot engine by setting up the Cohere API client using the API key.
        """
        # Retrieve API key from environment variables
        api_key = os.getenv('COHERE_API_KEY')
        
        # If the API key is missing, log an error and raise an exception
        if not api_key:
            log.error("Cohere API key not found in environment variables.")
            raise ValueError("COHERE_API_KEY environment variable is not set.")
        
        # Initialize the Cohere client with the API key
        self.cohere_client = cohere.ClientV2(api_key)
        log.info("Cohere client initialized successfully.")  # Log successful initialization

    def generate_response(self, context, query):
        """
        Generate a chatbot response using Cohere's chat API.

        Parameters:
        - context (str): The context in which the query is asked.
        - query (str): The user's query.

        Returns:
        - str: The response from the chatbot.
        """
        try:
            # Prepare the message structure for the chat API
            messages = [
                {"role": "system", "content": context},  # Context message
                {"role": "user", "content": query},  # User's query
            ]

            # Send the request to Cohere's chat API and get the response
            response = self.cohere_client.chat(messages=messages, model='command-r')
            
            # Return the response text
            return response.message.content[0].text

        except Exception as e:
            # Log any errors that occur during response generation
            log.error(f"Error while generating response: {e}")
            return "I'm unable to generate a response right now. Please try again later."
