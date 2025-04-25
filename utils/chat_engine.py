import os
import cohere
from dotenv import load_dotenv
from logs import log

load_dotenv()

class ChatbotEngine:
    def __init__(self, model_name='command-r', temperature=0.3, max_tokens=300):
        api_key = os.getenv('COHERE_API_KEY')
        if not api_key:
            log.error("COHERE_API_KEY is missing from environment")
            raise ValueError("Missing COHERE_API_KEY")

        self.model = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.cohere_client = cohere.ClientV2(api_key)

    def generate_response(self, context: str, query: str) -> str:
        """Generate conversational response using Cohere's chat API"""
        try:
            messages = [
                {"role": "system", "content": context},
                {"role": "user", "content": query}
            ]

            response = self.cohere_client.chat(
                messages=messages,
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )

            return response.message.content[0].text

        except Exception as e:
            log.error(f"Response generation error: {e}")
            return "I'm unable to generate a response right now."
