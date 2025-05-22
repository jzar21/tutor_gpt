from abc import ABC, abstractmethod
from rag_config import RAGArgs
from langchain.embeddings.base import Embeddings
import numpy as np
from rag_config import RAGArgs
import requests
import os
import sys
import google.generativeai as genai
from google.api_core import exceptions as google_exceptions


class AbstractLLM(ABC):
    def __init__(self, args: RAGArgs):
        super().__init__()
        self.args = args

    @abstractmethod
    def ask(self, prompt: dict) -> str:
        pass


class AbstractEmbedder(Embeddings):
    def __init__(self, args: RAGArgs):
        super().__init__()
        self.args = args

    @abstractmethod
    def embed_documents(self, texts):
        pass

    @abstractmethod
    def embed_query(self, query):
        pass


class OllamaLLM(AbstractLLM):
    def __init__(self, args: RAGArgs):
        super().__init__(args)

    def ask(self, prompt: dict) -> str:
        url = f'{self.args.url}/api/generate'

        data = {
            "model": self.args.model,
            "prompt": prompt.get("text", ""),
            "stream": self.args.stream,
            "options": {
                "temperature": self.args.temperature,
                "seed": self.args.seed
            }
        }

        if "images" in prompt.keys():
            data["images"] = prompt["images"]

        response = requests.post(url, json=data)

        if response.status_code == 200:
            return response.json()['response']
        else:
            raise Exception(
                f"Error in Ollama response: {response.status_code} - {response.text}")


class OllamaEmbedder(AbstractEmbedder):
    def __init__(self, args: RAGArgs):
        super().__init__(args)

    def embed_documents(self, texts):
        return [self.create_embedding(text) for text in texts]

    def embed_query(self, query):
        return self.create_embedding(query)

    def create_embedding(self, text: str):
        url = f'{self.args.url_embed}/api/embed'

        data = {
            "model": self.args.model_embbedding,
            "input": text,
            "stream": False
        }

        response = requests.post(url, json=data)

        if response.status_code == 200:
            embedding = response.json()['embeddings']
            return np.array(embedding, dtype=np.float32).flatten()
        else:
            raise Exception(
                f"Error en la solicitud a Ollama: {response.status_code} - {response.text}")


class OpenAILLM(AbstractLLM):
    def __init__(self, args: RAGArgs):
        super().__init__(args)

    def ask(self, prompt: dict) -> str:
        url = f'{self.args.url}/v1/completions'

        data = {
            "model": self.args.model,
            "prompt": prompt.get("text", ""),
            "temperature": self.args.temperature,
            "n": 1,  # Number of responses to generate
            "stream": self.args.stream
        }

        if "images" in prompt.keys():  # TODO: check
            data["images"] = prompt["images"]

        response = requests.post(url, json=data)

        if response.status_code == 200:
            return response.json()['choices'][0]['text'].strip()
        else:
            raise Exception(
                f"Error in Ollama embedding request: {response.status_code} - {response.text}")


class OpenAIEmbedder(AbstractEmbedder):
    def __init__(self, args: RAGArgs):
        super().__init__(args)

    def embed_documents(self, texts):
        return [self.create_embedding(text) for text in texts]

    def embed_query(self, query):
        return self.create_embedding(query)

    def create_embedding(self, text: str):
        url = f'{self.args.url_embed}/v1/embeddings'

        data = {
            "model": self.args.model_embbedding,
            "input": text
        }

        response = requests.post(url, json=data)

        if response.status_code == 200:
            embedding = response.json()['data'][0]['embedding']
            return np.array(embedding, dtype=np.float32).flatten()
        else:
            raise Exception(
                f"Error in Ollama embedding request: {response.status_code} - {response.text}")


class GeminiLLM(AbstractLLM):
    """LLM implementation using the Google Gemini API."""

    def __init__(self, args: RAGArgs):
        super().__init__(args)
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set.")
        genai.configure(api_key=api_key)

        generation_config = {}
        generation_config["temperature"] = args.temperature
        try:
            self.model = genai.GenerativeModel(
                args.model,
                generation_config=genai.types.GenerationConfig(
                    **generation_config) if generation_config else None
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize Gemini GenerativeModel: {e}")

    def ask(self, prompt: dict) -> str:
        """
        Sends a prompt to the Gemini model and returns the generated text.
        """
        try:
            response = self.model.generate_content(prompt.get("text", ""))
            if "images" in prompt.keys():
                print("Gemini currently does not support images", file=sys.stderr)

            if response.parts:
                return response.text
            else:
                block_reason = getattr(
                    getattr(response, 'prompt_feedback', None), 'block_reason', 'N/A')
                safety_ratings = getattr(
                    getattr(response, 'prompt_feedback', None), 'safety_ratings', [])
                print(
                    f"Warning: Gemini returned no content. Block Reason: {block_reason}, Safety Ratings: {safety_ratings}")
                return ""
        except google_exceptions.GoogleAPIError as e:
            raise RuntimeError(f"Gemini API error during generation: {e}")
        except Exception as e:
            raise RuntimeError(
                f"An unexpected error occurred calling Gemini: {e}")


class GeminiEmbedder(AbstractEmbedder):
    """Embedding implementation using the Google Gemini API."""

    def __init__(self, args: RAGArgs):
        super().__init__(args)
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set.")
        genai.configure(api_key=api_key)

        self.embedding_model_name = args.model_embbedding

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Generates embeddings for a list of documents using the Gemini API.
        Handles batching automatically if the list is large.
        """
        if not texts:
            return []
        try:
            result = genai.embed_content(
                model=self.embedding_model_name,
                content=texts,
                task_type="RETRIEVAL_DOCUMENT"
            )
            return result['embedding']
        except google_exceptions.GoogleAPIError as e:
            raise RuntimeError(
                f"Gemini API error during document embedding: {e}")
        except Exception as e:
            raise RuntimeError(
                f"An unexpected error occurred during document embedding: {e}")

    def embed_query(self, query: str) -> list[float]:
        """
        Generates an embedding for a single query string using the Gemini API.
        """
        if not query:
            print("Warning: embedding empty query string.")
            return []

        try:
            result = genai.embed_content(
                model=self.embedding_model_name,
                content=query,
                task_type="RETRIEVAL_QUERY"
            )
            return result['embedding']
        except google_exceptions.GoogleAPIError as e:
            raise RuntimeError(f"Gemini API error during query embedding: {e}")
        except Exception as e:
            raise RuntimeError(
                f"An unexpected error occurred during query embedding: {e}")
