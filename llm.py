from abc import ABC, abstractmethod
from rag_config import RAGArgs
from langchain.embeddings.base import Embeddings
import numpy as np
from rag_config import RAGArgs
import requests


class AbstractLLM(ABC):
    def __init__(self, args: RAGArgs):
        super().__init__()
        self.args = args

    @abstractmethod
    def ask(self, promt: str) -> str:
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

    def ask(self, prompt: str) -> str:
        url = f'{self.args.url}/api/generate'

        data = {
            "model": self.args.model,
            "prompt": prompt,
            "stream": self.args.stream,
            "options": {
                "temperature": self.args.temperature,
                "seed": self.args.seed
            }
        }

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
        url = f'{self.args.url}/api/embed'

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

    def ask(self, prompt: str) -> str:
        url = f'{self.args.url}/v1/completions'

        data = {
            "model": self.args.model,
            "prompt": prompt,
            "temperature": self.args.temperature,
            "n": 1,  # Number of responses to generate
            "stream": self.args.stream
        }

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
        url = f'{self.args.url}/v1/embeddings'

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
