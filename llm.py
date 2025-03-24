from rag_config import RAGArgs
from langchain.embeddings.base import Embeddings
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.base import Embeddings
from rag_config import RAGArgs
import requests


class LLM:
    def __init__(self, args: RAGArgs):
        self.args = args

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


class Embedder(Embeddings):
    def __init__(self, args: RAGArgs):
        self.args = args

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
