from dataclasses import dataclass


@dataclass
class RAGArgs:
    model: str = 'llama3.2'
    model_embbedding: str = 'llama3.2'
    url: str = 'http://localhost:11434'
    chunk_size: int = 1000
    chunk_overlap: int = 100
    temperature: float = 0
    top_k: int = 5
    fetch_k: int = 5
    stream: bool = False
    seed: int = 42
    rag_type: str = 'naive'
