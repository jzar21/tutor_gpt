from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from rag_config import RAGArgs
from rag import *
import json
import argparse
import time
from contextlib import asynccontextmanager
import uvicorn
import sys
from typing import Optional, List


class QueryRequest(BaseModel):
    query: str
    model: str


class QueryResponse(BaseModel):
    response: str
    prompt_time: float
    prompt_length: int
    response_length: int


def load_config(config_path: str) -> RAGArgs:
    """Loads the RAG configuration from a JSON file."""
    try:
        with open(config_path, 'r') as f:
            data = json.load(f)

        args = RAGArgs(**data)
        print('Configuration:')
        for key, val in vars(args).items():
            print(f'{key}: {val}')
        print('-' * 80)
        return args
    except Exception as e:
        print(
            f"Error: {e}", file=sys.stderr
        )
        raise


def create_rag_system(args: RAGArgs, files: Optional[List[str]] = None) -> Rag:
    """Initializes and returns the RAG system based on the provided configuration."""
    embedder = None
    llm = None

    if args.open_ai_api:
        embedder = OpenAIEmbedder(args)
        llm = OpenAILLM(args)
    elif 'gemini' in args.model:
        llm = GeminiLLM(args)
        embedder = OllamaEmbedder(args)
    else:
        embedder = OllamaEmbedder(args)
        llm = OllamaLLM(args)

    rag_system = NaiveRag(args, embedder, llm)
    if files:
        rag_system.store_docs(files)
    else:
        print("Warning: No files provided for the knowledge base.", file=sys.stderr)

    return rag_system


def get_rag_system(config_path: str, files: Optional[List[str]] = None) -> Rag:
    """Dependency function to create and provide the RAG system."""
    config = load_config(config_path)
    rag_system = create_rag_system(config, files)

    return rag_system


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manages the application lifecycle events."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config_path',
        type=str,
        help='config file path',
        required=True
    )
    parser.add_argument(
        '--files',
        type=str,
        help='Files to use as Knowledge DB (solo para inicialización)',
        required=True,
        nargs='+'
    )
    cmd_args = parser.parse_args()
    app.state.rag_config_path = cmd_args.config_path
    app.state.rag_files = cmd_args.files
    app.state.rag_system = get_rag_system(cmd_args.config_path, cmd_args.files)

    yield
    # Perform any cleanup here if needed


app = FastAPI(lifespan=lifespan)


@app.post("/rag_query", response_model=QueryResponse)
async def rag_query(request: QueryRequest):
    """Endpoint for querying the RAG system."""
    user_question = request.query.strip()

    if not user_question:
        raise HTTPException(
            status_code=400, detail="Query cannot be empty"
        )

    app.state.rag_system.llm.args.model = request.model  # TODO: hacer esto mejor

    start_time = time.time()
    response, _ = app.state.rag_system.ask_llm(user_question)
    end_time = time.time()

    return QueryResponse(
        response=response.strip(),
        prompt_time=end_time - start_time,
        prompt_length=len(user_question.split()),
        response_length=len(response.split())
    )

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5001)
