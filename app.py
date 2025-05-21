from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from rag_config import RAGArgs
import httpx
from rag import *
import json
import argparse
import time
from contextlib import asynccontextmanager
import uvicorn
import sys
from typing import Any, Dict, Optional, List


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


@app.get("/api/version")
async def get_version():
    return {
        "version": "0.5.1"
    }


@app.get("/api/ps")
async def get_models():
    async with httpx.AsyncClient() as client:
        response = await client.get("http://localhost:11434/api/ps")
        return response.json()


class WrapperShowInfo(BaseModel):
    model: str
    verbose: Optional[bool] = None


@app.post("/api/show")
async def show_info(request: WrapperShowInfo):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:11434/api/show",
            headers={"Content-Type": "application/json"},
            content=request.model_dump_json()
        )
        return response.json()


def add_gemini_models(data: dict):
    extra_model = {
        "name": "gemini-1.5-flash:latest",
        "model": "gemini-1.5-flash:latest",
        "size": 0,
        "digest": "manual",
    }

    if "models" in data:
        data["models"].append(extra_model)
    else:
        data = {"models": [extra_model]}

    return data


@app.get("/api/tags")
async def get_tags():
    async with httpx.AsyncClient() as client:
        response = await client.get("http://localhost:11434/api/tags")
        data = response.json()
        data = add_gemini_models(data)
        return data


class WrapperMessage(BaseModel):
    role: str = Field(...,
                      description="Role of the message (system, user, assistant, tool)")
    content: str = Field(None, description="Content of the message")
    images: Optional[List[str]] = Field(
        None, description="List of base64-encoded images")
    tool_calls: Optional[List[Dict[str, Any]]] = Field(
        None, description="List of tools the model wants to use")


class WrapperChatRequest(BaseModel):
    model: str = Field(..., description="The model name")
    messages: List[WrapperMessage] = Field(...,
                                           description="List of chat messages")

    tools: Optional[List[Dict[str, Any]]] = Field(
        None, description="List of tools in JSON for the model to use")

    format: Optional[str] = Field(
        None, description="The format to return a response in (json or a JSON schema)")

    options: Optional[Dict[str, Any]] = None
    stream: Optional[bool] = False
    keep_alive: Optional[str] = "5m"


class WrapperChatResponse(BaseModel):
    model: str
    created_at: str
    message: WrapperMessage
    done: bool
    total_duration: Optional[int] = None
    load_duration: Optional[int] = None
    prompt_eval_count: Optional[int] = None
    prompt_eval_duration: Optional[int] = None
    eval_count: Optional[int] = None
    eval_duration: Optional[int] = None
    done_reason: Optional[str] = None


def setup_diferent_model(request: WrapperChatRequest):
    if 'gemini' in request.model:
        request.model = request.model.split(':')[0]

    if request.model != app.state.rag_system.llm.args.model:
        app.state.rag_system.args.model = request.model

    if 'gemini' in request.model and type(app.state.rag_system.llm) != GeminiLLM:
        app.state.rag_system.llm = GeminiLLM(app.state.rag_system.args)


async def call_ollama_chat(request: WrapperChatRequest):
    """Calls your RAG system to generate a chat response."""
    last_user_message = None
    last_user_image = None

    for message in reversed(request.messages):
        if message.role == "user":
            last_user_message = message.content
            last_user_image = message.images
            break

    if not last_user_message:
        raise HTTPException(
            status_code=400, detail="No user message found in the chat history.")

    setup_diferent_model(request)

    start_time = time.time()
    response, metadata = app.state.rag_system.ask_llm(
        last_user_message, last_user_image
    )
    end_time = time.time()
    total_duration = int((end_time - start_time) * 10**6)

    prompt_tokens = metadata.get("prompt_tokens", 0)
    response_tokens = metadata.get("response_tokens", 0)
    prompt_eval_duration = metadata.get("prompt_eval_duration", 0.0)
    eval_duration = metadata.get("eval_duration", 0.0)

    return {
        "model": request.model,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S.%fZ", time.gmtime()),
        "message": {"role": "assistant", "content": response.strip(), "images": None, "tool_calls": None},
        "done": True,
        "total_duration": total_duration,
        "load_duration": 0.0,  # Placeholde
        "prompt_eval_count": prompt_tokens,
        "prompt_eval_duration": prompt_eval_duration,
        "eval_count": response_tokens,
        "eval_duration": eval_duration,
        "done_reason": "stop",
    }


@app.post("/api/chat")
async def chat_endpoint(request: WrapperChatRequest):
    """
    Generates the next message in a chat using your RAG system.
    Currently only supports non-streaming responses.
    """
    ollama_response = await call_ollama_chat(request)
    return WrapperChatResponse(**ollama_response)


class WrapperEmbedRequest(BaseModel):
    model: str
    input: List[str] | str
    truncate: Optional[bool] = True
    keep_alive: Optional[str] = '5m'


class WrapperEmbedResponse(BaseModel):
    model: str
    embeddings: List[float] | List[List[List[float]]] | List[List[float]]
    total_duration: int
    load_duration: int
    prompt_eval_count: int


@app.post("/api/embed")
async def generate_embed(request: WrapperEmbedRequest):
    start_time = time.time()
    response = app.state.rag_system.embed(request.input[0])
    end_time = time.time()
    total_duration = int((end_time - start_time) * 10**6)

    return WrapperEmbedResponse(**{
        "model": request.model,
        "embeddings": [response],
        "total_duration": total_duration,
        "load_duration": 0.0,  # Placeholde
        "prompt_eval_count": 0,
    })


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5001)
