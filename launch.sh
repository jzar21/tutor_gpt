#!/bin/bash

export WEBUI_URL="http://localhost:3000"
export THREAD_POOL_SIZE=0 # DEFAULT
export ENV="prod" # dev or prod
export WEBUI_NAME="Universidad de Granada"
export PORT=8080
export DATA_DIR="./data" # DEFAULT
export OLLAMA_BASE_URL="http://localhost:5000" # OUR RAG
export ENABLE_CODE_EXECUTION="false"
export ENABLE_CODE_INTERPRETER="false"
export RAG_EMBEDDING_ENGINE="ollama"
export RAG_EMBEDDING_MODEL="nomic-embed-text"
export RAG_TOP_K=5
export RAG_ALLOWED_FILE_EXTENSIONS=["pdf,docx,txt,md"]

open-webui serve