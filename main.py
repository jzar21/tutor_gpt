import numpy as np
import faiss
import requests
import json
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.base import Embeddings
from rag_config import RAGArgs
import sys


class Embedder(Embeddings):
    def __init__(self, args: RAGArgs):
        self.args = args

    # def __call__(self, *args, **kwds):
    #     return self.create_embedding(args[0])

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


def encode_pdf(path: str, embedder: Embedder, args):
    loader = PyPDFLoader(path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap, length_function=len
    )
    texts = text_splitter.split_documents(documents)

    embeddings_list = []
    for text in texts:
        embedding = embedder.embed_query(text.page_content)
        embeddings_list.append(embedding)

    embeddings_array = np.vstack(embeddings_list).astype(np.float32)

    dim = embeddings_array.shape[1]
    faiss_index = faiss.IndexFlatL2(dim)

    faiss_index.add(embeddings_array)

    docstore = InMemoryDocstore()

    index_to_docstore_id = [str(i) for i in range(len(texts))]
    doc_dict = {
        index_to_docstore_id[i]: texts[i] for i, text in enumerate(texts)}
    docstore.add(doc_dict)

    vectorstore = FAISS(
        embedding_function=embedder,
        index=faiss_index,
        docstore=docstore,
        index_to_docstore_id=index_to_docstore_id
    )

    return vectorstore


def ask_llm(promt: str, args: RAGArgs):
    url = f'{args.url}/api/generate'

    data = {
        "model": args.model,
        "prompt": promt,
        "stream": args.stream,
        "options": {
            "temperature": args.temperature,
            "seed": args.seed
        }
    }

    response = requests.post(url, json=data)

    if response.status_code == 200:
        return response.json()['response']
    else:
        raise Exception(
            f"Error en la solicitud a Ollama: {response.status_code} - {response.text}")


def ask_llm_rag(prompt: str, retriever, args: RAGArgs):
    relevant_docs = retriever.invoke(prompt)
    context = "".join([doc.page_content for doc in relevant_docs])

    new_promt = f"<contexto> {context} <contexto> {prompt}"

    return ask_llm(new_promt, args)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Error: config.json file needed', file=sys.stderr)
        sys.exit(1)

    with open(sys.argv[1], 'r') as f:
        data = json.load(f)

    args = RAGArgs(**data)
    print('Configuration')
    for key, val in vars(args).items():
        print(f'{key}: {val}')

    embedder = Embedder(args)

    vec = encode_pdf(
        './data_test/Guion P1a LocalGreedy QKP MHs 2023-24.pdf', embedder, args)
    retri = vec.as_retriever({'k': args.top_k,
                              'fetch_k': args.fetch_k,
                              })

    promt = input('Pregunta algo: ')

    response = ask_llm_rag(promt, retri, args)
    print(response)
