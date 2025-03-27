from abc import ABC, abstractmethod
import numpy as np
import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.base import Embeddings
from rag_config import RAGArgs
from llm import *


class Rag(ABC):
    def __init__(self, args: RAGArgs, embbedder: AbstractEmbedder, llm: AbstractLLM):
        super().__init__()
        self.args = args
        self.embbedder = embbedder
        self.llm = llm
        self.retriever = None
        self.vector_store = None

    @abstractmethod
    def retrieve(self, prompt: str) -> str:
        pass

    @abstractmethod
    def ask_llm(self, prompt: str) -> str:
        pass

    def embed(self, text: str) -> np.ndarray:
        return np.array(self.embbedder.embed_query(text), dtype=np.float32)

    def store_pdf(self, path: str) -> FAISS:
        loader = PyPDFLoader(path)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.args.chunk_size,
            chunk_overlap=self.args.chunk_overlap,
            length_function=len
        )
        texts = text_splitter.split_documents(documents)

        embeddings_list = []
        for text in texts:
            embedding = self.embbedder.embed_query(text.page_content)
            embeddings_list.append(embedding)

        embeddings_array = np.vstack(embeddings_list).astype(np.float32)

        dim = embeddings_array.shape[1]
        faiss_index = faiss.IndexFlatL2(dim)

        faiss_index.add(embeddings_array)

        docstore = InMemoryDocstore()

        index_to_docstore_id = [str(i) for i in range(len(texts))]
        doc_dict = {index_to_docstore_id[i]: texts[i]
                    for i in range(len(texts))}
        docstore.add(doc_dict)

        vectorstore = FAISS(
            embedding_function=self.embbedder,
            index=faiss_index,
            docstore=docstore,
            index_to_docstore_id=index_to_docstore_id
        )

        if self.vector_store is None:
            self.vector_store = vectorstore
        else:
            raise ValueError(
                'Vectorstore already exists, currently no additions supported')

        self.retriever = vectorstore.as_retriever(search_kwargs={'k': self.args.top_k,
                                                                 'fetch_k': self.args.fetch_k,
                                                                 })

        return self.vector_store

    def embed(self, text: str) -> np.ndarray:
        return np.array(self.embbedder.embed_query(text), dtype=np.float32)


class NaiveRag(Rag):
    def __init__(self, args, embbedder, llm):
        super().__init__(args, embbedder, llm)

    def retrieve(self, prompt: str) -> str:
        if self.retriever is None:
            raise ValueError(
                'Retriever is None, need to load some information in order to use the retriever')
        return self.retriever.invoke(prompt)

    def ask_llm(self, prompt: str) -> str:
        relevant_docs = self.retrieve(prompt)
        context = "".join([doc.page_content for doc in relevant_docs])
        new_promt = f"<contexto> {context} <contexto> {prompt}"
        retrieved_metadata = self.__get_full_metadata(relevant_docs)

        return self.llm.ask(new_promt), retrieved_metadata

    def __get_full_metadata(self, relevant_docs: list):
        if len(relevant_docs) <= 0:
            raise ValueError('No relevant doc found')

        retrieved_metadata = {}
        for k, _ in relevant_docs[0].metadata.items():
            retrieved_metadata[k] = []

        for doc in relevant_docs:
            for k, v in doc.metadata.items():
                retrieved_metadata[k].append(v)

        return retrieved_metadata
