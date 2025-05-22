from abc import ABC, abstractmethod
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from rag_config import RAGArgs
from llm import *
import os
import hashlib


class Rag(ABC):
    CACHED_PATH = "~/.cache/tutor_gpt/embedings"

    def __init__(self, args: RAGArgs, embbedder: AbstractEmbedder, llm: AbstractLLM):
        super().__init__()
        self.args = args
        self.embbedder = embbedder
        self.llm = llm
        self.retriever = None
        self.vector_db = None

    @abstractmethod
    def retrieve(self, prompt: str) -> str:
        pass

    @abstractmethod
    def ask_llm(self, prompt: str) -> str:
        pass

    def embed(self, text: str) -> np.ndarray:
        return np.array(self.embbedder.embed_query(text), dtype=np.float32)

    def _is_cached(self, paths: list[str]) -> bool:
        full_name = "".join([path for path in paths])
        store_folder = os.path.expanduser(
            f"{Rag.CACHED_PATH}/{self.args.model_embbedding}"
        )
        full_name += f"{self.args.chunk_size}-{self.args.chunk_overlap}"
        hash_name = hashlib.md5(full_name.encode('utf-8')).hexdigest()
        is_cached = os.path.exists(
            store_folder + '/' + hash_name + '.faiss'
        )

        if is_cached:
            self.vector_db = FAISS.load_local(
                store_folder,
                self.embbedder,
                index_name=hash_name,
                allow_dangerous_deserialization=True
            )
            self.retriever = self.vector_db.as_retriever(
                search_tipe="mmr",
                search_kwargs={'k': self.args.top_k,
                               'fetch_k': self.args.fetch_k}
            )

            return True

        return False

    def get_lists_docs(self, paths: list[str]) -> list[Document]:
        docs = []

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.args.chunk_size,
            chunk_overlap=self.args.chunk_overlap,
            length_function=len
        )
        for file in paths:
            if file.endswith('.pdf'):
                docs.extend(
                    text_splitter.split_documents(
                        PyPDFLoader(file).load()
                    )
                )
            if file.endswith('.txt') or file.endswith('.md'):
                docs.extend(
                    text_splitter.split_documents(
                        TextLoader(file).load()
                    )
                )
        return docs

    def store_docs(self, paths: list[str], save: bool = True) -> tuple[FAISS, bool]:
        if self._is_cached(paths):
            return self.vector_db, True

        docs = self.get_lists_docs(paths)

        self.vector_db = FAISS.from_documents(docs, self.embbedder)
        self.retriever = self.vector_db.as_retriever(
            search_tipe=self.args.rag_search_tipe,
            search_kwargs={'k': self.args.top_k,
                           'fetch_k': self.args.fetch_k}
        )

        if save:
            self._store_embeading(paths)

        return self.vector_db, False

    def store_pdfs(self, paths: list[str], save: bool = True) -> tuple[FAISS, bool]:
        if self._is_cached(paths):
            return self.vector_db, True

        docs = []
        for path in paths:
            loader = PyPDFLoader(path)
            docs.extend(loader.load())

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.args.chunk_size,
            chunk_overlap=self.args.chunk_overlap,
            length_function=len
        )

        docs = text_splitter.split_documents(docs)

        self.vector_db = FAISS.from_documents(docs, self.embbedder)
        self.retriever = self.vector_db.as_retriever(
            search_tipe="mmr",
            search_kwargs={'k': self.args.top_k,
                           'fetch_k': self.args.fetch_k}
        )

        if save:
            self._store_embeading(paths)

        return self.vector_db, False

    def _store_embeading(self, paths: list[str]):
        full_name = "".join([path for path in paths])
        full_name += f"{self.args.chunk_size}-{self.args.chunk_overlap}"
        store_folder = os.path.expanduser(
            f"{Rag.CACHED_PATH}/{self.args.model_embbedding}"
        )
        hash_name = hashlib.md5(full_name.encode('utf-8')).hexdigest()
        self.vector_db.save_local(
            folder_path=store_folder,
            index_name=hash_name
        )

    def store_string(self, texts: list[str], save: bool = True) -> tuple[FAISS, bool]:
        if self._is_cached(texts):
            return self.vector_db, True

        full_text = "".join([text for text in texts])
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.args.chunk_size,
            chunk_overlap=self.args.chunk_overlap,
            length_function=len
        )

        docs = text_splitter.split_text(full_text)

        self.vector_db = FAISS.from_texts(docs, self.embbedder)
        self.retriever = self.vector_db.as_retriever(
            search_tipe="mmr",
            search_kwargs={'k': self.args.top_k,
                           'fetch_k': self.args.fetch_k}
        )

        if save:
            self._store_embeading(texts)

        return self.vector_db, False


class NaiveRag(Rag):
    def __init__(self, args, embbedder, llm):
        super().__init__(args, embbedder, llm)

    def retrieve(self, prompt: str) -> str:
        if self.retriever is None:
            raise ValueError(
                'Retriever is None, need to load some information in order to use the retriever'
            )

        return self.retriever.invoke(prompt)

    def ask_llm(self, prompt: dict) -> str:
        relevant_docs = self.retrieve(prompt.get("text", ""))
        context = "".join([doc.page_content for doc in relevant_docs])
        new_context = f"""
        You are an usefull profesor assistant, you need to answer the questions with the following context.
        <context>
        {context}
        <context>
        Answer the following question. If you don't know the answer, please tell that you don't know and
        response in the same lenguage as the question and give a brief and precise answer: {prompt["text"]}"""

        new_promt = prompt.copy()
        # TODO: Discutir con tutores como procedemos
        new_promt["text"] = new_context  # prompt["text"]

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
                try:
                    retrieved_metadata[k].append(v)
                except:
                    retrieved_metadata[k] = [v]

        return retrieved_metadata
