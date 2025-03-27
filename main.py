import pandas as pd
import json
from rag_config import RAGArgs
import sys
from rag import *
from langchain_community.document_loaders import PyPDFLoader
import time


def get_num_pages_words(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    num_pages = len(documents)
    num_words = 0
    for doc in documents:
        num_words += len(doc.page_content.replace('\n',
                         ' ').split(' '))

    return num_pages, num_words


def create_df(args: RAGArgs, num_pages: int, num_words: int):
    df = pd.DataFrame()
    for key, val in vars(args).items():
        df[key] = [val]
    df['pdf_path'] = ''
    df['num_pages'] = num_pages
    df['num_words'] = num_words
    df['embedding_time(s)'] = np.nan
    df['promt_time(s)'] = np.nan
    df['promt_len'] = np.nan
    df['response_len'] = np.nan

    return df


def main(args):
    embedder = None
    llm = None
    if args.open_ai_api:
        embedder = OpenAIEmbedder(args)
        llm = OpenAILLM(args)
    else:
        embedder = OllamaEmbedder(args)
        llm = OllamaLLM(args)

    rag_sys = None
    if args.rag_type == 'naive':
        rag_sys = NaiveRag(args, embedder, llm)
    else:
        raise ValueError('RAG type unknown')

    pdf_path = './data_test/Guion P1a LocalGreedy QKP MHs 2023-24.pdf'
    # pdf_path = './data_test/Presentación_P1.pdf'
    num_pages, num_words = get_num_pages_words(pdf_path)
    df = create_df(args, num_pages, num_words)
    df['pdf_path'] = pdf_path

    init = time.time()
    rag_sys.store_pdf(pdf_path)
    end = time.time()

    df['embedding_time(s)'] = end - init
    df['promt_time(s)'] = np.nan
    df['promt_len'] = np.nan
    df['response_len'] = np.nan

    prompt = input('Ask something: ')
    response_df = df.copy()

    while prompt != '':
        init = time.time()
        response, metadata = rag_sys.ask_llm(prompt)
        end = time.time()
        response_df['promt_time(s)'] = end - init
        response_df['promt_len'] = len(prompt.split(' '))
        response_df['response_len'] = len(response.split(' '))

        df = pd.concat([df, response_df], ignore_index=True)

        print(response)
        print('-' * 80)
        prompt = input('Ask something: ')

    df_csv = pd.read_csv('./data_test/datos_tiempos_ollama.csv')
    df = pd.concat([df_csv, df], ignore_index=True)
    df.to_csv('./data_test/datos_tiempos_ollama.csv', index=False)


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
    print('-' * 80)

    main(args)
