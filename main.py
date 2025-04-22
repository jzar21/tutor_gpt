import pandas as pd
import json
from rag_config import RAGArgs
import sys
from rag import *
from langchain_community.document_loaders import PyPDFLoader
import time
from evaluation import *
import ast
import argparse


def str2bool(value):
    if value.lower() in ['true', '1', 't', 'y', 'yes']:
        return True
    elif value.lower() in ['false', '0', 'f', 'n', 'no']:
        return False
    else:
        raise ValueError('Valor no válido para un booleano')


def get_num_pages_words(pdf_path):
    total_pages = 0
    total_words = 0
    for pdf in pdf_path:
        if pdf.endswith('pdf'):
            loader = PyPDFLoader(pdf)
            documents = loader.load()
            num_pages = len(documents)
            num_words = 0
            for doc in documents:
                num_words += len(doc.page_content.replace('\n',
                                                          ' ').split(' '))

            total_pages += num_pages
            total_words += num_words

    return total_pages, total_words


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


def interactive_session(rag: Rag, df_times: pd.DataFrame):
    prompt = input('Ask something: ')
    response_df = df_times.copy()

    while prompt != '':
        init = time.time()
        response, _ = rag.ask_llm(prompt)
        end = time.time()

        response_df['promt_time(s)'] = end - init
        response_df['promt_len'] = len(prompt.split(' '))
        response_df['response_len'] = len(response.split(' '))

        df_times = pd.concat([df_times, response_df], ignore_index=True)

        print(response)
        print('-' * 80)
        prompt = input('Ask something: ')

    df_csv = pd.read_csv('./data_test/datos_tiempos_ollama.csv')
    df_times = pd.concat([df_csv, df_times], ignore_index=True)
    df_times.to_csv('./data_test/datos_tiempos_ollama.csv', index=False)


def batch_eval(rag: Rag, df_times: pd.DataFrame, df_labels: pd.DataFrame):
    response_df = df_times.copy()

    sources = []
    pages = []

    ground_truth = {}
    for file in df_labels['path'].unique().tolist():
        ground_truth[file] = []

    for _, row in df_labels.iterrows():
        ground_truth[row['path']].extend(ast.literal_eval(row['paginas']))

    for prompt in df_labels['pregunta']:
        init = time.time()
        response, metadata = rag.ask_llm(prompt)
        end = time.time()

        sources.append(metadata['source'])
        pages.append(list(map(int, metadata['page_label'])))

        response_df['promt_time(s)'] = end - init
        response_df['promt_len'] = len(prompt.split(' '))
        response_df['response_len'] = len(response.split(' '))

        df_times = pd.concat([df_times, response_df], ignore_index=True)

        print(response)
        print('-' * 80)

    responses_info = {}

    for files in sources:
        for file in files:
            responses_info[file] = []

    for file, page in zip(sources, pages):
        for file in files:
            responses_info[file].extend(page)

    precision = precision_k(responses_info, ground_truth)
    recall = recall_k(responses_info, ground_truth)
    f1 = f1_score_k(responses_info, ground_truth)
    mrr = mean_reciprocal_rank(responses_info, ground_truth)
    acc = accuracy(responses_info, ground_truth)
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"MRR: {mrr:.4f}")
    print(f"Accuracy: {acc:.4f}")

    df_csv = pd.read_csv('./data_test/datos_tiempos_ollama.csv')
    df_times = pd.concat([df_csv, df_times], ignore_index=True)
    df_times.to_csv('./data_test/datos_tiempos_ollama.csv', index=False)


def main(args, cmd_args):
    embedder = None
    llm = None
    if args.open_ai_api:
        embedder = OpenAIEmbedder(args)
        llm = OpenAILLM(args)
    else:
        if 'gemini' in args.model:
            llm = GeminiLLM(args)
            embedder = OllamaEmbedder(args)
            # embedder = GeminiEmbedder(args)
        else:
            embedder = OllamaEmbedder(args)
            llm = OllamaLLM(args)

    rag_sys = None
    if args.rag_type == 'naive':
        rag_sys = NaiveRag(args, embedder, llm)
    else:
        raise ValueError('RAG type unknown')

    num_pages, num_words = get_num_pages_words(cmd_args.files)
    df_times = create_df(args, num_pages, num_words)
    df_times['pdf_path'] = "".join([path for path in cmd_args.files])

    init = time.time()
    _, pdf_cached = rag_sys.store_docs(cmd_args.files)
    end = time.time()

    if not pdf_cached:
        df_times['embedding_time(s)'] = end - init

    df_times['promt_time(s)'] = np.nan
    df_times['promt_len'] = np.nan
    df_times['response_len'] = np.nan

    if cmd_args.interactive:
        interactive_session(rag_sys, df_times)
    else:
        df_labels = pd.read_csv(cmd_args.batch_questions)
        batch_eval(rag_sys, df_times, df_labels)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config_path',
        type=str,
        help='config file path',
        required=True
    )
    parser.add_argument(
        '--interactive',
        type=str2bool,
        help='Interactive chat',
        default=False
    )
    parser.add_argument(
        '--files',
        type=str,
        help='Files to use as Knowledge DB',
        required=True,
        nargs='+'
    )
    parser.add_argument(
        '--batch_questions',
        type=str,
        help='CVS with the questions and answers. Only used is Interactive is False',
        required=False,
    )
    cmd_args = parser.parse_args()

    with open(cmd_args.config_path, 'r') as f:
        data = json.load(f)

    args = RAGArgs(**data)
    print('Configuration')
    for key, val in vars(args).items():
        print(f'{key}: {val}')
    print('-' * 80)

    main(args, cmd_args)
