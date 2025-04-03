import pandas as pd
import json
from rag_config import RAGArgs
import sys
from rag import *
from langchain_community.document_loaders import PyPDFLoader
import time
from evaluation import *
import ast


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

    df_csv = pd.read_csv('./data_test/datos_tiempos_vllm.csv')
    df_times = pd.concat([df_csv, df_times], ignore_index=True)
    df_times.to_csv('./data_test/datos_tiempos_vllm.csv', index=False)


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

    # pdf_path = './data_test/Guion P1a LocalGreedy QKP MHs 2023-24.pdf'
    pdf_path = './data_test/Presentación_P1.pdf'
    num_pages, num_words = get_num_pages_words(pdf_path)
    df_times = create_df(args, num_pages, num_words)
    df_times['pdf_path'] = pdf_path

    init = time.time()
    _, pdf_cached = rag_sys.store_pdf(pdf_path)
    end = time.time()

    if not pdf_cached:
        df_times['embedding_time(s)'] = end - init

    df_times['promt_time(s)'] = np.nan
    df_times['promt_len'] = np.nan
    df_times['response_len'] = np.nan

    if not True:  # interactive
        interactive_session(rag_sys, df_times)
    else:
        # df_labels = pd.read_csv('./data_test/preguntas_mh.csv')
        df_labels = pd.read_csv('./data_test/preguntas_vc.csv')
        batch_eval(rag_sys, df_times, df_labels)


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
