import numpy as np
import json
from rag_config import RAGArgs
import sys
from rag import *


def main(args):
    embedder = Embedder(args)
    llm = LLM(args)
    rag_sys = None
    if args.rag_type == 'naive':
        rag_sys = NaiveRag(args, embedder, llm)
    else:
        raise ValueError('RAG type unknown')

    rag_sys.store_pdf('./data_test/Guion P1a LocalGreedy QKP MHs 2023-24.pdf')

    prompt = input('Ask something: ')

    while prompt != '':
        response = rag_sys.ask_llm(prompt)
        print(response)
        print('-' * 80)
        prompt = input('Ask something: ')


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
