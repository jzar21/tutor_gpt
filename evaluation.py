import numpy as np


def precision_k(retrieved_docs: dict, ground_truth: dict) -> float:
    ground_truth_dict = {doc_id: set(pages)
                         for doc_id, pages in ground_truth.items()}

    true_positives = 0
    total_retrieved = 0

    for doc_id, retrieved_pages in retrieved_docs.items():
        if doc_id in ground_truth_dict:
            relevant_pages = ground_truth_dict[doc_id]
            retrieved_set = set(retrieved_pages)

            true_positives += len(retrieved_set & relevant_pages)
            total_retrieved += len(retrieved_set)

    if np.isclose(0, total_retrieved):
        return 0.0

    precision = true_positives / total_retrieved
    return precision


def recall_k(retrieved_docs: dict, ground_truth: dict) -> float:
    ground_truth_dict = {doc_id: set(pages)
                         for doc_id, pages in ground_truth.items()}

    true_positives = 0
    total_relevant = 0

    for doc_id, relevant_pages in ground_truth_dict.items():
        total_relevant += len(relevant_pages)

        if doc_id in retrieved_docs:
            retrieved_set = set(retrieved_docs[doc_id])
            true_positives += len(retrieved_set & relevant_pages)

    if np.isclose(0, total_relevant):
        return 0.0

    recall = true_positives / total_relevant
    return recall


def f1_score_k(retrieved_docs: dict, ground_truth: dict) -> float:
    precision = precision_k(retrieved_docs, ground_truth)
    recall = recall_k(retrieved_docs, ground_truth)

    if np.isclose(0, precision + recall):
        return 0

    f1_score = 2 * precision * recall / (precision + recall)

    return f1_score


def mean_reciprocal_rank(retrieved_docs: dict, ground_truth: dict) -> float:
    mrr_sum = 0.0
    total_queries = 0

    for doc_id, relevant_pages in ground_truth.items():
        if doc_id in retrieved_docs:
            retrieved_set = retrieved_docs[doc_id]
            rank = None
            for i, page in enumerate(retrieved_set):
                if page in relevant_pages:
                    rank = i + 1
                    break

            if rank is not None:
                mrr_sum += 1.0 / rank

            total_queries += 1

    if np.isclose(0, total_queries):
        return 0.0

    mrr = mrr_sum / total_queries
    return mrr


def accuracy(retrieved_docs: dict, ground_truth: dict) -> float:
    correct_retrieved = 0
    total_retrieved = 0

    for doc_id, retrieved_pages in retrieved_docs.items():
        if doc_id in ground_truth:
            relevant_pages = set(ground_truth[doc_id])
            retrieved_set = set(retrieved_pages)

            correct_retrieved += len(retrieved_set & relevant_pages)
            total_retrieved += len(retrieved_set)

    if np.isclose(0, total_retrieved):
        return 0.0

    accuracy_score = correct_retrieved / total_retrieved
    return accuracy_score


if __name__ == '__main__':
    retrieved_docs = {
        "1": [2, 3],  # Document 1 retrieved pages
        "4": [1, 3],  # Document 1 retrieved pages
    }

    ground_truth = {
        "1": [3, 2, 5],   # Document 1 has relevant pages 2 and 5
        "2": [1, 4],      # Document 2 has relevant pages 1 and 4
        "3": [3, 2, 4]    # Document 3 has relevant pages 3 and 4
    }

    precision = precision_k(retrieved_docs, ground_truth)
    recall = recall_k(retrieved_docs, ground_truth)
    f1 = f1_score_k(retrieved_docs, ground_truth)
    mrr = mean_reciprocal_rank(retrieved_docs, ground_truth)
    acc = accuracy(retrieved_docs, ground_truth)
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"MRR: {mrr:.4f}")
    print(f"Accuracy: {acc:.4f}")
