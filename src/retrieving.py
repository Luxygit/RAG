"""tokenising the query, compare against our index and calc relevance"""


import json
import math
from typing import Any
from pathlib import Path
from src.indexing import tokenise
from src.validating import MinimalSource, MinimalSearchResults


def load_index(file_path: Path) -> dict[str, Any]:
    """opening our content index file back to memory"""
    with open(file_path, "r", encoding="utf-8") as f:
        res = json.load(f)
    return res


def search_index(query: str, index_data: dict[str, Any], k: int) -> list[Any]:
    """bm25 scoring engine"""
    # turning the query into a clean string of words
    tk_query = tokenise(query)
    doc_lengths = index_data["doc_lengths"]
    avgdl = index_data["avgdl"]
    doc_freqs = index_data["doc_freqs"]
    term_freqs = index_data["term_freqs"]
    chunks = index_data["chunks"]
    N = len(chunks)
    # control term freq saturation, so words dont scale infinitely
    k1 = 1.5
    # controls how we penalise long doc lengths
    b = 0.75
    # scores include {chunk_id: relevancefloat}
    scores: dict[int, float] = {}
    for w in tk_query:
        if w not in doc_freqs:
            continue
        # inverse document frequency
        idf = math.log(((N - doc_freqs[w] + 0.5) / (doc_freqs[w] + 0.5)) + 1)
        for chunk_id_str, tf in term_freqs[w].items():
            length_penalty = 1 - b + b * (doc_lengths[chunk_id_str] / avgdl)
            tf_score = (tf * (k1 + 1)) / (tf + k1 * length_penalty)
            scores[int(chunk_id_str)] = scores.get(
                    int(chunk_id_str), 0.0) + idf * tf_score
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    # {chunk_id: score}
    top_k_ids = ranked[:k]
    results = []
    # from the top_k_ids tuple we get the first item (ID) and use it.
    for t in top_k_ids:
        chunk_index = t[0]
        results.append(chunks[chunk_index])
    return results


def ranking_chunks(sources: list[MinimalSource]) -> MinimalSearchResults:
    res = {
            "question_id": None,
            "question": None,
            "retrieved_sources": sources,
            }
    return res
