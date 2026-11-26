"""tokenising the query, compare against our index and calc relevance"""


import json
import math
from typing import Any
from pathlib import Path
from tqdm import tqdm
from src.indexing import tokenise
from src.validating import MinimalSource, MinimalSearchResults


def load_index(file_path: Path) -> dict[str, Any]:
    """opening our content index file back to memory"""
    if not isinstance(file_path, Path) or not file_path.is_file():
        return {}
    try:
        with tqdm(total=1, desc="Loading BM25 index") as pbar:
            with open(file_path, "r", encoding="utf-8") as f:
                res: dict[str, Any] = json.load(f)
            pbar.update(1)
        if not isinstance(res, dict):
            return {}
        return res
    except Exception as e:
        print(e)
        return {}


def search_index(query: str, index_data: dict[str, Any], k: int
                 ) -> list[dict[str, Any]]:
    """bm25 scoring engine"""
    if (
            not isinstance(query, str)
            or not isinstance(index_data, dict)
            or isinstance(k, bool)
            or not isinstance(k, int)
            or not index_data
            or k <= 0
    ):
        return []
    try:
        # turning the query into a clean string of words like those in index
        tk_query = tokenise(query)
        # unpacking the lookup tables from the index
        doc_lengths = index_data["doc_lengths"]
        avgdl = index_data["avgdl"]
        doc_freqs = index_data["doc_freqs"]
        term_freqs = index_data["term_freqs"]
        chunks = index_data["chunks"]
        # total number of indexed chunks
        N = len(chunks)
        if not chunks or avgdl <= 0:
            return []
        # control term freq saturation, so words dont scale infinitely
        k1 = 1.5
        # controls how we penalise docs longer than average
        b = 0.75
        # scoring each matching chunk {chunk_id: floatscore}
        scores: dict[int, float] = {}
        for w in tk_query:
            # skip query words that dont appear in the index
            if w not in doc_freqs:
                continue
            # rare words accross the dataset get higher IDF weight
            idf = math.log(
                    ((N - doc_freqs[w] + 0.5) / (doc_freqs[w] + 0.5)) + 1)
            for chunk_id_str, tf in term_freqs[w].items():
                length_penalty = (
                        1 - b + b * (doc_lengths[chunk_id_str] / avgdl))
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
    except Exception as e:
        print(f"Search error: {e}")
        return []


def ranking_chunks(sources: list[MinimalSource]) -> MinimalSearchResults:
    """wraps an ordered list of MinimalSource objects in search schema"""
    return MinimalSearchResults(
           question_id="",
           question="",
           retrieved_sources=sources,
           )
