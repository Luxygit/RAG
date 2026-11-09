""""""


import json
from pathlib import Path
from src.validation import MinimalSource


def tokenise(text: str) -> list[str]:
    """turning text into lowercase alphannum words"""
    clean_chars = []
    for c in text:
        if c.isalnum():
            clean_chars.append(c.lower())
        else:
            clean_chars.append(" ")
    res = "".join(clean_chars)
    return res.split()


def build_and_save_index(chunks: list[MinimalSource], save_dir: Path) -> None:
    """
    reads file slices, tokenises them with some maths into a JSON
    """
    # chunk_id:totalwordcount penalises matches in huge chunks
    doc_lengths: dict[int, int] = {}
    # word:numberofchunkscontainingit so that rare words score higher
    doc_freqs: dict[str, int] = {}
    # word:{chunk_id:count} {"server":{chunk0:2, chunk5: 1}}
    term_freqs: dict[str, dict[int, int]] = {}
    # sum of all words, to calc average document length
    total_tokens = 0
    for chunk_id, chunk in enumerate(chunks):
        with open(chunk.file_path, "r", encoding="utf-8") as f:
            content = f.read()
            sliced = content[chunk.first_character_index:chunk.last_character_index]
            tokens_list = tokenise(sliced)
            doc_lengths[chunk_id] = len(tokens_list)
            total_tokens += len(tokens_list)
            chunk_freq: dict[str, int] = {}
            for t in tokens_list:
                chunk_freq[t] = chunk_freq.get(t, 0) + 1
            for w, count in chunk_freq.items():
                if w not in term_freqs:
                    term_freqs[w] = {}
                    doc_freqs[w] = 0
                term_freqs[w][chunk_id] = count
                doc_freqs[w] += 1
    avgdl = 0.0
    if len(chunks):
        avgdl = total_tokens / len(chunks)
    # serialising the pydantic chunks object so it can be output to JSON
    index_data = {
            "chunks": [chunk.model_dump() for chunk in chunks],
            "doc_lenghts": doc_lenghts,
            "avgdl": avgdl,
            "doc_freqs": doc_freqs,
            "term_freqs": term_freqs,
            }
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / "bm25_index.json", "w", encoding="utf-8") as fj:
        json.dump(index_data, fj)

