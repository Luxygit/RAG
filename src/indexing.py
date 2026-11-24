"""indexing the raw data into accesible retrievable number ids"""


import json
import re
from tqdm import tqdm
from pathlib import Path
from src.validating import MinimalSource


STOPWORDS: frozenset[str] = frozenset({
    "a", "an", "and", "are", "as", "at", "by", "can", "do",
    "for", "from", "how", "i", "if", "in", "into", "is", "it", "of",
    "on", "or", "should", "that", "the", "to", "use", "using", "vllm",
    "what", "when", "where", "which", "with", "you", "your",
    })


def tokenise(text: str) -> list[str]:
    """turning text into lowercase alphannum words"""
    raw_words = re.findall(r"[a-zA-Z0-9]+", text)
    tokens: list[str] = []
    for word in raw_words:
        lower_word = word.lower()
        if lower_word in STOPWORDS or len(lower_word) < 2:
            continue
        tokens.append(lower_word)
        split_camel = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", word)
        split_camel = re.sub(
                r"([A-Z]+)([A-Z][a-z])", r"\1 \2", split_camel)
        sub_words = split_camel.lower().split()
        if len(sub_words) > 1:
            tokens.extend(sub_words)
    return tokens


def build_and_save_index(chunks: list[MinimalSource], save_dir: Path) -> bool:
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
    for chunk_id, chunk in tqdm(
            enumerate(chunks),
            total=len(chunks),
            desc="BM25 Indexing"):
        try:
            with open(chunk.file_path, "r", encoding="utf-8") as f:
                content = f.read()
            sliced = (content[
                chunk.first_character_index:chunk.last_character_index])
            tokens_list = tokenise(sliced)
            parts = Path(chunk.file_path).parts
            rel_parts = (
                    parts[parts.index("raw") + 2:]
                    if "raw" in parts
                    else parts[-3:]
                    )
            path_tokens = tokenise(" ".join(rel_parts))
            tokens_list.extend(path_tokens * 2)
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
        except Exception as e:
            print(e)
            continue
    avgdl = 0.0
    if len(chunks):
        avgdl = total_tokens / len(chunks)
    # serialising the pydantic chunks object so it can be output to JSON
    index_data = {
            "chunks": [chunk.model_dump() for chunk in chunks],
            "doc_lengths": doc_lengths,
            "avgdl": avgdl,
            "doc_freqs": doc_freqs,
            "term_freqs": term_freqs,
            }
    try:
        save_dir.mkdir(parents=True, exist_ok=True)
        with open(save_dir / "bm25_index.json", "w", encoding="utf-8") as fj:
            json.dump(index_data, fj)
        return True
    except Exception as e:
        print(f"Error saving index: {e}")
        return False
