"""indexing the raw data into accesible retrievable number ids"""


import json
import re
from tqdm import tqdm
from pathlib import Path
from src.validating import MinimalSource


STOPWORDS: frozenset[str] = frozenset({
    "a", "an", "and", "are", "as", "at", "by", "can", "do",
    "for", "from", "how", "i", "if", "in", "into", "is", "it", "of",
    "on", "or", "should", "that", "the", "to", "use", "using",
    "what", "when", "where", "which", "with", "you", "your",
    })


def tokenise(text: str) -> list[str]:
    """turning text into lowercase alphannum words"""
    # extracting only groups of letters and numbers
    raw_words = re.findall(r"[a-zA-Z0-9]+", text)
    tokens: list[str] = []
    for word in raw_words:
        lower_word = word.lower()
        # filtering out common words and single chars
        if lower_word in STOPWORDS or len(lower_word) < 2:
            continue
        tokens.append(lower_word)
        # inseting a space before an Uppercase after a lowercase char
        split_camel = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", word)
        # inserting space before last uppercase if followed by a lowercase
        split_camel = re.sub(
                r"([A-Z]+)([A-Z][a-z])", r"\1 \2", split_camel)
        sub_words = split_camel.lower().split()
        # if the word is split into sub_words add each part
        if len(sub_words) > 1:
            tokens.extend(sub_words)
    return tokens


def build_and_save_index(chunks: list[MinimalSource], save_dir: Path) -> bool:
    """
    reads file slices, tokenises them with some maths into a JSON
    in the form of an inverted index lookup table
    """
    # tracking how many words in each chunk so BM25 can penalise long chunks
    # [chunk_id:totalwordcount]
    doc_lengths: dict[int, int] = {}
    # counts how many different chunks contain a word, to reward rare words
    # [word:numberofchunksthatcontainit]
    doc_freqs: dict[str, int] = {}
    # nested dict mapping for BM25 to find the chunks tha contain a query word
    # word:{chunk_id:count} {"server":{chunk0:2, chunk5: 1}}
    term_freqs: dict[str, dict[int, int]] = {}
    # sum of all tokens across chunks, to calc average
    total_tokens = 0
    for chunk_id, chunk in tqdm(
            enumerate(chunks),
            total=len(chunks),
            desc="BM25 Indexing"):
        try:
            with open(chunk.file_path, "r", encoding="utf-8") as f:
                content = f.read()
            # slice the text contained in this chunk
            sliced = (content[
                chunk.first_character_index:chunk.last_character_index])
            tokens_list = tokenise(sliced)
            # extracting folder/file names after "data/raw"
            parts = Path(chunk.file_path).parts
            # skipping data, raw and vllm parent folders
            rel_parts = parts[3:]
            # adding these path names as extra words to the chunk text twice
            path_tokens = tokenise(" ".join(rel_parts))
            tokens_list.extend(path_tokens * 2)
            # save total word count for this chunk
            doc_lengths[chunk_id] = len(tokens_list)
            total_tokens += len(tokens_list)
            # counting how many times each word repeats inside this chunk
            chunk_freq: dict[str, int] = {}
            for t in tokens_list:
                # get(t, 0) asks the dict look for t if no t assigns 0
                chunk_freq[t] = chunk_freq.get(t, 0) + 1
            # merge this chunk word counts into the global index
            for w, count in chunk_freq.items():
                if w not in term_freqs:
                    term_freqs[w] = {}
                    doc_freqs[w] = 0
                term_freqs[w][chunk_id] = count
                doc_freqs[w] += 1
        except Exception as e:
            print(e)
            continue
    # calc average chunk length
    avgdl = 0.0
    if len(chunks):
        avgdl = total_tokens / len(chunks)
    # packaging chunks and BM25 lookups into a JSON dict
    # model_dump is a pydantic converter to a plain python dict
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
