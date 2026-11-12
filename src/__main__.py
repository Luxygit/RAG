"""main execution of the module"""


import fire
import json
from tqdm import tqdm
from pathlib import Path
from src.chunking import chunk_markdown, chunk_py
from src.indexing import build_and_save_index
from src.retrieving import load_index, search_index
from src.validating import (
        MinimalSource,
        MinimalSearchResults,
        StudentSearchResults)


def index(max_chunk_size: int = 2000) -> None:
    """executes chunking and indexin pipeline"""
    all_chunks = []
    target_dir = Path("data/raw/")
    paths_md = list(target_dir.rglob("*.md"))
    paths_py = list(target_dir.rglob("*.py"))
    paths = [
            p for p in (paths_md + paths_py)
            if ".venv" not in p.parts and ".git" not in p.parts
            ]
    for path in tqdm(paths, desc="Chunking raw files"):
        if path.suffix == ".md":
            all_chunks.extend(chunk_markdown(path, max_chunk_size))
        else:
            all_chunks.extend(chunk_py(path, max_chunk_size))
    save_dir = Path("data/processed")
    build_and_save_index(all_chunks, save_dir)
    print(f"Saved processed {len(all_chunks)} chunks")


def search(query: str, k: int = 5) -> None:
    """"""
    target_dir = Path("data/processed/bm25_index.json")
    index_data = load_index(target_dir)
    # searchindex returns a list of raw dicts[MinimalSource]
    results = search_index(query, index_data, k)
    # 
    retrieved_sources = [MinimalSource(**chunk) for chunk in results]
    min_search = MinimalSearchResults(
            question_id="q1",
            question=query,
            retrieved_sources=retrieved_sources)
    student_res = StudentSearchResults(
            search_results=[min_search],
            k=k
            )
    print(student_res.model_dump_json(indent=4))


def search_dataset(dataset_path: str, k: int, save_directory: str) -> None:
    """runs search over a dataset and saves results to json"""
    # loading dataset and bm25 index
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    index_path = Path("data/processed/bm25_index.json")
    index_data = load_index(index_path)
    search_results_list = []
    # looping through questions, either raw lists or dicts
    questions = dataset.get("rag_questions",
            dataset) if isinstance(dataset, dict) else dataset
    for item in tqdm(questions, desc="Searching datasets"):
        q_id = item.get("question_id", "q1")
        q_text = item.get("question", "")
        # run search engine
        raw_chunks = search_index(q_text, index_data, k)
        retrieved_sources = [MinimalSource(**chunk) for chunk in raw_chunks]
        min_search = MinimalSearchResults(
                question_id=q_id,
                question=q_text,
                retrieved_sources=retrieved_sources,
                )
        search_results_list.append(min_search)
    # packaging into StudentSearchResults
    student_res = StudentSearchResults(
            search_results=search_results_list,
            k=k)
    # turning string into Path object
    save_dir = Path(save_directory)
    save_dir.mkdir(parents=True, exist_ok=True)
    output_file = save_dir / Path(dataset_path).name
    with open(output_file, "w", encoding="utf-8") as f:
            f.write(student_res.model_dump_json(indent=4))
    print(f"Saved student_search_results to {output_file}")


def main() -> None:
    """initialising master list of chunks and calling the chunk functions"""
    fire.Fire({
        "index": index,
        "search": search,
        "search_dataset": search_dataset,
        "evaluate": evaluate,
        })

if __name__ == "__main__":
    main()
