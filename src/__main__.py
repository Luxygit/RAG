"""main execution of the module"""


import fire
import json
from tqdm import tqdm
from pathlib import Path
from pydantic import ValidationError
from src.chunking import chunk_markdown, chunk_py
from src.indexing import build_and_save_index
from src.retrieving import load_index, search_index
from src.evaluating import evaluate
from src.validating import (
        MinimalSource,
        MinimalSearchResults,
        StudentSearchResults,
        MinimalAnswer,
        StudentSearchResultsAndAnswer)
from src.generating import generate_answer, load_model


def index(max_chunk_size: int = 2000) -> None:
    """executes chunking and indexin pipeline"""
    all_chunks = []
    target_dir = Path("data/raw/")
    paths_py = list(target_dir.rglob("*.py"))
    paths_txt = list(target_dir.rglob("*.txt"))
    paths_md = list(target_dir.rglob("*.md"))
    paths = [
            p for p in (paths_md + paths_py + paths_txt)
            if ".venv" not in p.parts and ".git" not in p.parts
            ]
    for path in tqdm(paths, desc="Chunking raw files"):
        if path.suffix in {".md", ".txt"}:
            all_chunks.extend(chunk_markdown(path, max_chunk_size))
        else:
            all_chunks.extend(chunk_py(path, max_chunk_size))
    save_dir = Path("data/processed")
    build_and_save_index(all_chunks, save_dir)
    print(f"Saved processed {len(all_chunks)} chunks")


def search(query: str, k: int = 5) -> None:
    """"""
    if not query.strip() or k <= 0:
        print("No valid query or no valid k int")
        return
    target_dir = Path("data/processed/bm25_index.json")
    if not target_dir.is_file():
        print("No valid target path")
        return
    index_data = load_index(target_dir)
    # searchindex returns a list of raw dicts[MinimalSource]
    results = search_index(query, index_data, k)
    # expplain this
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
    try:
        with open(dataset_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
    except (OSError, json.JSONDecodeError):
        print("Error opening file")
        return
    index_path = Path("data/processed/bm25_index.json")
    index_data = load_index(index_path)
    search_results_list = []
    # looping through questions, either raw lists or dicts
    questions = dataset.get(
            "rag_questions",
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


def answer(query: str, k: int = 5) -> None:
    """"""
    if not query.strip() or k <= 0:
        print("No valid query or no valid k int")
        return
    index_path = Path("data/processed/bm25_index.json")
    if not index_path.is_file():
        print("No valid processed file path")
        return
    index_data = load_index(index_path)
    src = search_index(query, index_data, k)
    sources = [MinimalSource(**chunk) for chunk in src]
    tokenizer, model = load_model()
    gen_ans = generate_answer(query, sources, tokenizer, model)
    ans = MinimalAnswer(
            question_id="q1",
            question=query,
            retrieved_sources=sources,
            answer=gen_ans,
            )
    res = StudentSearchResultsAndAnswer(search_results=[ans], k=k)
    print(res.model_dump_json(indent=4))


def answer_dataset(
        student_search_results_path: str,
        save_directory: str
        ) -> None:
    """"""
    input_file = Path(student_search_results_path)
    if not input_file.is_file():
        print("Invalid input file")
        return
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            j = json.load(f)
            # pydantic validation gives back minimalsearchresults
            search_data = StudentSearchResults.model_validate(j)
    except (OSError, json.JSONDecodeError, ValidationError, Exception) as e:
        print(e)
        return
    try:
        tokenizer, model = load_model()
    except Exception as e:
        print(e)
        return
    list_ans: list[MinimalAnswer] = []
    for item in tqdm(search_data.search_results, desc="Generating answers"):
        q_text = item.question
        ans_text = generate_answer(
                q_text,
                item.retrieved_sources,
                tokenizer,
                model)
        answer = MinimalAnswer(
                question_id=item.question_id,
                question=q_text,
                retrieved_sources=item.retrieved_sources,
                answer=ans_text
                )
        list_ans.append(answer)
    formatted_answer = StudentSearchResultsAndAnswer(
            search_results=list_ans,
            k=search_data.k)
    save_dir = Path(save_directory)
    try:
        save_dir.mkdir(parents=True, exist_ok=True)
        with open(save_dir / input_file.name, "w", encoding="utf-8") as f:
            f.write(formatted_answer.model_dump_json(indent=4))
            print("Saved Student Search Restuls and Answer to file")
    except Exception as e:
        print(f"could not save results to file {e}")
        return


def main() -> None:
    """initialising master list of chunks and calling the chunk functions"""
    fire.Fire({
        "index": index,
        "search": search,
        "search_dataset": search_dataset,
        "evaluate": evaluate,
        "answer": answer,
        "answer_dataset": answer_dataset,
        })


if __name__ == "__main__":
    main()
