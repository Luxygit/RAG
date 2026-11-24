"""main execution of the module"""


import fire
import json
import sys
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
    if (
            isinstance(max_chunk_size, bool)
            or not isinstance(max_chunk_size, int)
            or max_chunk_size <= 0
            ):
        print("Invalid max_chunk_size: must be positive int")
        return
    all_chunks = []
    target_dir = Path("data/raw/")
    if not target_dir.is_dir():
        print("Invalid target folder path")
        return
    try:
        paths_py = list(target_dir.rglob("*.py"))
        paths_txt = list(target_dir.rglob("*.txt"))
        paths_md = list(target_dir.rglob("*.md"))
    except Exception as e:
        print(f"Error scanning dir: {e}")
        return
    # skip hidden venv and git dirs
    paths = [
            p for p in (paths_md + paths_py + paths_txt)
            if ".venv" not in p.parts and ".git" not in p.parts
            ]
    for path in tqdm(paths, desc="Chunking raw files"):
        try:
            if path.suffix in {".md", ".txt"}:
                all_chunks.extend(chunk_markdown(path, max_chunk_size))
            else:
                all_chunks.extend(chunk_py(path, max_chunk_size))
        except Exception:
            continue
    save_dir = Path("data/processed")
    if build_and_save_index(all_chunks, save_dir):
        print(f"Saved processed {len(all_chunks)} chunks")


def search(query: str, k: int = 5) -> None:
    """runs a single BM25 search and prints JSON results"""
    if (
        not isinstance(query, str)
        or not isinstance(k, int)
        or isinstance(k, bool)
        or not query.strip()
        or k <= 0
    ):
        print("No valid query or no valid k int")
        return
    index_path = Path("data/processed/bm25_index.json")
    if not index_path.is_file():
        print("No valid target path")
        return
    try:
        index_data = load_index(index_path)
        # searchindex returns a list of raw dicts[MinimalSource]
        results = search_index(query, index_data, k)
        # **chunk unpacks dict keys into MinimalSource(file_path=...)
        retrieved_sources = [MinimalSource(**chunk) for chunk in results]
        # wrap the single query and its top-k sources into the schema
        min_search = MinimalSearchResults(
            question_id="q1",
            question=query,
            retrieved_sources=retrieved_sources)
    except Exception as e:
        print(e)
        return
    # outer schema required by subject
    student_res = StudentSearchResults(
            search_results=[min_search],
            k=k
            )
    # model_dump_json turns pydantic models into a json string
    print(student_res.model_dump_json(indent=4))


def search_dataset(dataset_path: str, k: int, save_directory: str) -> None:
    """runs search over a dataset and saves results to json"""
    # loading dataset and bm25 index and saves results to json
    if (
        not isinstance(dataset_path, (str, Path))
        or not str(dataset_path).strip()
        or not isinstance(save_directory, (str, Path))
        or not str(save_directory).strip()
        or isinstance(k, bool)
        or not isinstance(k, int)
        or k <= 0
    ):
        print("Invalid arguments for search_dataset")
        return
    try:
        with open(dataset_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
    except (OSError, json.JSONDecodeError, TypeError):
        print("Error opening file")
        return
    index_path = Path("data/processed/bm25_index.json")
    if not index_path.is_file():
        print("invalid index path")
        return
    try:
        index_data = load_index(index_path)
    except Exception as e:
        print(e)
        return
    search_results_list = []
    # looping through questions, either raw lists or dicts
    questions = (
            dataset.get("rag_questions", dataset)
            if isinstance(dataset, dict)
            else dataset
            )
    if not isinstance(questions, list):
        return
    for item in tqdm(questions, desc="Searching datasets"):
        if not isinstance(item, dict):
            continue
        try:
            q_id = str(item.get("question_id") or "q1")
            q_text = str(item.get("question") or "")
            # run search engine
            raw_chunks = search_index(q_text, index_data, k)
            # unpack each chunk dict into a validated MinimalSource
            retrieved_sources = [
                    MinimalSource(**chunk) for chunk in raw_chunks]
            min_search = MinimalSearchResults(
                    question_id=q_id,
                    question=q_text,
                    retrieved_sources=retrieved_sources,
                    )
            search_results_list.append(min_search)
        except Exception:
            continue
    # packaging into StudentSearchResults
    student_res = StudentSearchResults(
            search_results=search_results_list,
            k=k)
    # turning string into Path object
    try:
        save_dir = Path(save_directory)
        # create parent folders if missing without failing if they exist
        save_dir.mkdir(parents=True, exist_ok=True)
        output_file = save_dir / Path(dataset_path).name
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(student_res.model_dump_json(indent=4))
        print(f"Saved student_search_results to {output_file}")
    except Exception as e:
        print(e)
        return


def answer(query: str, k: int = 5) -> None:
    """answers a single query using retrieved context and prints JSON"""
    if (
        not isinstance(query, str)
        or not isinstance(k, int)
        or isinstance(k, bool)
        or not query.strip()
        or k <= 0
    ):
        print("No valid query or no valid k int")
        return
    index_path = Path("data/processed/bm25_index.json")
    if not index_path.is_file():
        print("No valid processed file path")
        return
    try:
        index_data = load_index(index_path)
        src = search_index(query, index_data, k)
        # unpack retrieved dicst into MinimalSource for type safety
        sources = [MinimalSource(**chunk) for chunk in src]
        tokenizer, model = load_model()
        if tokenizer is None or model is None:
            return
        gen_ans = generate_answer(query, sources, tokenizer, model)
    except Exception as e:
        print(e)
        return
    # MinimalAnswer inherits from MinimalSearchResults and adds 'answer'
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
    """generates answers for a search results dataset and saved to json"""
    if (
        not isinstance(student_search_results_path, (str, Path))
        or not str(student_search_results_path).strip()
        or not isinstance(save_directory, (str, Path))
        or not str(save_directory).strip()
    ):
        print("Invalid arguments for answer_dataset")
        return
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
        if tokenizer is None or model is None:
            return
    except Exception as e:
        print(e)
        return
    list_ans: list[MinimalAnswer] = []
    for item in tqdm(search_data.search_results, desc="Generating answers"):
        q_text = item.question
        try:
            ans_text = generate_answer(
                    q_text,
                    item.retrieved_sources,
                    tokenizer,
                    model)
        except Exception:
            ans_text = ""
        ans_item = MinimalAnswer(
                question_id=item.question_id,
                question=q_text,
                retrieved_sources=item.retrieved_sources,
                answer=ans_text
                )
        list_ans.append(ans_item)
    formatted_answer = StudentSearchResultsAndAnswer(
            search_results=list_ans,
            k=search_data.k)
    save_dir = Path(save_directory)
    try:
        save_dir.mkdir(parents=True, exist_ok=True)
        with open(save_dir / input_file.name, "w", encoding="utf-8") as f:
            f.write(formatted_answer.model_dump_json(indent=4))
            print("Loaded all questions")
            print("Saved Student Search Results and Answer to"
                  f" {save_dir}/{input_file.name}")
    except Exception as e:
        print(f"could not save results to file {e}")
        return


def main() -> None:
    """entry point for all the RAG CLI commands"""
    try:
        # exposing functions as CLI commands
        fire.Fire({
            "index": index,
            "search": search,
            "search_dataset": search_dataset,
            "evaluate": evaluate,
            "answer": answer,
            "answer_dataset": answer_dataset,
            })
    except KeyboardInterrupt:
        print("\nProcess interrupted", file=sys.stderr)
        sys.exit(130)
    except BrokenPipeError:
        sys.exit(0)


if __name__ == "__main__":
    main()
