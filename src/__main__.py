"""main execution of the module"""

import fire
from pathlib import Path
from src.chunking import chunk_markdown, chunk_py
from src.indexing import build_and_save_index

def index(max_chunk_size: int = 2000) -> None:
    """executes chunking and indexin pipeline"""
    all_chunks = []
    target_dir = Path("data/raw/")
    # for testing only!
    # target_dir = Path(".")
    paths_md = list(target_dir.rglob("*.md"))
    paths_py = list(target_dir.rglob("*.py"))
    paths = [
            p for p in (paths_md + paths_py)
            if ".venv" not in p.parts and ".git" not in p.parts
            ]
    for path in paths:
        if path.suffix == ".md":
            all_chunks.extend(chunk_markdown(path, max_chunk_size))
        else:
            all_chunks.extend(chunk_py(path, max_chunk_size))
    # for testing
    print(f"Total chunks: {len(all_chunks)}")
    save_dir = Path("data/processed")
    build_and_save_index(all_chunks, save_dir)
    print(f"Saved processed {len(all_chunks)} chunks")


def main() -> None:
    """initialising master list of chunks and calling the chunk functions"""
    fire.Fire({"index": index})


if __name__ == "__main__":
    main()
