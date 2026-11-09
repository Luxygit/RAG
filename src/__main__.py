"""main execution of the module"""

import fire
from pathlib import Path
from src.chunking import chunk_markdown, chunk_py

def index(max_chunk_size: int = 2000) -> None:
    """executes chunking and indexin pipeline"""
    all_chunks = []
    # target_dir = Path("data/raw")
    # for testing only!
    target_dir = Path(".")
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
    print(f"Total chunks: {len(all_chunks)}")


def main() -> None:
    """initialising master list of chunks and calling the chunk functions"""
    fire.Fire({"index": index})


if __name__ == "__main__":
    main()
