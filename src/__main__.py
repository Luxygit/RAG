"""main execution of the module"""

import argparse
from pathlib import Path
from src.chunking import chunk_markdown, chunk_py


def main() -> None:
    """initialising master list of chunks and calling the chunk functions"""
    parser = argparse.ArgumentParser(
            description="RAG pipeline")
    parser.add_argument(
            "--max_chunk_size",
            "-c",
            type=int,
            default=2000,
            help="Max char size per chunk. Default 2000")
    args = parser.parse_args()
    all_chunks = []
    target_dir = Path(".")
    paths_md = list(target_dir.rglob("*.md"))
    paths_py = list(target_dir.rglob("*.py"))
    paths = [
            p for p in (paths_md + paths_py)
            if ".venv" not in p.parts and ".git" not in p.parts
            ]
    for path in paths:
        if path.suffix == ".md":
            all_chunks.extend(chunk_markdown(path, args.max_chunk_size))
        else:
            all_chunks.extend(chunk_py(path, args.max_chunk_size))
    print(f"Total chunks: {len(all_chunks)}")

if __name__ == "__main__":
    main()
