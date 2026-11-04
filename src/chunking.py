"""establishing chunking functions for py and md files"""

from validation import MinimalSource
from pathlib import Path


def chunk_markdown(path: Path) -> list[MinimalSource]:
    """
    splitting the content of every file into a list of 
    objects including their file_path, start and last index
    """
    chunks = []
    buffer = ""
    with open(path) as f:
        stringed = f.read()
        paragraphed = stringed.split("\n\n")
        curr_pos = 0
        chunk_start = 0
        for p in paragraphed:
            first_i = chunk_start
            last_i = curr_pos
            if len(buffer) + len(p) > 2000:
                chunks.append(MinimalSource(
                    file_path=str(path),
                    first_character_index=first_i,
                    last_character_index=last_i))
                chunk_start = curr_pos
                buffer = p
            else:
                if buffer:
                    buffer += "\n\n"
                buffer += p
            curr_pos += len(p) + 2
        if buffer:
            chunks.append(MinimalSource(
                        file_path=str(path),
                        first_character_index=chunk_start,
                        last_character_index=curr_pos))
    return chunks
