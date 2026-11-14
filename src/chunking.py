"""establishing chunking functions for py and md files"""


from pathlib import Path
import ast
from src.validating import MinimalSource


def chunk_markdown(
        path: Path,
        chunk_size: int = 2000
        ) -> list[MinimalSource]:
    """
    splitting the content of every file by paragraphs using indexes
    into a list of objects including their file_path, start and last index
    chunk_start is the starting index of the current buffer
    chunk_end the end index of the current buffer
    curr_pos tracks the position in file during the loop
    p_start p_end the boundaries of the current paragraph
    """
    chunks = []
    with open(path, encoding="utf-8") as f:
        stringed = f.read()
    paragraphed = stringed.split("\n\n")
    curr_pos = 0
    chunk_start: int | None = None
    chunk_end: int | None = None
    for p in paragraphed:
        p_start = curr_pos
        p_end = curr_pos + len(p)
        # flushing chunk if exceeding
        if (chunk_start is not None
           and chunk_end is not None
           and (p_end - chunk_start > chunk_size)):
            chunks.append(
                    MinimalSource(
                        file_path=str(path),
                        first_character_index=chunk_start,
                        last_character_index=chunk_end,
                        )
                    )
            chunk_start = None
            chunk_end = None
        # slice if the p itself exceeds
        if p_end - p_start > chunk_size:
            curr_split = p_start
            while curr_split < p_end:
                split_end = min(curr_split + chunk_size, p_end)
                chunks.append(
                        MinimalSource(
                            file_path=str(path),
                            first_character_index=curr_split,
                            last_character_index=split_end,
                            )
                        )
                curr_split = split_end
        # buffer if we are still under the limit
        else:
            if chunk_start is None:
                chunk_start = p_start
            chunk_end = p_end
        # advancing the position for the \n\n
        curr_pos += len(p) + 2
    # flushing leftovers after loop
    if chunk_start is not None and chunk_end is not None:
        chunks.append(
                MinimalSource(
                    file_path=str(path),
                    first_character_index=chunk_start,
                    last_character_index=chunk_end,
                    )
                )
    return chunks


def chunk_py(
        path: Path,
        chunk_size: int = 2000
        ) -> list[MinimalSource]:
    """using ast lib to get a py file tree and branches"""
    chunks = []
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    total_chars = 0
    line_map = [0]
    for line in lines:
        line_map.append(total_chars)
        total_chars += len(line)
    code_string = "".join(lines)
    tree = ast.parse(code_string)
    chunk_start = None
    chunk_end = None
    for node in tree.body:
        if node.end_lineno is None:
            continue
        node_start = line_map[node.lineno]
        last_line = lines[node.end_lineno - 1]
        node_end = line_map[node.end_lineno] + len(last_line)
        # if adding this nodes goes over the limit, flush it
        if (
                chunk_start is not None
                and chunk_end is not None
                and (node_end - chunk_start > chunk_size)
                ):
            chunks.append(
                    MinimalSource(
                        file_path=str(path),
                        first_character_index=chunk_start,
                        last_character_index=chunk_end,
                        ))
            chunk_start = None
            chunk_end = None
        # if node is larger than the limit, slice it
        if node_end - node_start > chunk_size:
            curr_split = node_start
            while curr_split < node_end:
                split_end = min(curr_split + chunk_size, node_end)
                chunks.append(
                        MinimalSource(
                            file_path=str(path),
                            first_character_index=curr_split,
                            last_character_index=split_end,))
                curr_split = split_end
        # otherwise we add it to the buffer
        else:
            if chunk_start is None:
                chunk_start = node_start
            chunk_end = node_end
    # flush the remaining piece
    if chunk_start is not None and chunk_end is not None:
        chunks.append(MinimalSource(
                            file_path=str(path),
                            first_character_index=chunk_start,
                            last_character_index=chunk_end))
    return chunks
