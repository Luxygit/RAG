"""establishing chunking functions for py and md files"""


import ast
from pathlib import Path
from src.validating import MinimalSource


def _flatten_large_classes(
    nodes: list[ast.stmt],
    line_map: list[int],
    lines: list[str],
    chunk_size: int,
   ) -> list[ast.stmt]:
    """unpacks class bodies into methods when a class > chunk_size"""
    # ast.stmt is the parent base class for all nodes in python code
    flattened: list[ast.stmt] = []
    for node in nodes:
        # only inspect class def that have a known ending line number
        # nodes could also be def or import
        if isinstance(node, ast.ClassDef) and node.end_lineno is not None:
            # node.lineno is the starting line
            start = line_map[node.lineno]
            # finding last char index in last line end_lineno
            end = line_map[node.end_lineno] + len(lines[node.end_lineno - 1])
            # node.body is a list of methods in the class
            # if class too big unpack its methods(nodes) for later checking
            if end - start > chunk_size and node.body:
                flattened.extend(node.body)
                continue
        flattened.append(node)
    return flattened


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
    if (
        not isinstance(path, Path)
        or isinstance(chunk_size, bool)
        or not isinstance(chunk_size, int)
        or chunk_size <= 0
        or not path.is_file()
    ):
        return []
    chunks = []
    try:
        with open(path, encoding="utf-8") as f:
            stringed = f.read()
    except Exception as e:
        print(e)
        return []
    # cut the text into separate paragraphs when a blank line appears
    paragraphed = stringed.split("\n\n")
    # exact char position in the file
    curr_pos = 0
    # char position of paragraphs now in the chunk
    chunk_start: int | None = None
    chunk_end: int | None = None
    for p in paragraphed:
        if not p.strip():
            curr_pos += len(p) + 2
            continue
        # exact char position of this single paragraph
        p_start = curr_pos
        p_end = curr_pos + len(p)
        # markdown title section start with #
        is_heading = p.lstrip().startswith("#")
        # save chunk if adding p exceeds maxcharsize or a new section starts
        if (chunk_start is not None
           and chunk_end is not None
           and (p_end - chunk_start > chunk_size or is_heading)):
            chunks.append(
                    MinimalSource(
                        file_path=str(path),
                        first_character_index=chunk_start,
                        last_character_index=chunk_end,
                        )
                    )
            # empty chunk so we start a new one
            chunk_start = None
            chunk_end = None
        # slice p if it exceeds maxchunksize
        if p_end - p_start > chunk_size:
            curr_split = p_start
            while curr_split < p_end:
                # min() chooses the lowest index so we never go past
                # either maxchunksize or p_end
                split_end = min(curr_split + chunk_size, p_end)
                chunks.append(
                        MinimalSource(
                            file_path=str(path),
                            first_character_index=curr_split,
                            last_character_index=split_end,
                            )
                        )
                curr_split = split_end
        # buffer if we are still under the limit include p in chunk
        else:
            # if chunk was empy mark where this new chunk begins
            if chunk_start is None:
                chunk_start = p_start
            # stretch end to fit this p in the chunk
            chunk_end = p_end
        # move cursor past this p + 2 chars for the removed \n\n
        curr_pos += len(p) + 2
    # save leftover p left in the chunk at the end of the file
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
    if (
        not isinstance(path, Path)
        or isinstance(chunk_size, bool)
        or not isinstance(chunk_size, int)
        or chunk_size <= 0
        or not path.is_file()
    ):
        return []
    chunks = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception as e:
        print(e)
        return []
    # lookup list: line_map[line_num] = starting char index
    # index 0 is a spacejust there to align it with AST starting at 1
    total_chars = 0
    line_map = [0]
    for line in lines:
        line_map.append(total_chars)
        total_chars += len(line)
    # join lines back into one string so ast.parse can read the whole file
    code_string = "".join(lines)
    try:
        # parse code into a map of blocks(imports, functions, classes)
        tree = ast.parse(code_string)
    except Exception as e:
        print(e)
        return []
    # stard end chard positions of chunk currently grouped
    chunk_start = None
    chunk_end = None
    # splitting potential big classes into smaller methods
    nodes = _flatten_large_classes(
            tree.body, line_map, lines, chunk_size
            )
    for node in nodes:
        # skip internal AST nodes
        if node.end_lineno is None:
            continue
        # converting a node start end lines into exact char indexes
        node_start = line_map[node.lineno]
        last_line = lines[node.end_lineno - 1]
        node_end = line_map[node.end_lineno] + len(last_line)
        # if adding this nodes goes over the limit, save it
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
                # use min() as a limit every time
                split_end = min(curr_split + chunk_size, node_end)
                chunks.append(
                        MinimalSource(
                            file_path=str(path),
                            first_character_index=curr_split,
                            last_character_index=split_end,))
                curr_split = split_end
        # otherwise add this function/import to the current chunk
        else:
            if chunk_start is None:
                chunk_start = node_start
            chunk_end = node_end
    # save the remaining piece of code left in the chunk at the end of the file
    if chunk_start is not None and chunk_end is not None:
        chunks.append(MinimalSource(
                            file_path=str(path),
                            first_character_index=chunk_start,
                            last_character_index=chunk_end))
    return chunks
