"""augmenting the minimalsources, loading the model and inferencing"""


def read_source(source: MinimalSource) -> str:
    """reads the chunk included in the minimalsource slices of the chunks"""
    if not Path(source.file_path).is_file():
        return ""
    try:
        with open(source.file_path,
                  "r",
                  encoding="utf-8",
                  errors="replace") as f:
            content = f.read()
            return (content[
                source.first_character_index:source.last_character_index])
    except OSError:
        print("Could not open file")
        return ""


def build_context(sources: list[MinimalSource], max_chars: int = 5000) -> str:
    """building the prompt that will be given to the model"""
    context: list[str] = []
    total_chars = 0
    for i, s in enumerate(sources, start=1):
        piece = read_source(s)
        header = f"[Source {i}: {s.file_path}]"
        if total_chars + len(piece) + len(header) >= max_chars:
            break
        if not piece.strip():
            continue
        context.append(header)
        context.append(piece)
        total_chars += len(piece) + len(header) + 4
    return "\n\n".join(context)


def answer_gen() -> str:
    return answer
