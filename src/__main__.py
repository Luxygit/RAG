""""""


from chunking import chunk_markdown


def main() -> None:
    my_path = "my_path/"
    paths_md = Path(my_path).rglob("*.md")
    paths_py = Path(my_path).rglob("*.py")
    paths = list(paths_md) + list(paths_py)
    for path in paths:
        chunk_markdown(path)
        py_markdown(path)
