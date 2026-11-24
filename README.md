*This project has been created as part of the 42 curriculum by dievarga*

# RAG against the machine

## Description

RAG Against the MAchine is a command line Retrieval Augmented Generation RAG
tool written in Python. Its goal is to answer technical questions about a
local databse and documentation data.
It works in three stages:
- Indexing: splits .py .md .txt files into character indexed chunks and builds
a BM25 inverted index saved as JSON
- Retrieval / Evaluation: Ranks chunks against user questions using BM25
scoring and measures retrieval recall against ground truth datasets using
character intersection over union (IoU)
- Generation: Feeds the top retrieved chunks into the local Qwen LM on CPU to
produce grounded answers formatted as validated Pydantic JSON schemas.

## Instructions

### Makefile targets

- make install      : create the .venv and install dependencies via uv sync
- make run          : run the project uv run python -m src <command> [options]
- make debug        : run the project under pdb pythons debugger __main__.py
- make clean        : remove caches and temp files
- make lint         : run flake8 . and mypy . to check src/ files
- make lint-strict  : run flake8 . and mypy . --strict 

### Installation

uv is used for dependency management.
make install runs uv sync which creates .venv and installs all dependencies.
- python-fire exposes some functions as CLI commands
- flake8 and mypy are required to check style, syntax and typing
- pydantic validates data schemas
- PyTorch is the framework that in this case runs the AI model on the CPU
- transformers loads and runs the AI model on PyTorch, tokenising and
generating text
- tqdm shows progress bars for visual feedback on operations

### Debugging
Using make debug' we got access to pythons debugger to examine the program:
- n (Next): Executes the current line and moves to the next line
- s (Step): Steps inside a function call
- c (Continue): Lets the program run normally until it hits a crash or finishes
- p <variable> (Print): Displays the live contents of any variable
- l (List): Prints the surrounding lines of code
- q (Quit): Instantly terminates the debugger

### Execution

make run - executes the whole pipeline and produces the answer for one question via RAG and model calls. It also installs dependencies if missing.
Mandatory CLI command and options:
- indexing the raw dataset and saving it to data/processed
uv run python -m src index
    -max_chunk_size [int]: 
- returns the top k sources for a query
uv run python -m src search [query]
    -k [int]
- searches for the sources for the whole dataset and produces StudentSearchRestuls JSON
uv run python -m src search_dataset
    -dataset_path [dataset_path]
    -k [int]
    -save_directory [output_path]
- gets an answere for a single query with the context given
uv run python -m src answer [query]
    -k [int]
- uv run python -m src answer_dataset
    -student_serach_results_path []
    -save_directory [output_path]
Common erros such as missing files, invalid paths, file permit errors,
duplicates, among others specified by the subject are handled via 

## Resources

### Documentation

-

### AI usage

AI (Claude & Gemini) were used to:
- Debug the code (mostly typos, parsing and path logic issues)

## System Architecture

### RAG Pipeline


### Chunking Strategy


### Retrieval Method

### Performance Analysisi

### Deisgn Decisions

### Challenges Faced

### Example Usage
