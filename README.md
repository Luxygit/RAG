*This project has been created as part of the 42 curriculum by dievarga*

# RAG Against the Machine

## Description

RAG Against the Machine is a command line Retrieval Augmented Generation RAG
tool written in Python. Its goal is to answer technical questions about a
local database and documentation data.
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
- make run          : run the whole rag program pipeline
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
Using make debug we got access to pythons debugger to examine the program:
- n (Next): Executes the current line and moves to the next line
- s (Step): Steps inside a function call
- c (Continue): Lets the program run normally until it hits a crash or finishes
- p <variable> (Print): Displays the live contents of any variable
- l (List): Prints the surrounding lines of code
- q (Quit): Instantly terminates the debugger

### Execution

make run - executes the whole RAG pipeline including the Moulinette tester as required.It also installs dependencies if missing.
Mandatory CLI command and options:
- indexing the raw dataset and saving it to data/processed/
uv run python -m src index
    -max_chunk_size [int] 
- returning the top k sources for a query
uv run python -m src search [query]
    -k [int]
- searching for the sources for the whole dataset and produces StudentSearchResults JSON file
uv run python -m src search_dataset
    -dataset_path [dataset_path]
    -k [int]
    -save_directory [output_path]
- getting an answer for a single query with the context given
uv run python -m src answer [query]
    -k [int]
- generating answers for a whole dataset, producing StudentSearchResultsAndAnswer JSON file
uv run python -m src answer_dataset
    -student_search_results_path [student results path]
    -save_directory [output_path]
- Testing our result against a ground truth dataset (recall@k)
uv run python -m src evaluate
    -student_search_results_path [student results path]
    -dataset_path [dataset path]

Common erros such as missing files, invalid paths, file permission errors,
duplicates, among others specified by the subject are handled via Pydantic and
try except blocks.

## Resources

### Documentation

- Okapi BM25 Ranking Function
- Intersection over Union (IoU) Metric
- Hugging Face Transformers Documentation
- Qwen3-0.6B Model Card
- PyTorch CPU Inference & no_grad Documentation
- Pydantic V2 Models & Validation
- Python ast Module Documentation
- Python Fire CLI Documentation
- Python tqdm documentation

### AI usage

AI (Claude & Gemini) were used to:
- Debug the code (mostly typos, parsing and path logic issues)
- Clarify transformers and pytorch concepts and formulas
- Understand BM25 theory

### System Architecture

Inside the data folder there are raw, processed, output and datasets.
The raw folder contains the raw data to be optimised by this program.
The processed folder holds the indexed data.
The datasets are provided as the questions and their expected answers.
And the output will contain the program output once it has found the right 
context for every question and then its answer from the model.

The project core lives inside a module in the src/ folder where each file name
describes what it does.
- chunking.py   Splits files into text/code pieces with character offsets
- indexing.py   Builds a BM25 word-to-chunk index
- retrieving.py Scores and ranks the top-k most relevant chunks for a question
- evaluating.py Measures retrieval accuracy (Recall@k via IoU)
- generating.py Sends top chunks + question to Qwen3-0.6B on CPU to generate an answer
- validating.py Validates all inputs/outputs with Pydantic and prints clean JSON
The __main__.py in src/ executes the principal functions of the project that are required by the subject via CLI commands as described above in the Execution section.

The RAG pipeline has four stages:
- Indexing: organise the data so it can be searched.
- Retrieving: match a question against the index and pull the most relevant snippets.
- Augmenting: filter those snippets and place them in the model’s context window.
- Generating: read that context and produce the answer.

### Chunking Strategy

Before searching in the raw files, we have to break them up into smaller chunks so that the model only receives relevant snippets.
Chunking was structured in 2 separate functions depending on the type of file:
- Markdown and Text files are split by paragraph boundaries (\n\n) and md headings (#).
Paragraphs are grouped together until the next one would pass the max chunk size set by default at 2000 chars or there is a new heading section.
- Python code files were mapped with a native AST module that turns them into 
abstract syntax trees that show where every import, def and class starts and ends.
This way grouping complete functions and classes and flattening to split it into methods in case one class is too big.

Every chunk saves its first char index and last char index and file path so that its content can be found later.

### Retrieval Method

An inverted Okapi BM25 index is used to find the best chunks for a question.
- First text is lowercased and split into alphanum words. Common filler words are removed. Code identifiers in camelCase are split into the word and its parts so natural language questions match function code names.
- Words from the files relative path are added twice to each chunk tokens so that the name of the file ranks higher.
- The indexed files follow a JSON inverted index structure that shows each word and a list of chunk IDs where it appears and how many times.
- When a query is passed BM25 scores each matching chunk with:
    - Inverse document frequency: so that rare words score higher
    - Term frequency saturation: repeating the same word in a file excessively gives
        diminishing returns
    - Length Normalisation: short chunks that match, score higher than huge chunks

### Performance Analysis

- Retrieval speed: using only simple math and python dictionaries with BM25 searching 200 questions can take only a few seconds (~3 sec).
- Recall@k Evaluation: Quality is measured using Intersection over Union between the program retrieved character spans and the ground truth answer spans. A question counts as a hit if at least one of the top-k chunks comes from the right file and overlaps by IoU >= 0.05. IoU is calculated dividing the overlap of both data character spans by their total covered area. 0 means no overlap and 1 is a exact match.
- Generation speed: Running Qwen3-0.6B on CPU takes about 15 seconds per question so getting answers from the model for 200 questions would take about 50 minutes.

### Design Decisions

- Lazy ML imports: torch and transformers are imported inside load_model() so that when running commands that do not need these modules are executed faster.
- float32 vs bfloat16 on CPU: Testing shows that some CPUs without native instructions have to emulate bfloat16 making the processing slower so torch.float32 was chosen.
- Grounded greedy decoding: using model.eval() torch.no_grad() do_sample=False and enable_thinking=False with a 2000 char context limit so that the model produces a fast, concise, grounded answer without <think> blocks.

### Challenges Faced

- Breaking code files up: Simple character count splitting broke functions in half and hurt retrieval accuracy. AST abstract trees helps avoiding this.
- Progress bar visibility: a hook had to be attached to every model.generate() 28 decoder layers made it possible to see the actual progress helped by BaseStreamer for token by token generation.
- python-fire edge cases: malformed flag inputs had to be checked through isinstance() and try excepts to avoid crashes.

### Example Usage

- Single query search
uv run python -m src search "how to uninstall openai server?" -k 2

- Single query answer
uv run python -m src answer "how to delete server?" -k 5
