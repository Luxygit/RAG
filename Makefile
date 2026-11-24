
all: install run

install:
	uv sync

run: install
	uv run python -m src index -max_chunk_size 2000
	uv run python -m src search_dataset -dataset_path data/dataset/UnansweredQuestions/dataset_docs_public.json -k 5 -save_directory data/output/search_results/UnansweredQuestions
	./moulinette evaluate_student_search_results data/output/search_results/UnansweredQuestions/dataset_docs_public.json data/dataset/AnsweredQuestions/dataset_docs_public.json -k 5 -max_context_length 2000
	uv run python -m src answer_dataset	-student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json -save_directory data/output/search_results_and_answer/UnansweredQuestions

debug: install
	uv run python -m pdb src/__main__.py

clean:
	rm -rf .mypy_cache dist build *.egg-info
	find . -type d -name "__pycache__" -exec rm -rf {} +

lint:
	uv run flake8 .
	uv run mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	uv run flake8 .
	uv run mypy . --strict

.PHONY: all install run debug clean lint lint-strict
