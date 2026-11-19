
all: install run

install:
	uv sync

run: install
	uv run python -m src answer "How to configure the OpenAI server" -k 5

debug: install
	uv run python -m pdb src

clean:
	rm -rf __pycache__ .mypy_cache dist build *.egg-info

lint:
	uv run flake8 .
	uv run mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	uv run flake8 .
	uv run mypy . --strict

.PHONY: all install run debug clean lint lint-strict
