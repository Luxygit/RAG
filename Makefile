
all: install run

install:
	uv sync
run: install
	uv run python -m src index
	uv run python -m src search "How to configure the OpenAI server" -k 5
	# add all required

debug: install
	uv run python -m pdb src

clean:
	rm -rf __pycache__ .mypy_cache dist build *.egg-info

lint:
	flake8 .
	mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	flake8 .
	mypy . --strict

.PHONY: all install run debug clean lint lint-strict
