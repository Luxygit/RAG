*This project has been created as part of the 42 curriculum by dievarga*

# RAG against the machine

## Description

RAG TODO LIST:
- Retrieving: refactor the CLI to accept commands, tokenize the raw text, build the dictionaries for the BM25 math, and save it all to disk.
- OUTPUT
- README
- Docstrings
- flake8 
- mypy 

## Instructions

### Makefile targets

- make install      : create the virtual env and install dependencies
- make run          : run the project
- make debug        : run the project under pdb pythons debugger
- make clean        : remove caches and temp files
- make lint         : run flake8 and mypy
- make lint-strict  : execute the commands flake8 . and mypy . --strict

### Installation

make install - creates .venv and installs pygame visual graphic output.

### Debugging
Using make debug' we got access to pythons debugger to examine the program:
- n (Next): Executes the current line and moves to the next line
- s (Step): Steps inside a function call
- c (Continue): Lets the program run normally until it hits a crash or finishes
- p <variable> (Print): Displays the live contents of any variable
- l (List): Prints the surrounding lines of code
- q (Quit): Instantly terminates the debugger

### Execution

make run - executes program and it also installs dependencies if missing.

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
