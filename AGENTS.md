# Repository Guidelines

## Project Structure & Module Organization

This repository currently contains a small Python benchmark around a local SentenceTransformers model.

- `benchmarks/bench_encode.py` runs a local embedding encode check.
- `requirements-bench.txt` lists dependencies needed for benchmark execution.
- `models/all-MiniLM-L6-v2/` contains the local model files used by the benchmark. Keep model paths relative to the repository root so scripts remain portable.

Place new benchmark scripts under `benchmarks/`. Put reusable source modules in a top-level package only when logic is shared across scripts. Keep generated output, caches, and temporary experiment files out of version control.

## Build, Test, and Development Commands

Create and activate a virtual environment before installing dependencies:

```sh
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-bench.txt
```

Run the current benchmark:

```sh
python benchmarks/bench_encode.py
```

The benchmark loads `models/all-MiniLM-L6-v2` with `local_files_only=True`, so it should not require network access once dependencies are installed.

## Coding Style & Naming Conventions

Use Python 3 style conventions:

- 4-space indentation.
- `snake_case` for functions, variables, and script names.
- `PascalCase` only for classes.
- Keep scripts executable from the repository root.

Prefer explicit paths such as `models/all-MiniLM-L6-v2` over environment-dependent defaults. If adding formatters or linters, document the exact command here and keep configuration in the repository.

## Testing Guidelines

There is no committed automated test suite yet. For now, validate changes by running:

```sh
python benchmarks/bench_encode.py
```

When adding tests, use `pytest`, place tests under `tests/`, and name files `test_*.py`. Cover model-loading behavior, path handling, and any parsing or filtering logic added to the project.

## Commit & Pull Request Guidelines

This branch has no existing commits, so no project-specific commit convention is established. Use concise imperative commit messages, for example:

```text
Add benchmark dependency file
Update local model encode benchmark
```

Pull requests should include a short summary, the commands run for validation, and any changes to model files or dependency requirements. Link related issues when available. If a change affects benchmark output or performance, include representative before/after notes.

## Security & Configuration Tips

Do not commit credentials, API keys, private datasets, or machine-specific paths. Keep model assets reproducible and document their source when adding or replacing files under `models/`.
