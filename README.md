# Dosu-2

Dosu-2 is an experimental modular system for understanding and documenting codebases. It is built around the [PocketFlow](https://github.com/the-pocket/PocketFlow) framework where each capability is expressed as a flow of interconnected nodes.

Currently the repository only contains the **codebase-understanding** module, located in the `codebase-understanding/` directory. New independent or dependent modules can be added alongside it without changing the existing structure.

## Setup

1. Install Python **3.10+**.
2. Install the dependencies:

```bash
pip install -r requirements.txt
```

3. Provide credentials for your preferred LLM provider. The default implementation expects the `GEMINI_API_KEY` environment variable, but `codebase-understanding/utils/call_llm.py` can be adapted for other providers.
4. When crawling GitHub repositories, supply a personal access token using `--token` or the `GITHUB_TOKEN` environment variable to avoid rate limits.

## Running the codebase-understanding flow

From the repository root you can analyze a GitHub repo or a local directory.

```bash
# Analyze a public GitHub repository
python codebase-understanding/main.py --repo https://github.com/example/repo --token YOUR_GITHUB_TOKEN

# Analyze a local folder
python codebase-understanding/main.py --dir /path/to/project
```

This process downloads or reads the code, identifies abstractions and relationships, then produces tutorial chapters inside the directory specified by `--output` (default: `./output`).

### Common options

* `--include` / `--exclude` – glob patterns controlling which files are processed.
* `--max-size` – maximum file size in bytes (default: 100000).
* `--language` – language for the generated tutorial (default: `english`).
* `--no-cache` – disable LLM response caching.
* `--max-abstractions` – limit the number of abstractions (default: 10).

Run `python codebase-understanding/main.py --help` for the full list of parameters.

## Future modules

Dosu-2 is designed to grow. As new functionalities are developed, additional modules will appear beside `codebase-understanding` and plug into the overall system. This README will evolve to document how they interact.

Contributions and ideas are welcome.
