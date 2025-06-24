# InfraDoc

InfraDoc analyzes live server infrastructure and generates useful documentation. It connects to hosts via SSH, collects running processes and configuration files, and can leverage LLMs to produce architecture diagrams and security reports.

## Project Files

- `infradoc_core.py` – SSH connector, LLM orchestrator and discovery helpers.
- `infradoc_analyzer.py` – orchestrates scans and aggregates results.
- `infradoc_docs.py` – turns analysis data into markdown documentation.
- `infradoc_cli.py` – command line interface for running scans.

## Installation

Install the required libraries:

```bash
pip install paramiko openai anthropic
```

Set API keys if you want LLM-powered analysis:

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
```

## Usage

Quick scan of a server:

```bash
python infradoc_cli.py quick --host myserver --key-file ~/.ssh/id_rsa
```

Deep analysis with full documentation:

```bash
python infradoc_cli.py deep --host myserver --key-file ~/.ssh/id_rsa
```

Custom analysis with specific options:

```bash
python infradoc_cli.py analyze --host myserver --key-file ~/.ssh/id_rsa \
  --depth standard --output-dir ./output
```

## Output

Results are saved in a folder named `infradoc_<scan_id>`. Each scan includes a JSON report and, if documentation is enabled, a `documentation/` directory with Markdown files and diagrams.

InfraDoc performs read-only operations and can run without AI features using `--no-ai`.
