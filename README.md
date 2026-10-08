# Integrum

**Integrum** turns any Python module, library, function, or class into a [Model Context Protocol (MCP)](https://modelcontextprotocol.io) server powered by [FastMCP](https://github.com/jlowin/fastmcp).

Integrum uses Python reflection (`inspect`), regex filtering, and an in-memory **Word-Based Object Store** (`playful_llama`, `swift_falcon`) so that object-oriented libraries (such as `scikit-learn`, `pandas`, `PyTorch`) can be used naturally by LLMs across multi-turn conversations without token bloat.

## Features

- **Module Reflection**: Automatically introspects functions and classes using `inspect`.
- **Regex Filtering**: Expose only what you need with `--include` and `--exclude` regex patterns.
- **Word-Based State Management**: Object IDs are short, memorable words (e.g. `playful_llama`) to save tokens and avoid UUID hallucination.
- **Multi-User Isolation**: Supports FastMCP `Context` sessions to partition objects per connecting client.
- **Docstring Instruction Injection**: Transforms Python docstrings into LLM tool documentation and automatically injects state-handling instructions.
- **Dot-Notation Attribute Resolution**: Pass `swift_falcon.data` or `swift_falcon.target` directly to subsequent tool calls like `fit()`.
- **CLI & Python API**: Run from the command line (`integrum expose`) or integrate directly into Python scripts.

## Installation

```bash
uv pip install integrum
# or
pip install integrum
```

## Quick Start (CLI)

Expose `scikit-learn` dataset loaders and RandomForestClassifier as an MCP server:

```bash
integrum expose sklearn.datasets.load_iris sklearn.ensemble.RandomForestClassifier
```

Preview what tools will be exposed:

```bash
integrum list sklearn.datasets.load_iris sklearn.ensemble.RandomForestClassifier
```

Filter functions and methods with regex:

```bash
integrum expose sklearn.datasets sklearn.ensemble \
  --include "^(load_iris|RandomForestClassifier)$"
```

Run with HTTP or SSE transport:

```bash
integrum expose sklearn.datasets.load_iris --transport http --port 8000
```

## Python API

```python
from integrum import IntegrumServer

server = IntegrumServer(name="ml-server")
server.expose(["sklearn.datasets.load_iris", "sklearn.ensemble.RandomForestClassifier"])

# Run over stdio (default for MCP desktop clients)
server.run(transport="stdio")
```

## How State Management Works

When a class is exposed, Integrum registers:
1. `<ClassName>_create`: Instantiates the class and stores it under a word ID (e.g., `playful_llama`).
2. `<ClassName>_<method>`: Accepts a virtual parameter `obj_id: str` (e.g., `obj_id="playful_llama"`), resolves the underlying instance, and executes the method.

Returned objects (such as datasets or numpy arrays) are also registered in the store with word IDs, allowing subsequent tools to receive them as arguments.

## Examples

Check out [`examples/gemma_sklearn_session.md`](examples/gemma_sklearn_session.md) and [`examples/gemma_sklearn_session.json`](examples/gemma_sklearn_session.json) for a complete, verified multi-turn transcript of Gemma 4 26B loading the Iris dataset, performing `train_test_split`, training a `RandomForestClassifier`, and evaluating accuracy.
