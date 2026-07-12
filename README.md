# knowcard

Local agent memory system with hybrid vector + keyword search. Knowledge is stored as markdown "cards" with YAML front matter, indexed for semantic and keyword retrieval, and versioned with an embedded git repository.

## Features

- **Knowledge cards** — markdown files with structured YAML front matter (title, keywords, summary, body, reference)
- **Hybrid search** — combines vector similarity (chromem-go) with BM25 keyword matching, fused via Reciprocal Rank Fusion (RRF)
- **CPU embedding** — runs embedding models locally via [yzma](https://github.com/hybridgroup/yzma) (llama.cpp Go bindings, no CGo required); also supports Ollama/OpenAI-compatible APIs
- **CJK support** — custom tokenizer handles Chinese/Japanese/Korean text with character bigrams alongside Latin word segmentation
- **Versioned** — every card change is auto-committed via go-git; full revision history is queryable
- **Separated VCS** — git metadata lives in `_vcs/`, not `.git/` in the cards directory, so standard `git` CLI commands won't interfere
- **Single binary** — serves as a Go library, CLI tool, and MCP server

## Architecture

```
<project-root>/.knowcard/
├── cards/              Knowledge cards as .md files (semantic tree structure)
├── _vcs/               Git metadata (separated from cards, CLI-invisible)
├── index/
│   └── chromem/        Vector index (derived, rebuildable)
├── models/             GGUF embedding model cache
├── manifest.json       Integrity checkpoint (HEAD commit + card count)
└── knowcard.yaml       Config file
```

knowcard searches upward from the current directory to find the `.knowcard/` directory, just like git finds `.git/`. Each project has its own knowledge base scoped to its domain.

**Card format:**

```markdown
---
id: 7f3a2b1c
title: Go 内存逃逸分析
keywords: [逃逸分析, 栈分配, 闭包]
summary: 解释什么情况下变量从栈逃逸到堆
reference: /docs/go/escape-analysis.md
tags: [go, performance]
created: 2026-07-10T10:00:00Z
updated: 2026-07-10T10:00:00Z
---

# Go 内存逃逸分析

正文内容...
```

## Usage

### CLI

```bash
# Initialize .knowcard in current project directory
knowcard init

# Add a card
knowcard add --path "programming/go/escape-analysis" \
  --title "Go 内存逃逸分析" \
  --summary "解释变量逃逸到堆的条件和检测方法" \
  --keywords "逃逸分析,栈分配,go" \
  --body "$(cat card_body.md)"

# Search
knowcard recall "逃逸分析" -k 10

# Read full card
knowcard show <card-id>

# List all cards
knowcard list

# View revision history
knowcard history <card-id>
```

### MCP Server

```bash
knowcard serve
```

Exposes 4 tools for AI agents:

| Tool | Description |
|---|---|
| `recall` | Hybrid search — returns card summaries ranked by relevance |
| `get_cards` | Retrieve full card content by IDs |
| `upsert_card` | Create or update a knowledge card |
| `delete_card` | Delete a card by ID |

### Go Library

```go
import kc "github.com/robert/knowcard"

store, err := kc.Open(cfg)
results, err := store.Recall("逃逸分析", kc.RecallOpts{TopK: 5})
cards, err := store.GetCards([]string{results[0].ID})
```

## Configuration

### Qwen3-VL-Embedding via DashScope (multimodal, recommended)

```yaml
# <project>/.knowcard/knowcard.yaml
embed:
  backend: qwen_cloud
  model: qwen3-vl-embedding          # or tongyi-embedding-vision-plus
  api_key: sk-xxx                    # DashScope API key
  dashscope_international: false     # true for intl endpoint
  dimensions: 1024                   # MRL: 64-4096 depending on model
  enable_fusion: true                # qwen3-vl: fuse text+image into one vector
  instruct: "Retrieve relevant knowledge cards"
```

### Qwen3-Embedding local (CPU, GGUF)

```yaml
embed:
  backend: local
  model_path: ~/.knowcard/models/Qwen3-Embedding-0.6B-GGUF/qwen3-embedding-0.6b.Q8_0.gguf
  pooling: last                      # Qwen models use last-token pooling
  context_size: 2048
  batch_size: 512
```

### Qwen3-VL-Embedding local (CPU, GGUF, text-only)

```yaml
embed:
  backend: local
  model_path: ~/.knowcard/models/Qwen3-VL-Embedding-2B-Q4_K_M.gguf
  pooling: last
  context_size: 8192
```

### Qwen text-embedding-v4 via DashScope (text-only)

```yaml
embed:
  backend: qwen_cloud
  model: text-embedding-v4
  api_key: sk-xxx
  dimensions: 1024                   # MRL: 64-2048
```

### Other backends

```yaml
# Local bge-m3
embed:
  backend: local
  model_path: ~/.knowcard/models/bge-m3.Q8_0.gguf
  pooling: mean

# Ollama
embed:
  backend: ollama
  model: nomic-embed-text
```

### Local model setup guide

The `local` backend uses [yzma](https://github.com/hybridgroup/yzma) to call llama.cpp shared libraries via FFI (no CGo). You need to install the shared libraries and download a GGUF model before configuring the backend.

#### Step 1: Install libllama shared libraries

Download pre-built binaries from [llama.cpp releases](https://github.com/ggml-org/llama.cpp/releases):

```bash
# Check the latest version
curl -s https://hybridgroup.github.io/llama-cpp-builder/version.json
# {"tag_name":"b9975"}

# Download and extract (Linux x64 CPU example)
mkdir -p ~/.knowcard/lib
download https://github.com/ggml-org/llama.cpp/releases/download/b9975/llama-b9975-bin-ubuntu-x64.tar.gz
tar xzf llama-b9975-bin-ubuntu-x64.tar.gz -C /tmp/llama-extract

# Copy all .so files (and symlinks) into a flat directory
cp /tmp/llama-extract/llama-b9975/lib*.so* ~/.knowcard/lib/
```

The lib directory must contain these files (symlinks resolved):

| File | Required by |
|---|---|
| `libggml.so` | GGML core |
| `libggml-base.so` | GGML base |
| `libllama.so` | Llama model loading, inference |
| `libmtmd.so` | Multimodal support |

#### Step 2: Download a GGUF embedding model

Two recommended small models (both ~600 MB, 1024-dim):

```bash
mkdir -p ~/.knowcard/models

# Qwen3-Embedding-0.6B (pooling: last)
# From: https://huggingface.co/Qwen/Qwen3-Embedding-0.6B-GGUF
# File: Qwen3-Embedding-0.6B-Q8_0.gguf (~610 MB)

# bge-m3 (pooling: mean)
# From: https://huggingface.co/gpustack/bge-m3-GGUF
# File: bge-m3-Q8_0.gguf (~606 MB)
```

#### Step 3: Write the config

**Qwen3-Embedding-0.6B:**

```yaml
embed:
  backend: local
  model_path: ~/.knowcard/models/Qwen3-Embedding-0.6B-Q8_0.gguf
  lib_path: ~/.knowcard/lib           # directory containing libllama.so
  pooling: last                       # Qwen models use last-token pooling
  context_size: 2048
  batch_size: 512
```

**bge-m3:**

```yaml
embed:
  backend: local
  model_path: ~/.knowcard/models/bge-m3-Q8_0.gguf
  lib_path: ~/.knowcard/lib           # directory containing libllama.so
  pooling: mean                       # bge-m3 requires mean pooling
  context_size: 2048
  batch_size: 512
```

#### Config field reference

| Field | Description | Default |
|---|---|---|
| `backend` | Set to `local` for yzma/llama.cpp | `local` |
| `model_path` | Absolute path to the GGUF model file | *required* |
| `lib_path` | **Directory** containing `libllama.so` (not the file itself) | `"llama"` (system linker) |
| `pooling` | Pooling strategy: `last` (Qwen), `mean` (bge-m3), `cls` | `last` |
| `context_size` | Model context window size | `2048` |
| `batch_size` | Batch processing size | `512` |

> **Pooling matters**: Qwen embedding models use `last`-token pooling. bge-m3 requires `mean` pooling. Setting the wrong pooling type will produce nil embeddings or poor search quality.

> **lib_path is a directory**: yzma internally does `filepath.Join(lib_path, "libllama.so")`. Do not point it at the `.so` file directly.

> **LD_LIBRARY_PATH**: If the `.so` dependency chain is not in a system path, set `export LD_LIBRARY_PATH=~/.knowcard/lib` before running knowcard.

## Supported Embedding Models

### Qwen Models (recommended)

| Model | Mode | Backend | Notes |
|---|---|---|---|
| Qwen3-VL-Embedding-2B/8B | Multimodal (text+image+video) | `local` (GGUF) or `qwen_cloud` | 33 languages, MRL dims, latest Qwen VL embed |
| Qwen3-Embedding-0.6B/4B/8B | Text-only | `local` (GGUF) or `qwen_cloud` | #1 MTEB multilingual, 100+ languages |
| text-embedding-v4 | Text-only | `qwen_cloud` | DashScope cloud API, MRL dims |
| tongyi-embedding-vision-plus | Multimodal | `qwen_cloud` | DashScope cloud, text+image+video |

### Other Models

| Model | Backend | Notes |
|---|---|---|
| bge-m3 | `local` | Chinese/English bilingual |
| nomic-embed-text | `ollama` | Via Ollama service |
| text-embedding-3-small | `openai` | Via OpenAI API |
| Any OpenAI-compatible | `custom` | Configurable base URL |

## Tech Stack

| Component | Technology |
|---|---|
| Embedding engine | [yzma](https://github.com/hybridgroup/yzma) (llama.cpp, purego, no CGo) |
| Vector DB | [chromem-go](https://github.com/philippgille/chromem-go) |
| Keyword search | Custom BM25 with CJK bigram tokenizer |
| Versioning | [go-git](https://github.com/go-git/go-git) with separated VCS directory |
| MCP protocol | [mcp-go](https://github.com/mark3labs/mcp-go) |

## License

MIT
