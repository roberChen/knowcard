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
~/.knowcard/
├── cards/              Knowledge cards as .md files (semantic tree structure)
├── _vcs/               Git metadata (separated from cards, CLI-invisible)
├── index/
│   └── chromem/        Vector index (derived, rebuildable)
└── models/             GGUF embedding model cache
```

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
# Initialize
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

```yaml
# ~/.knowcard/knowcard.yaml
root: ~/.knowcard
embed:
  backend: local          # local | ollama | openai | custom
  model_path: ~/.knowcard/models/bge-m3.Q8_0.gguf
  lib_path: ""            # path to libllama (empty = system default)
  context_size: 2048
  batch_size: 512
rrf_k: 60
candidate_pool: 30
```

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
