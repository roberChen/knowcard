# AGENTS.md

Guide for AI agents working in the knowcard codebase.

## What This Is

knowcard is a local agent memory system: knowledge stored as markdown "cards" with YAML front matter, indexed for hybrid vector + keyword search, and versioned with an embedded git repository. It ships as a single Go binary usable three ways: Go library, CLI tool, and MCP server (stdio).

## Commands

```bash
go build ./...                    # build all packages
go build -o knowcard ./cmd/knowcard  # build CLI binary
go test ./...                     # run all tests
go test ./search/... ./card/...   # run specific package tests
go vet ./...                      # lint (passes clean)
```

No Makefile, CI config, or external linter. The module is `github.com/robert/knowcard` (local/private, not published).

## Architecture

### Package layout and dependency flow

```
cmd/knowcard  ──►  knowcard (root)  ──►  card, embed, search
mcp            ──►  knowcard (root)  ──►  card
```

- **`knowcard` (root package)**: The `Store` orchestrator (`store.go` + `store_ops.go`). Owns the chromem-go vector DB, in-memory BM25 index, go-git repo, and the id-to-path map. All CRUD operations go through `Store`. Config lives here too (`config.go`).
- **`card`**: `Card` struct, YAML front matter + markdown body parsing/serialization (`parse.go`), path validation, file I/O.
- **`embed`**: `Embedder` interface with 5 backends: `local` (yzma/llama.cpp via purego, no CGo), `api` (OpenAI-compatible), `qwen_cloud` (DashScope native), `siliconflow`. Factory in `embed.go`.
- **`search`**: In-memory BM25 with CJK-aware tokenizer (`tokenizer.go`), RRF fusion (`fuse.go`).
- **`mcp`**: MCP server wrapping `Store` with 4 tools (`recall`, `get_cards`, `upsert_card`, `delete_card`). Tool descriptions and server instructions are embedded as string constants in `server.go`.
- **`cmd/knowcard`**: CLI entry point. Manual subcommand dispatch via `switch` on `os.Args[1]` using the standard `flag` package per subcommand.

### Data flow for search (Recall)

1. Query is embedded via the configured backend
2. **Semantic lane**: chromem-go `Query` returns top-N by cosine similarity
3. **Keyword lane**: BM25 scores the query against all indexed card text
4. **RRF fusion**: `1/(k+rank)` scoring combines both lanes (default k=60)
5. Results filtered by path prefix / tags, truncated to TopK

### Data directory layout (`~/.knowcard/`)

```
cards/           markdown files (source of truth, semantic tree structure)
_vcs/            git metadata (SEPARATED from cards dir — see gotcha)
index/chromem/   persistent vector index (derived, rebuildable via Store.Rebuild)
models/          GGUF model cache for local embedding
manifest.json    integrity checkpoint (head commit, card count)
knowcard.yaml    config file
```

## Key Design Decisions and Gotchas

### Separated VCS directory

Git metadata lives in `_vcs/`, not `.git/` inside `cards/`. `openOrInitRepo()` in `store.go` deliberately removes the `.git` pointer file from the worktree after init. This means **standard `git` CLI commands run inside `cards/` will report "not a git repository"** — this is intentional. go-git accesses the repo through the explicit storer path. This is tested in `TestStore_StandardGitNotVisible`.

### Dual index systems that must stay in sync

Every card mutation must update BOTH indexes:
- **chromem-go** (persistent, disk-backed) — semantic vector search
- **BM25** (in-memory only, rebuilt on every startup) — keyword search

BM25 is NOT persisted. On startup, `loadIndex()` rebuilds it from card files. If you add a new mutation path, you must update both `addToVectorIndex`/`deleteFromVectorIndex` AND `bm25.AddDocument`/`RemoveDocument`. The pattern is visible in `UpsertCard`, `DeleteCard`, `MoveCard`.

### chromem-go Query requires nResults <= Count

`Recall()` clamps `CandidatePool` to `col.Count()` because chromem-go panics/errors if you request more results than documents exist. Always check `s.col.Count()` before querying.

### Config field gap: EnableFusion

`embed.Config` has an `EnableFusion` field (for qwen3-vl-embedding), but `knowcard.EmbedConfig` does NOT have this field, and `Open()` in `store.go` does NOT pass it through to `embed.New()`. The README documents `enable_fusion: true` in YAML config, but this currently cannot be set via the config file. If you need fusion, either add the field to `EmbedConfig` + wire it in `Open()`, or construct the embedder directly.

### API embedders probe on first Dim() call

`APIEmbedder.Dim()`, `DashScopeEmbedder.Dim()`, and `SiliconFlowEmbedder.Dim()` will make a real network call ("dimension probe") if the dimension hasn't been determined yet. Opening a `Store` with an API backend may trigger network I/O during initialization.

### Embedding content composition

`addToVectorIndex()` embeds a composite string: `Title + Keywords + Summary + Body` — not just the body. BM25 indexes the same concatenation. This affects search relevance behavior.

### Card Path vs ID

- **ID**: 32-char hex, immutable, primary key, stored in YAML front matter. Generated by `card.NewID()`.
- **Path**: semantic relative location (e.g. `programming/go/escape-analysis`), NOT stored in YAML — it's derived from the file's location on disk. Has a strict regex validator (`[a-zA-Z0-9][a-zA-Z0-9\-_./]*`), no spaces, no leading/trailing slashes, no `..`, no CJK characters.

### Default pooling is "last"

`embed.New()` defaults to `"last"` pooling for local models (correct for Qwen embedding models). For bge-m3, you must explicitly set `pooling: mean` in config.

### Body token limit

`card.MaxBodyTokens = 4096`. Validation uses the real tokenizer if the embedder implements `embed.TokenCounter` (only `LocalEmbedder` does). Otherwise falls back to a char heuristic (`len(body)/3`). API-based embedders always use the heuristic.

### CJK tokenizer

The BM25 tokenizer does NOT use word segmentation dictionaries. CJK text is split into character unigrams + bigrams. Latin text is lowercased and split on non-alphanumeric. This is a proven technique for CJK IR without dictionaries. Affects how Chinese/Japanese/Korean queries match.

### Tags encoding in chromem-go metadata

Tags are stored as `tag_0`, `tag_1`, etc. in chromem-go metadata (positional, lossy). Tag filtering in `Recall()` is done in Go after retrieval, not pushed down to chromem-go.

### Config env var expansion

Config YAML supports `${VAR}` syntax — all string fields are passed through `os.ExpandEnv` before YAML parsing in `config.go:Load()`.

## Testing Patterns

- **Store tests** (`store_test.go`): Use `OpenWithEmbedder()` with a `fakeEmbedder` that produces deterministic hash-based vectors (no model needed). The fake embedder is defined inline in the test file. Uses `t.TempDir()` for isolation.
- **API embedder tests** (`dashscope_test.go`, `siliconflow_test.go`): Use `httptest.NewServer` to mock API endpoints. Construct embedder structs directly with the mock server URL.
- **Card/search tests**: Pure unit tests, no I/O.
- All tests use `t.Helper()`, `t.TempDir()`, and standard Go testing conventions.
- Test data includes Chinese text (CJK path coverage).

## Conventions

- Standard library `flag` package for CLI — no cobra/urfave/cli.
- Error wrapping with `fmt.Errorf("context: %w", err)` throughout.
- Compile-time interface checks: `var _ Embedder = (*SomeEmbedder)(nil)`.
- Exported types have doc comments; internal helpers do not.
- Commit messages from the store use prefixes: `upsert:`, `delete:`, `move:`.
- The `move` commit message uses a Unicode arrow (`→`).
- Config struct uses YAML tags matching the documented config keys.
- Module requires Go 1.26.4.
