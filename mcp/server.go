package mcp

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"

	"github.com/mark3labs/mcp-go/mcp"
	"github.com/mark3labs/mcp-go/server"

	kc "github.com/robert/knowcard"
	"github.com/robert/knowcard/card"
)

const serverInstructions = `# knowcard — Agent Memory System

This server is the project's local wiki and knowledge base. It stores knowledge as "cards" (self-contained markdown notes with structured metadata), indexed for hybrid vector + keyword search. Think of it as the first place to look when you need to understand something about this project or its domain, and the place to write back when you learn something new.

## CRITICAL: The Recall-Record Cycle

When a task involves the project or its domain, your work follows a two-phase cycle: **recall first, record after.** If the task has nothing to do with the project or domain (e.g., general chat, standard library syntax, universal knowledge), skip both phases entirely — no recall needed, no recording expected.

### Phase 1 — Recall (when you need project/domain knowledge)

1. **RECALL FIRST** — When a task requires understanding anything about this project or its domain, call recall with relevant keywords BEFORE doing other work (code search, web search, file reading, etc.). The knowledge base is the project's local wiki — it may already contain the answer, architecture details, debugging solutions, conventions, or context you need. Consult it first so you don't reinvent the wheel.
   - If the task is entirely unrelated to the project/domain, skip this phase.
2. **JUDGE RELEVANCE** — After recall returns results, explicitly state whether the results are highly relevant, partially relevant, or not relevant to the current task. Judge relevance by reading each card's title and summary and considering how directly it addresses your task — do NOT rely on the score value, which is relative and varies widely across queries and index sizes. Do NOT silently skip to the next step.
   - **Highly relevant** (the summary clearly describes exactly your topic): call get_cards to read the full content before investigating.
   - **Partially relevant** (the summary touches the topic but isn't directly about it): call get_cards on the best 1–2 candidates, but proceed to investigation in parallel.
   - **Not relevant** (the results are about unrelated topics despite any score): explicitly state "recall returned no relevant cards" and proceed to investigation from scratch.
3. **GET_CARDS** — If recall found relevant cards (high or partial), read their full content by passing the card IDs.
4. **WORK** — Continue with whatever the task requires: code search, web search, file editing, investigation, etc. Use what you learned from cards as your starting point, and fill gaps with your own work.

### Phase 2 — Record (when you discovered project/domain knowledge)

After completing your work, reflect on whether you discovered or learned new project/domain knowledge that isn't already in the wiki. If so, record it promptly — keeping the wiki up to date is part of completing your task. New knowledge worth recording includes:
- Code details: function signatures, file:line references, architecture patterns, data flow
- Specific constants, thresholds, config values, or limits
- Business logic, behavioral flows, design decisions, or conventions
- Debugging solutions, gotchas, or pitfalls encountered
- Domain knowledge or conceptual insights relevant to the project
- Anything that contradicts, refines, or extends what existing cards say
- Any topic with no existing card coverage after a recall search

**If new knowledge was discovered, you MUST complete these steps BEFORE composing your final response:**

5. **RECALL BEFORE UPSERT** — Call recall with keywords related to the new knowledge to find existing cards that should be updated. Do NOT create a card without checking first.
6. **DECIDE: UPDATE OR CREATE** — Explicitly state your decision and reasoning:
   - **UPDATE** (existing card found): "Updating card <id> because <what changed>" — pass the existing card's id to upsert_card.
   - **CREATE** (no relevant card found): "Creating new card because <why this topic is distinct and no existing card covers it>" — omit id in upsert_card.
7. **UPSERT** — Execute the upsert_card call.

### When the Cycle Does Not Apply

Skip both phases entirely when the task is unrelated to the project or domain — e.g., general conversation, universal programming knowledge, standard library syntax. In these cases there is nothing to recall and nothing worth recording.

Phase 1 (recall) always applies when the task touches the project or domain. Phase 2 (record) can additionally be skipped if your work discovered nothing new beyond what existing cards already cover.

**If you decide to skip Phase 2, you MUST explicitly state why.** For example: "Not recording — this is general knowledge about standard library usage, not specific to this project." Do not silently skip.

**When in doubt, record.** A slightly redundant card is far better than lost knowledge.

### Pre-Response Checklist

Before sending your final reply to the user, verify:
- If the task involved the project/domain: did I recall before doing other work?
- Did my work discover new project/domain knowledge not already in the wiki?
- If yes: did I recall for duplicates, decide update/create, and execute upsert_card?
- If no new knowledge: did I explicitly state why I am not recording?

## When to Use Each Tool

- **recall**: When you need project/domain knowledge. Always before other work (code search, web search, file reading). This is the project's local wiki — consult it first.
- **get_cards**: After recall returns relevant results.
- **upsert_card**: When you discovered new project/domain knowledge. Keep the wiki up to date — recording is part of completing your task, not an optional follow-up.
- **delete_card**: When a card is outdated, incorrect, or no longer relevant.
- **init**: When no knowledge base exists for the current directory.

## When to UPDATE a Card

Pass the existing card's id to upsert_card when you discover ANY of:
- Specific constants, numbers, or thresholds not in the card (buffer sizes, timeout values, config keys)
- Implementation details not captured (function signatures, file:line references, data flow)
- Business logic or behavioral flow the card doesn't describe
- Behavior that contradicts, refines, or extends the card's current description
- Missing context that would help a future reader act on the knowledge

## When to CREATE a New Card

- No existing card covers the topic after a recall search
- The topic is distinct enough to warrant a separate card (don't mix unrelated topics)

## Common Mistakes (AVOID)

- Doing code search, web search, or file reading before consulting the wiki via recall
- Doing fresh work without reading existing relevant cards first
- Getting recall results and silently skipping get_cards — you MUST explicitly judge relevance and act on it
- **Completing your work, then sending your response WITHOUT recording new knowledge to the wiki**
- **Treating recording as optional or "when the user asks"** — updating the wiki IS part of completing your task
- Discovering new project/domain details but only recording them when explicitly asked
- Creating duplicate cards instead of updating existing ones (always pass the existing card id)
- Calling upsert_card without first calling recall to check for existing cards
- Calling upsert_card without explicitly stating whether it's an update or create, and why

## Path Convention

Use forward-slash hierarchy with lowercase-with-hyphens:
  - programming/go/goroutine-scheduler
  - databases/postgres/query-tuning
  - devops/docker/multi-stage-builds
  - concepts/cap-theorem

## What Makes a Good Card

- Self-contained: a future reader should understand it without external context
- Focused: one topic per card, not a dump of everything
- Actionable: includes specifics (code snippets, commands, config) not just theory
- Concise: keep the body under 4096 tokens. If content is too long, shorten the body to a summary and attach the full version as a reference file
- Summarizable: the summary field should let you decide if the card is worth reading`

// Server wraps the knowcard Store as an MCP server.
// The store is opened eagerly in NewServer by searching for .knowcard/
// upward from the process's working directory. If no knowledge base is
// found, the server still starts — tool calls will fail with a hint to
// use the init tool.
type Server struct {
	cfg   kc.Config
	mu    sync.Mutex
	store *kc.Store
}

// NewServer creates an MCP server with the given global config.
// It eagerly attempts to find and open a knowledge base from the current
// working directory. If none is found, the server starts without a store.
func NewServer(cfg kc.Config) *Server {
	s := &Server{cfg: cfg}
	s.tryOpenStore()
	return s
}

// tryOpenStore searches for .knowcard/ upward from CWD and opens the store
// if found. Failures are silent — the server can still start without a store.
func (s *Server) tryOpenStore() {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.store != nil {
		return
	}
	cwd, err := os.Getwd()
	if err != nil {
		return
	}
	root, err := kc.FindRoot(cwd)
	if err != nil {
		return
	}
	cfg := s.cfg
	cfg.Root = root
	store, err := kc.Open(cfg)
	if err != nil {
		return
	}
	s.store = store
}

// store returns the opened store, or an error if no knowledge base is
// available.
func (s *Server) getStore() (*kc.Store, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.store != nil {
		return s.store, nil
	}
	return nil, fmt.Errorf("no knowledge base found for the current directory")
}

// Serve starts the MCP server over stdio.
func (s *Server) Serve() error {
	mcpSrv := server.NewMCPServer("knowcard", "0.1.0",
		server.WithToolCapabilities(false),
		server.WithInstructions(serverInstructions),
	)

	mcpSrv.AddTool(s.recallTool(), s.handleRecall)
	mcpSrv.AddTool(s.getCardsTool(), s.handleGetCards)
	mcpSrv.AddTool(s.upsertCardTool(), s.handleUpsertCard)
	mcpSrv.AddTool(s.deleteCardTool(), s.handleDeleteCard)
	mcpSrv.AddTool(s.initTool(), s.handleInit)

	return server.ServeStdio(mcpSrv)
}

// noKnowledgeBaseResult returns an error result instructing the agent
// to call the init tool.
func noKnowledgeBaseResult() *mcp.CallToolResult {
	return mcp.NewToolResultError(
		"No knowledge base (.knowcard/) found for the current directory.\n" +
			"Use the 'init' tool to create one in the current directory, " +
			"then retry your request.",
	)
}

// --- recall ---

func (s *Server) recallTool() mcp.Tool {
	return mcp.NewTool("recall",
		mcp.WithDescription(`Search the knowledge card memory system using hybrid vector + keyword search.

Search the project's local wiki using hybrid vector + keyword search.

Returns matching cards with title, summary, relevance score, and hit type. Call this when you need to understand something about the project or its domain — BEFORE code search, web search, file reading, or any other work. The wiki may already contain the answer, architecture details, debugging solutions, conventions, or context you need.

The "hit_type" field indicates how the card matched: "semantic" (meaning-based), "keyword" (exact term match), or "both" (strongest match).

**After recall, you MUST explicitly judge relevance based on the card's title, summary, and how directly it addresses your task — NOT on the score value.** Scores are relative and vary widely across queries and index sizes, so fixed score thresholds are unreliable. Use your own judgment reading the content instead:
- Highly relevant (the summary clearly describes exactly your topic): call get_cards to read full content.
- Partially relevant (the summary touches but doesn't directly address your topic): call get_cards on the best 1-2 candidates.
- Not relevant (the results are about unrelated topics despite any score): explicitly state "recall returned no relevant cards" and proceed without get_cards.

Do NOT silently skip to the next step without stating your relevance judgment.`),
		mcp.WithString("query",
			mcp.Required(),
			mcp.Description("Search query — keywords or a natural language sentence in any language (Chinese/English both work)"),
		),
		mcp.WithNumber("top_k",
			mcp.Description("Maximum number of results (default: 10). Use 3-5 for focused lookups, 10-20 for broad exploration"),
		),
		mcp.WithString("path_prefix",
			mcp.Description("Filter results to a specific knowledge branch, e.g. 'programming/go' or 'databases'"),
		),
	)
}

func (s *Server) handleRecall(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	store, err := s.getStore()
	if err != nil {
		return noKnowledgeBaseResult(), nil
	}

	query, err := req.RequireString("query")
	if err != nil {
		return mcp.NewToolResultError("query is required"), nil
	}

	opts := kc.RecallOpts{TopK: 10}
	if k, err := req.RequireInt("top_k"); err == nil && k > 0 {
		opts.TopK = int(k)
	}
	if pp := req.GetArguments()["path_prefix"]; pp != nil {
		if ppStr, ok := pp.(string); ok {
			opts.PathPref = ppStr
		}
	}

	results, err := store.Recall(query, opts)
	if err != nil {
		return mcp.NewToolResultError(fmt.Sprintf("recall failed: %v", err)), nil
	}

	var sb strings.Builder
	if len(results) == 0 {
		sb.WriteString("No matching cards found. This means either:\n")
		sb.WriteString("- No relevant knowledge has been stored yet — consider creating a card with upsert_card\n")
		sb.WriteString("- Try different keywords or rephrase the query")
	} else {
		sb.WriteString(fmt.Sprintf("Found %d card(s). Use get_cards to read full content of any card by its ID.\n\n", len(results)))
		for i, r := range results {
			sb.WriteString(fmt.Sprintf("%d. **%s**\n", i+1, r.Title))
			sb.WriteString(fmt.Sprintf("   - id: `%s`\n", r.ID))
			sb.WriteString(fmt.Sprintf("   - path: `%s`\n", r.Path))
			sb.WriteString(fmt.Sprintf("   - score: %.4f (%s)\n", r.Score, r.HitType))
			sb.WriteString(fmt.Sprintf("   - summary: %s\n\n", r.Summary))
		}
	}

	return mcp.NewToolResultText(sb.String()), nil
}

// --- get_cards ---

func (s *Server) getCardsTool() mcp.Tool {
	return mcp.NewTool("get_cards",
		mcp.WithDescription(`Retrieve the full content of knowledge cards by their IDs.

Each card is returned in full, including title, keywords, summary, body (markdown), and reference links. Pass IDs obtained from the recall tool.

**Only call get_cards when recall returned relevant results.** If recall returned no relevant results, do not call this tool — state that explicitly and proceed with your own work.

**Workflow**: recall → judge relevance → get_cards (only if relevant) → proceed with task.`),
		mcp.WithArray("ids",
			mcp.Required(),
			mcp.Description("Card IDs to retrieve (obtain these from recall results)"),
			mcp.Items(map[string]interface{}{"type": "string"}),
		),
	)
}

func (s *Server) handleGetCards(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	store, err := s.getStore()
	if err != nil {
		return noKnowledgeBaseResult(), nil
	}

	args := req.GetArguments()
	idsVal, ok := args["ids"]
	if !ok {
		return mcp.NewToolResultError("ids is required"), nil
	}

	var ids []string
	switch v := idsVal.(type) {
	case []interface{}:
		ids = make([]string, 0, len(v))
		for _, item := range v {
			if str, ok := item.(string); ok && str != "" {
				ids = append(ids, str)
			}
		}
	case string:
		if v != "" {
			ids = []string{v}
		}
	default:
		return mcp.NewToolResultError("ids must be an array of card ID strings"), nil
	}

	if len(ids) == 0 {
		return mcp.NewToolResultError("ids must contain at least one card ID"), nil
	}

	cards, err := store.GetCards(ids)
	if err != nil {
		return mcp.NewToolResultError(fmt.Sprintf("get cards failed: %v", err)), nil
	}

	var sb strings.Builder
	for i, c := range cards {
		if i > 0 {
			sb.WriteString("\n---\n\n")
		}
		// Resolve reference to absolute path for display
		if c.Reference != "" {
			c.Reference = filepath.Join(store.CardsDir(), c.Reference)
		}
		content, _ := card.Serialize(&c)
		sb.WriteString(content)
		sb.WriteString("\n")
	}

	if len(cards) == 0 {
		sb.WriteString("No cards found for the given IDs. Use recall to find valid card IDs.")
	}

	return mcp.NewToolResultText(sb.String()), nil
}

// --- upsert_card ---

func (s *Server) upsertCardTool() mcp.Tool {
	return mcp.NewTool("upsert_card",
		mcp.WithDescription(`Create or update a knowledge card. This is how the memory system learns.

**IMPORTANT — When to call this tool**: When your work discovered new project/domain knowledge that isn't already in the wiki — code details, architecture patterns, constants, behavioral flows, debugging solutions, conventions, domain insights. Recording it is part of completing your task, NOT an optional follow-up. When in doubt, record.

**Before calling this tool, you MUST**:
1. Call recall with relevant keywords to check if a similar card already exists.
2. Explicitly state your decision and reasoning:
   - If updating: "Updating card <id> because <what changed>" — pass the existing card's id.
   - If creating: "Creating new card because <why this topic is distinct and uncovered>" — omit id.
3. Then execute the upsert_card call.

**When to create**: you learned something worth remembering — code patterns, architectural decisions, debugging solutions, project conventions, domain knowledge, tool configurations, etc.

**When to update**: a card's information is incomplete or outdated and you have better/newer information.

**Card structure**:
- path: semantic location in the knowledge tree (e.g. 'programming/go/memory-escape')
- title: concise, descriptive (what is this card about?)
- summary: 1-2 sentences answering "what will I learn from this card?"
- body: the actual knowledge, in markdown — keep it concise (max 4096 tokens). If the body is too long, the upsert will fail with guidance to shorten it.
- keywords: terms someone would search for to find this card
- reference: path to a local file containing detailed documentation (optional). The file is copied into the knowledge base and version-controlled. Use this for content that doesn't fit in the body — full design docs, lengthy code samples, detailed configurations. If the file does not exist, upsert will fail.`),
		mcp.WithString("path",
			mcp.Required(),
			mcp.Description("Semantic path in the knowledge tree. Use lowercase-with-hyphens and / for hierarchy. Examples: 'programming/go/goroutine-scheduler', 'databases/redis/persistence', 'concepts/cap-theorem'. Choose a path that groups related cards together."),
		),
		mcp.WithString("title",
			mcp.Required(),
			mcp.Description("Card title — concise and descriptive, should clearly state what topic the card covers"),
		),
		mcp.WithString("summary",
			mcp.Required(),
			mcp.Description("1-2 sentence summary answering 'what will I learn from reading this card?'. Used for search matching and for deciding whether to read the full card."),
		),
		mcp.WithString("body",
			mcp.Required(),
			mcp.Description("Full card content in markdown. Include code blocks, commands, diagrams (text), and explanations. Keep UNDER 4096 tokens — if the body exceeds this limit, upsert will fail and prompt you to shorten the body and move detailed content to a reference file. Be concise and self-contained."),
		),
		mcp.WithArray("keywords",
			mcp.Description("Keywords that someone might search for to find this card. Include both Chinese and English terms if relevant."),
			mcp.Items(map[string]interface{}{"type": "string"}),
		),
		mcp.WithString("reference",
			mcp.Description("Path to a local file to attach as a reference document (e.g. '/home/user/project/docs/design.md'). The file is copied into the knowledge base, version-controlled, and its absolute path is shown when reading the card. Use this for detailed content that doesn't fit in the body. The file MUST exist — if it doesn't, upsert will fail."),
		),
		mcp.WithString("id",
			mcp.Description("To UPDATE an existing card, pass its ID here. Omit this field to create a new card."),
		),
	)
}

func (s *Server) handleUpsertCard(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	store, err := s.getStore()
	if err != nil {
		return noKnowledgeBaseResult(), nil
	}

	c := &card.Card{}

	c.Path, _ = req.GetArguments()["path"].(string)
	c.Title, _ = req.GetArguments()["title"].(string)
	c.Summary, _ = req.GetArguments()["summary"].(string)
	c.Body, _ = req.GetArguments()["body"].(string)
	c.Reference, _ = req.GetArguments()["reference"].(string)

	if id, ok := req.GetArguments()["id"].(string); ok && id != "" {
		c.ID = id
	} else {
		c.ID = card.NewID()
	}

	// Keywords
	if kw, ok := req.GetArguments()["keywords"]; ok {
		switch v := kw.(type) {
		case []interface{}:
			for _, item := range v {
				if str, ok := item.(string); ok {
					c.Keywords = append(c.Keywords, str)
				}
			}
		case string:
			c.Keywords = strings.Split(v, ",")
		}
	}

	if err := store.UpsertCard(c); err != nil {
		return mcp.NewToolResultError(fmt.Sprintf("upsert failed: %v", err)), nil
	}

	result := map[string]interface{}{
		"status": "saved",
		"id":     c.ID,
		"path":   c.Path,
		"title":  c.Title,
	}
	data, _ := json.MarshalIndent(result, "", "  ")
	return mcp.NewToolResultText(fmt.Sprintf("Card saved successfully:\n%s", string(data))), nil
}

// --- delete_card ---

func (s *Server) deleteCardTool() mcp.Tool {
	return mcp.NewTool("delete_card",
		mcp.WithDescription(`Delete a knowledge card permanently. Use with caution — deleted knowledge cannot be recovered through the MCP interface (though git history is preserved internally).

Only delete cards that are incorrect, outdated beyond repair, or completely irrelevant. Prefer upsert_card (update) over delete when possible.`),
		mcp.WithString("id",
			mcp.Required(),
			mcp.Description("ID of the card to delete (obtain from recall results)"),
		),
	)
}

func (s *Server) handleDeleteCard(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	store, err := s.getStore()
	if err != nil {
		return noKnowledgeBaseResult(), nil
	}

	id, err := req.RequireString("id")
	if err != nil {
		return mcp.NewToolResultError("id is required"), nil
	}

	if err := store.DeleteCard(id); err != nil {
		return mcp.NewToolResultError(fmt.Sprintf("delete failed: %v", err)), nil
	}

	return mcp.NewToolResultText(fmt.Sprintf("Card %s has been deleted.", id)), nil
}

// --- init ---

func (s *Server) initTool() mcp.Tool {
	return mcp.NewTool("init",
		mcp.WithDescription(`Initialize a knowledge base in the current directory.

This creates a .knowcard/ directory with the required subdirectories (cards/, _vcs/, index/). Call this when no knowledge base exists for the current project.

After init succeeds, recall/upsert_card/get_cards/delete_card become available for this directory.`),
		mcp.WithString("dir",
			mcp.Description("Directory to initialize the knowledge base in (default: current working directory)"),
		),
	)
}

func (s *Server) handleInit(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	dir, _ := req.GetArguments()["dir"].(string)
	if dir == "" {
		var err error
		dir, err = os.Getwd()
		if err != nil {
			return mcp.NewToolResultError(fmt.Sprintf("getting current directory: %v", err)), nil
		}
	}

	absDir, err := filepath.Abs(dir)
	if err != nil {
		return mcp.NewToolResultError(fmt.Sprintf("resolving path: %v", err)), nil
	}

	kcDir := filepath.Join(absDir, kc.DirName)
	if _, err := os.Stat(kcDir); err == nil {
		return mcp.NewToolResultText(fmt.Sprintf("Knowledge base already exists at %s", kcDir)), nil
	}

	cfg := s.cfg
	cfg.Root = kcDir
	if err := cfg.EnsureDirs(); err != nil {
		return mcp.NewToolResultError(fmt.Sprintf("creating knowledge base: %v", err)), nil
	}

	store, err := kc.Open(cfg)
	if err != nil {
		return mcp.NewToolResultError(fmt.Sprintf("opening store: %v", err)), nil
	}

	s.mu.Lock()
	if s.store != nil {
		s.store.Close()
	}
	s.store = store
	s.mu.Unlock()

	return mcp.NewToolResultText(fmt.Sprintf("Initialized knowledge base at %s", kcDir)), nil
}
