package mcp

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/mark3labs/mcp-go/mcp"
	"github.com/mark3labs/mcp-go/server"

	kc "github.com/robert/knowcard"
	"github.com/robert/knowcard/card"
)

const serverInstructions = `# knowcard — Agent Memory System

This server provides a persistent knowledge memory for the current project. Use it to store, search, and manage knowledge cards (self-contained markdown notes with structured metadata).

## Core Workflow

1. **recall** — When you encounter domain-specific knowledge or a question about a topic, search first with keywords or a natural language query. This returns card summaries (not full content).
2. **get_cards** — If recall found relevant cards, read their full content by passing the card IDs.
3. **upsert_card** — After solving a problem or learning something worth remembering, save it as a new card OR update an existing card (pass the id to update).

## When to Use Each Tool

- **recall**: Before answering a question, check if relevant knowledge already exists. Also use it to check for duplicates before creating a new card.
- **get_cards**: After recall gives you relevant card IDs, use this to read full details.
- **upsert_card**: When you learn something new that would be useful in future conversations — code patterns, architectural decisions, debugging solutions, project conventions, domain knowledge, etc. ALWAYS call recall first to check if a similar card exists; update rather than duplicate.
- **delete_card**: When a card is outdated, incorrect, or no longer relevant.

## Path Convention

Paths form a semantic knowledge tree. Use forward-slash hierarchy:
  - programming/go/goroutine-scheduler
  - databases/postgres/query-tuning
  - devops/docker/multi-stage-builds
  - concepts/cap-theorem

Choose paths that group related knowledge together. Use lowercase with hyphens.

## What Makes a Good Card

- Self-contained: a future reader should understand it without external context
- Focused: one topic per card, not a dump of everything
- Actionable: includes specifics (code snippets, commands, config) not just theory
- Summarizable: the summary field should let you decide if the card is worth reading`

// Server wraps the knowcard Store as an MCP server.
type Server struct {
	store *kc.Store
}

// NewServer creates an MCP server backed by the given Store.
func NewServer(s *kc.Store) *Server {
	return &Server{store: s}
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

	return server.ServeStdio(mcpSrv)
}

// --- recall ---

func (s *Server) recallTool() mcp.Tool {
	return mcp.NewTool("recall",
		mcp.WithDescription(`Search the knowledge card memory system using hybrid vector + keyword search.

Returns matching cards with title, summary, relevance score, and hit type. Use this to:
- Check if knowledge about a topic already exists before answering or creating a card
- Find relevant cards to read in full (via get_cards)

The "hit_type" field indicates how the card matched: "semantic" (meaning-based), "keyword" (exact term match), or "both" (strongest match).

**Workflow**: Call recall first → review summaries → use get_cards on selected IDs to read full content.`),
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

	results, err := s.store.Recall(query, opts)
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

**Workflow**: recall → get_cards (selected IDs).`),
		mcp.WithArray("ids",
			mcp.Required(),
			mcp.Description("Card IDs to retrieve (obtain these from recall results)"),
			mcp.Items(map[string]interface{}{"type": "string"}),
		),
	)
}

func (s *Server) handleGetCards(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
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

	cards, err := s.store.GetCards(ids)
	if err != nil {
		return mcp.NewToolResultError(fmt.Sprintf("get cards failed: %v", err)), nil
	}

	var sb strings.Builder
	for i, c := range cards {
		if i > 0 {
			sb.WriteString("\n---\n\n")
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

**Before creating**: call recall with relevant keywords to check if a similar card already exists. If it does, pass its "id" to update it instead of creating a duplicate.

**When to create**: you learned something worth remembering — code patterns, architectural decisions, debugging solutions, project conventions, domain knowledge, tool configurations, etc.

**When to update**: a card's information is incomplete or outdated and you have better/newer information.

**Card structure**:
- path: semantic location in the knowledge tree (e.g. 'programming/go/memory-escape')
- title: concise, descriptive (what is this card about?)
- summary: 1-2 sentences answering "what will I learn from this card?"
- body: the actual knowledge, in markdown — code blocks, commands, explanations
- keywords: terms someone would search for to find this card
- reference: link to source docs/repos for deeper context (optional)`),
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
			mcp.Description("Full card content in markdown. Include code blocks, commands, diagrams (text), and explanations. Keep under 4096 tokens for best embedding quality. Be self-contained — a future reader should understand it without external context."),
		),
		mcp.WithArray("keywords",
			mcp.Description("Keywords that someone might search for to find this card. Include both Chinese and English terms if relevant."),
			mcp.Items(map[string]interface{}{"type": "string"}),
		),
		mcp.WithString("reference",
			mcp.Description("Path or URL to reference documentation, source code, or related resources for deeper context"),
		),
		mcp.WithString("id",
			mcp.Description("To UPDATE an existing card, pass its ID here. Omit this field to create a new card."),
		),
	)
}

func (s *Server) handleUpsertCard(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
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
				if s, ok := item.(string); ok {
					c.Keywords = append(c.Keywords, s)
				}
			}
		case string:
			c.Keywords = strings.Split(v, ",")
		}
	}

	if err := s.store.UpsertCard(c); err != nil {
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
	id, err := req.RequireString("id")
	if err != nil {
		return mcp.NewToolResultError("id is required"), nil
	}

	if err := s.store.DeleteCard(id); err != nil {
		return mcp.NewToolResultError(fmt.Sprintf("delete failed: %v", err)), nil
	}

	return mcp.NewToolResultText(fmt.Sprintf("Card %s has been deleted.", id)), nil
}
