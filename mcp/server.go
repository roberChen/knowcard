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
		mcp.WithDescription(`Search the knowledge card memory system using hybrid semantic + keyword search.

Returns a list of matching cards with their title, summary, and relevance score.
Use this to find relevant knowledge cards before reading their full content with get_cards.`),
		mcp.WithString("query",
			mcp.Required(),
			mcp.Description("Search query - keywords or a natural language sentence"),
		),
		mcp.WithNumber("top_k",
			mcp.Description("Maximum number of results (default: 10)"),
		),
		mcp.WithString("path_prefix",
			mcp.Description("Filter results by path prefix (e.g. 'programming/go')"),
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
	sb.WriteString(fmt.Sprintf("Found %d card(s):\n\n", len(results)))
	for i, r := range results {
		sb.WriteString(fmt.Sprintf("%d. **%s**\n", i+1, r.Title))
		sb.WriteString(fmt.Sprintf("   - id: `%s`\n", r.ID))
		sb.WriteString(fmt.Sprintf("   - path: `%s`\n", r.Path))
		sb.WriteString(fmt.Sprintf("   - score: %.4f (%s)\n", r.Score, r.HitType))
		sb.WriteString(fmt.Sprintf("   - summary: %s\n\n", r.Summary))
	}

	if len(results) == 0 {
		sb.Reset()
		sb.WriteString("No matching cards found. Try different keywords or check if the knowledge exists in the system.")
	}

	return mcp.NewToolResultText(sb.String()), nil
}

// --- get_cards ---

func (s *Server) getCardsTool() mcp.Tool {
	return mcp.NewTool("get_cards",
		mcp.WithDescription(`Retrieve the full content of one or more knowledge cards by their IDs.

Use this after calling recall to read the complete content of the cards you selected.`),
		mcp.WithArray("ids",
			mcp.Required(),
			mcp.Description("Array of card IDs to retrieve"),
			mcp.Items(map[string]interface{}{"type": "string"}),
		),
	)
}

func (s *Server) handleGetCards(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	idsRaw, err := req.RequireString("ids")
	if err != nil {
		// Might be an array
		args := req.GetArguments()
		idsVal, ok := args["ids"]
		if !ok {
			return mcp.NewToolResultError("ids is required"), nil
		}
		idsRaw, _ = idsVal.(string)
		if idsRaw == "" {
			// Try as array
			if arr, ok := idsVal.([]interface{}); ok {
				ids := make([]string, len(arr))
				for i, v := range arr {
					ids[i], _ = v.(string)
				}
				return s.getCardsByIDs(ids)
			}
			return mcp.NewToolResultError("ids must be an array of strings"), nil
		}
	}
	// Single ID as string
	return s.getCardsByIDs([]string{idsRaw})
}

func (s *Server) getCardsByIDs(ids []string) (*mcp.CallToolResult, error) {
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
		sb.WriteString("No cards found for the given IDs.")
	}

	return mcp.NewToolResultText(sb.String()), nil
}

// --- upsert_card ---

func (s *Server) upsertCardTool() mcp.Tool {
	return mcp.NewTool("upsert_card",
		mcp.WithDescription(`Create or update a knowledge card in the memory system.

This is how the memory system learns new information. The card content should be
self-contained knowledge that would be useful for future reference.

The path determines where the card lives in the knowledge tree and should reflect
its topic hierarchy (e.g. 'programming/go/memory-escape-analysis'). Choose paths
that group related knowledge together.`),
		mcp.WithString("path",
			mcp.Required(),
			mcp.Description("Semantic path for the card (e.g. 'databases/postgres/query-tuning'). Use / for hierarchy."),
		),
		mcp.WithString("title",
			mcp.Required(),
			mcp.Description("Card title - concise, descriptive"),
		),
		mcp.WithString("summary",
			mcp.Required(),
			mcp.Description("Brief summary of what information this card contains (used for search and preview)"),
		),
		mcp.WithString("body",
			mcp.Required(),
			mcp.Description("Full card content in markdown. Keep under 4096 tokens."),
		),
		mcp.WithArray("keywords",
			mcp.Description("Keywords for precise matching"),
			mcp.Items(map[string]interface{}{"type": "string"}),
		),
		mcp.WithString("reference",
			mcp.Description("Path or URL to reference documentation for more detail"),
		),
		mcp.WithString("id",
			mcp.Description("Existing card ID to update. If omitted, a new card is created."),
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
		"status": "success",
		"id":     c.ID,
		"path":   c.Path,
		"title":  c.Title,
	}
	data, _ := json.MarshalIndent(result, "", "  ")
	return mcp.NewToolResultText(fmt.Sprintf("Card saved:\n%s", string(data))), nil
}

// --- delete_card ---

func (s *Server) deleteCardTool() mcp.Tool {
	return mcp.NewTool("delete_card",
		mcp.WithDescription(`Delete a knowledge card from the memory system.`),
		mcp.WithString("id",
			mcp.Required(),
			mcp.Description("ID of the card to delete"),
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

	return mcp.NewToolResultText(fmt.Sprintf("Card %s deleted.", id)), nil
}
