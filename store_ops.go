package knowcard

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/go-git/go-git/v5"
	"github.com/go-git/go-git/v5/plumbing/object"
	"github.com/robert/knowcard/card"
	"github.com/robert/knowcard/embed"
	"github.com/robert/knowcard/search"
)

// Recall performs hybrid (semantic + keyword) search and returns ranked results.
func (s *Store) Recall(query string, opts RecallOpts) ([]RecallResult, error) {
	if opts.TopK <= 0 {
		opts.TopK = 10
	}
	pool := s.cfg.CandidatePool
	if pool <= 0 {
		pool = 30
	}
	// Clamp pool to actual document count (chromem-go requires nResults <= Count)
	if colCount := s.col.Count(); colCount > 0 && pool > colCount {
		pool = colCount
	}

	// Semantic search via chromem-go
	semanticRes, err := s.col.Query(context.Background(), query, pool, nil, nil)
	if err != nil {
		return nil, fmt.Errorf("vector search: %w", err)
	}
	semanticLane := make([]search.SearchResult, 0, len(semanticRes))
	for _, r := range semanticRes {
		semanticLane = append(semanticLane, search.SearchResult{DocID: r.ID, Score: float64(r.Similarity)})
	}

	// Keyword search via BM25
	keywordLane := s.bm25.Search(query, pool)

	// RRF fusion
	fused := search.RRF([][]search.SearchResult{semanticLane, keywordLane}, s.cfg.RRFK)

	// Build results with filtering
	results := make([]RecallResult, 0, opts.TopK)
	for _, f := range fused {
		if len(results) >= opts.TopK {
			break
		}

		c, err := s.readCardByID(f.DocID)
		if err != nil {
			continue
		}

		// Path prefix filter
		if opts.PathPref != "" && !strings.HasPrefix(c.Path, opts.PathPref) {
			continue
		}

		// Tag filter (card must have ALL requested tags)
		if len(opts.Tags) > 0 {
			cardTags := make(map[string]bool)
			for _, t := range c.Tags {
				cardTags[t] = true
			}
			matched := true
			for _, want := range opts.Tags {
				if !cardTags[want] {
					matched = false
					break
				}
			}
			if !matched {
				continue
			}
		}

		hitType := "both"
		if len(f.HitTypes) > 0 {
			hitType = f.HitTypes[0]
		}

		results = append(results, RecallResult{
			ID:      c.ID,
			Path:    c.Path,
			Title:   c.Title,
			Summary: c.Summary,
			Score:   f.Score,
			HitType: hitType,
		})
	}

	return results, nil
}

// GetCards retrieves full card content by IDs.
func (s *Store) GetCards(ids []string) ([]card.Card, error) {
	results := make([]card.Card, 0, len(ids))
	for _, id := range ids {
		c, err := s.readCardByID(id)
		if err != nil {
			return nil, fmt.Errorf("getting card %s: %w", id, err)
		}
		results = append(results, *c)
	}
	return results, nil
}

// GetCard retrieves a single card by ID.
func (s *Store) GetCard(id string) (*card.Card, error) {
	return s.readCardByID(id)
}

// ListCards returns all card IDs and paths, optionally filtered by path prefix.
func (s *Store) ListCards(pathPrefix string) []RecallResult {
	results := make([]RecallResult, 0)
	for id, p := range s.idToPath {
		if pathPrefix != "" && !strings.HasPrefix(p, pathPrefix) {
			continue
		}
		c, err := s.readCardByID(id)
		if err != nil {
			continue
		}
		results = append(results, RecallResult{
			ID:      id,
			Path:    p,
			Title:   c.Title,
			Summary: c.Summary,
		})
	}
	return results
}

// UpsertCard adds or replaces a card. It writes the .md file, updates both
// indexes, and auto-commits to git.
func (s *Store) UpsertCard(c *card.Card) error {
	// Token count validation
	if tc, ok := s.embedder.(embed.TokenCounter); ok {
		if err := c.Validate(tc.CountTokens); err != nil {
			return err
		}
	} else {
		if err := c.Validate(nil); err != nil {
			return err
		}
	}

	now := time.Now().UTC()
	if c.Created.IsZero() {
		c.Created = now
	}
	c.Updated = now

	// If card already exists (by ID), remove old entries
	if oldPath, exists := s.idToPath[c.ID]; exists {
		if err := s.deleteFromVectorIndex(c.ID); err != nil {
			return fmt.Errorf("removing old vector: %w", err)
		}
		s.bm25.RemoveDocument(c.ID)
		// If path changed, delete old file
		if oldPath != c.Path {
			card.DeleteCardFile(oldPath, s.cfg.CardsDir())
		}
	}

	// Write .md file
	if err := card.WriteCard(c, s.cfg.CardsDir()); err != nil {
		return fmt.Errorf("writing card file: %w", err)
	}

	// Add to vector index
	if err := s.addToVectorIndex(c); err != nil {
		return fmt.Errorf("indexing card: %w", err)
	}

	// Add to BM25
	s.bm25.AddDocument(c.ID, c.Title+" "+c.Summary+" "+strings.Join(c.Keywords, " ")+" "+c.Body)

	// Update id→path map
	s.idToPath[c.ID] = c.Path

	// Auto-commit
	if err := s.gitCommit(fmt.Sprintf("upsert: %s", c.Title)); err != nil {
		return fmt.Errorf("git commit: %w", err)
	}

	// Update manifest
	return s.writeManifest()
}

// DeleteCard removes a card from the store.
func (s *Store) DeleteCard(id string) error {
	p, ok := s.idToPath[id]
	if !ok {
		return fmt.Errorf("card not found: %s", id)
	}

	// Remove from indexes
	if err := s.deleteFromVectorIndex(id); err != nil {
		return fmt.Errorf("removing vector: %w", err)
	}
	s.bm25.RemoveDocument(id)

	// Remove file
	if err := card.DeleteCardFile(p, s.cfg.CardsDir()); err != nil {
		return fmt.Errorf("deleting card file: %w", err)
	}

	// Update map
	delete(s.idToPath, id)

	// Auto-commit
	if err := s.gitCommit(fmt.Sprintf("delete: %s", id)); err != nil {
		return fmt.Errorf("git commit: %w", err)
	}

	return s.writeManifest()
}

// MoveCard changes a card's path without modifying its content.
func (s *Store) MoveCard(id, newPath string) error {
	if err := card.ValidatePath(newPath); err != nil {
		return err
	}
	oldPath, ok := s.idToPath[id]
	if !ok {
		return fmt.Errorf("card not found: %s", id)
	}
	if oldPath == newPath {
		return nil
	}

	// Move file
	if err := card.MoveCardFile(oldPath, newPath, s.cfg.CardsDir()); err != nil {
		return fmt.Errorf("moving card file: %w", err)
	}

	// Update map
	s.idToPath[id] = newPath

	// Re-index in chromem-go (metadata update = delete + re-add)
	c, err := s.readCardByID(id)
	if err != nil {
		return fmt.Errorf("re-reading card after move: %w", err)
	}
	if err := s.deleteFromVectorIndex(id); err != nil {
		return fmt.Errorf("removing old vector: %w", err)
	}
	if err := s.addToVectorIndex(c); err != nil {
		return fmt.Errorf("re-indexing card: %w", err)
	}

	// Auto-commit
	if err := s.gitCommit(fmt.Sprintf("move: %s → %s", oldPath, newPath)); err != nil {
		return fmt.Errorf("git commit: %w", err)
	}

	return s.writeManifest()
}

// History returns the git revision history for a specific card.
func (s *Store) History(id string) ([]CardRevision, error) {
	p, ok := s.idToPath[id]
	if !ok {
		return nil, fmt.Errorf("card not found: %s", id)
	}

	filePath := p + ".md"
	ref, err := s.repo.Head()
	if err != nil {
		return nil, fmt.Errorf("getting HEAD: %w", err)
	}

	cIter, err := s.repo.Log(&git.LogOptions{
		From:       ref.Hash(),
		PathFilter: func(path string) bool { return path == filePath },
	})
	if err != nil {
		return nil, fmt.Errorf("git log: %w", err)
	}

	revisions := make([]CardRevision, 0)
	err = cIter.ForEach(func(c *object.Commit) error {
		revisions = append(revisions, CardRevision{
			Hash:    c.Hash.String(),
			Message: c.Message,
			When:    c.Author.When,
		})
		return nil
	})
	if err != nil {
		return nil, err
	}

	return revisions, nil
}

// gitCommit stages all changes and creates a commit.
func (s *Store) gitCommit(msg string) error {
	wt, err := s.repo.Worktree()
	if err != nil {
		return err
	}
	wt.AddWithOptions(&git.AddOptions{All: true})
	_, err = wt.Commit(msg, &git.CommitOptions{
		Author: &object.Signature{
			Name:  "knowcard",
			Email: "knowcard@local",
			When:  time.Now(),
		},
	})
	return err
}
