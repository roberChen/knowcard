package knowcard

import (
	"context"
	"encoding/json"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/go-git/go-billy/v5/osfs"
	"github.com/go-git/go-git/v5"
	"github.com/go-git/go-git/v5/plumbing/cache"
	gitstorage "github.com/go-git/go-git/v5/storage/filesystem"
	"github.com/philippgille/chromem-go"

	"github.com/robert/knowcard/card"
	"github.com/robert/knowcard/embed"
	"github.com/robert/knowcard/search"
)

const collectionName = "cards"

// Store is the main knowcard store, integrating card files, vector search,
// keyword search, and git-backed versioning.
type Store struct {
	cfg      Config
	embedder embed.Embedder
	db       *chromem.DB
	col      *chromem.Collection
	bm25     *search.BM25
	repo     *git.Repository
	idToPath map[string]string // card ID → relative path (no .md extension)
}

// RecallResult is a lightweight search result (no full body).
type RecallResult struct {
	ID       string  `json:"id"`
	Path     string  `json:"path"`
	Title    string  `json:"title"`
	Summary  string  `json:"summary"`
	Score    float64 `json:"score"`
	HitType  string  `json:"hit_type"` // "semantic", "keyword", "both"
}

// RecallOpts controls recall behaviour.
type RecallOpts struct {
	TopK     int      // final result count (default 10)
	PathPref string   // filter by path prefix (e.g. "programming/go")
	Tags     []string // filter by tags
}

// CardRevision represents one git revision of a card.
type CardRevision struct {
	Hash    string    `json:"hash"`
	Message string    `json:"message"`
	When    time.Time `json:"when"`
}

// Open initialises the store: creates directories, opens/creates the git repo,
// loads the embedder, and builds the index from card files if needed.
func Open(cfg Config) (*Store, error) {
	emb, err := embed.New(embed.Config{
		Backend:               string(cfg.Embed.Backend),
		ModelPath:             cfg.Embed.ModelPath,
		LibPath:               cfg.Embed.LibPath,
		ContextSize:           cfg.Embed.ContextSize,
		BatchSize:             cfg.Embed.BatchSize,
		Pooling:               cfg.Embed.Pooling,
		APIBase:               cfg.Embed.APIBase,
		APIKey:                cfg.Embed.APIKey,
		Model:                 cfg.Embed.Model,
		Dimensions:            cfg.Embed.Dimensions,
		DashScopeInternational: cfg.Embed.DashScopeInternational,
		Instruct:              cfg.Embed.Instruct,
	})
	if err != nil {
		return nil, fmt.Errorf("initializing embedder: %w", err)
	}
	return openWithEmbedder(cfg, emb)
}

// OpenWithEmbedder is like Open but allows injecting a custom embedder
// (useful for testing with mock embedders).
func OpenWithEmbedder(cfg Config, emb embed.Embedder) (*Store, error) {
	return openWithEmbedder(cfg, emb)
}

func openWithEmbedder(cfg Config, emb embed.Embedder) (*Store, error) {
	if err := cfg.EnsureDirs(); err != nil {
		return nil, err
	}

	// Open or create chromem-go persistent DB
	db, err := chromem.NewPersistentDB(cfg.ChromemDir(), false)
	if err != nil {
		return nil, fmt.Errorf("opening vector DB: %w", err)
	}

	// Get or create collection with the embedder's embedding function
	embedFunc := chromem.EmbeddingFunc(func(ctx context.Context, text string) ([]float32, error) {
		return emb.Embed(text)
	})
	col, err := db.GetOrCreateCollection(collectionName, nil, embedFunc)
	if err != nil || col == nil {
		return nil, fmt.Errorf("failed to get or create collection")
	}

	// Open or init git repo with separated git-dir and work-tree
	repo, err := openOrInitRepo(cfg.VcsDir(), cfg.CardsDir())
	if err != nil {
		return nil, fmt.Errorf("opening git repo: %w", err)
	}

	s := &Store{
		cfg:      cfg,
		embedder: emb,
		db:       db,
		col:      col,
		bm25:     search.NewBM25(),
		repo:     repo,
		idToPath: make(map[string]string),
	}

	// Build id→path map and populate BM25 from chromem-go metadata if available
	if err := s.loadIndex(); err != nil {
		return nil, fmt.Errorf("loading index: %w", err)
	}

	return s, nil
}

// Close releases resources.
func (s *Store) Close() error {
	if c, ok := s.embedder.(interface{ Close() error }); ok {
		c.Close()
	}
	return nil
}

// loadIndex scans all .md files and builds the id→path map.
// If the chromem-go collection is empty, it triggers a full rebuild.
func (s *Store) loadIndex() error {
	err := filepath.WalkDir(s.cfg.CardsDir(), func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() || !strings.HasSuffix(path, ".md") {
			return nil
		}
		c, err := card.ReadCard(path, s.cfg.CardsDir())
		if err != nil {
			return fmt.Errorf("reading card %s: %w", path, err)
		}
		s.idToPath[c.ID] = c.Path
		return nil
	})
	if err != nil {
		return err
	}

	// If chromem collection is empty but we have cards, rebuild
	if s.col.Count() == 0 && len(s.idToPath) > 0 {
		return s.Rebuild()
	}

	// Populate BM25 from existing cards
	for id := range s.idToPath {
		c, err := s.readCardByID(id)
		if err != nil {
			continue
		}
		s.bm25.AddDocument(id, c.Title+" "+c.Summary+" "+strings.Join(c.Keywords, " ")+" "+c.Body)
	}
	return nil
}

// Rebuild reconstructs the entire index (chromem-go + BM25) from .md files.
func (s *Store) Rebuild() error {
	// Clear existing vector documents by ID
	for id := range s.idToPath {
		s.col.Delete(context.Background(), nil, nil, id)
	}
	s.bm25.Clear()
	s.idToPath = make(map[string]string)

	// Walk and re-index all cards
	var cards []*card.Card
	err := filepath.WalkDir(s.cfg.CardsDir(), func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() || !strings.HasSuffix(path, ".md") {
			return nil
		}
		c, err := card.ReadCard(path, s.cfg.CardsDir())
		if err != nil {
			return fmt.Errorf("reading card %s: %w", path, err)
		}
		cards = append(cards, c)
		s.idToPath[c.ID] = c.Path
		return nil
	})
	if err != nil {
		return err
	}

	// Batch add to chromem-go
	for _, c := range cards {
		if err := s.addToVectorIndex(c); err != nil {
			return fmt.Errorf("indexing card %s: %w", c.ID, err)
		}
		s.bm25.AddDocument(c.ID, c.Title+" "+c.Summary+" "+strings.Join(c.Keywords, " ")+" "+c.Body)
	}

	// Update integrity manifest
	return s.writeManifest()
}

// readCardByID reads a card file by its ID.
func (s *Store) readCardByID(id string) (*card.Card, error) {
	p, ok := s.idToPath[id]
	if !ok {
		return nil, fmt.Errorf("card not found: %s", id)
	}
	fullPath := filepath.Join(s.cfg.CardsDir(), p+".md")
	return card.ReadCard(fullPath, s.cfg.CardsDir())
}

// addToVectorIndex adds a card to the chromem-go collection.
func (s *Store) addToVectorIndex(c *card.Card) error {
	content := c.Title + "\n" + strings.Join(c.Keywords, " ") + "\n" + c.Summary + "\n" + c.Body
	meta := map[string]string{
		"id":    c.ID,
		"path":  c.Path,
		"title": c.Title,
	}
	for i, tag := range c.Tags {
		meta["tag_"+fmt.Sprint(i)] = tag
	}
	return s.col.AddDocument(context.Background(), chromem.Document{
		ID:       c.ID,
		Content:  content,
		Metadata: meta,
	})
}

// deleteFromVectorIndex removes a card from chromem-go by its ID.
func (s *Store) deleteFromVectorIndex(id string) error {
	return s.col.Delete(context.Background(), nil, nil, id)
}

// Manifest for integrity checking (separated git-dir, so standard git CLI
// cannot see the repository).
type manifest struct {
	HeadCommit string `json:"head_commit"`
	CardCount  int    `json:"card_count"`
	Updated    string `json:"updated"`
}

func (s *Store) writeManifest() error {
	head, _ := s.repo.Head()
	var headHash string
	if head != nil {
		headHash = head.Hash().String()
	}
	m := manifest{
		HeadCommit: headHash,
		CardCount:  len(s.idToPath),
		Updated:    time.Now().Format(time.RFC3339),
	}
	data, _ := json.MarshalIndent(m, "", "  ")
	return os.WriteFile(s.cfg.ManifestPath(), data, 0644)
}

// openOrInitRepo opens or initialises a git repository with the git metadata
// stored in vcsDir and the work tree in cardsDir. This separation means
// standard `git` CLI commands in cardsDir will report "not a git repository".
func openOrInitRepo(vcsDir, cardsDir string) (*git.Repository, error) {
	gitFs := osfs.New(vcsDir)
	wtFs := osfs.New(cardsDir)

	storer := gitstorage.NewStorage(gitFs, cache.NewObjectLRUDefault())

	// Try to open first
	repo, err := git.Open(storer, wtFs)
	if err == nil {
		return repo, nil
	}

	// If open fails, init
	repo, err = git.Init(storer, wtFs)
	if err != nil {
		return nil, fmt.Errorf("git init: %w", err)
	}

	// Remove the .git pointer file that go-git creates in the worktree.
	// This prevents standard git CLI from discovering the repo in cardsDir.
	// go-git operates via the storer, so it doesn't need this file.
	dotGitPath := filepath.Join(cardsDir, ".git")
	os.Remove(dotGitPath)

	return repo, nil
}
