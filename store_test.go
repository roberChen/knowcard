package knowcard

import (
	"fmt"
	"math/rand"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/robert/knowcard/card"
)

// fakeEmbedder generates deterministic embeddings based on text hashing.
// It does not produce semantically meaningful vectors but is sufficient
// for testing the store integration logic.
type fakeEmbedder struct {
	dim int
}

func newFakeEmbedder() *fakeEmbedder {
	return &fakeEmbedder{dim: 64}
}

func (f *fakeEmbedder) Embed(text string) ([]float32, error) {
	// Deterministic pseudo-embedding: seed from text hash, fill vector
	rng := rand.New(rand.NewSource(int64(hashText(text))))
	vec := make([]float32, f.dim)
	var sum float64
	for i := range vec {
		vec[i] = float32(rng.NormFloat64())
		sum += float64(vec[i]) * float64(vec[i])
	}
	// L2 normalize
	norm := 1.0 / sqrt(sum)
	for i := range vec {
		vec[i] = float32(float64(vec[i]) * norm)
	}
	return vec, nil
}

func (f *fakeEmbedder) EmbedBatch(texts []string) ([][]float32, error) {
	out := make([][]float32, len(texts))
	for i, t := range texts {
		emb, _ := f.Embed(t)
		out[i] = emb
	}
	return out, nil
}

func (f *fakeEmbedder) Dim() int    { return f.dim }
func (f *fakeEmbedder) Close() error { return nil }

func hashText(s string) uint64 {
	var h uint64 = 1469598103934665603
	for _, c := range s {
		h ^= uint64(c)
		h *= 1099511628211
	}
	return h
}

func sqrt(x float64) float64 {
	if x <= 0 {
		return 0
	}
	z := x
	for i := 0; i < 20; i++ {
		z = z - (z*z-x)/(2*z)
	}
	return z
}

func tempConfig(t *testing.T) Config {
	t.Helper()
	dir := t.TempDir()
	return Config{
		Root:          dir,
		RRFK:          60,
		CandidatePool: 30,
		Embed: EmbedConfig{
			Backend: EmbedLocal,
		},
	}
}

func TestStore_UpsertAndGet(t *testing.T) {
	cfg := tempConfig(t)
	s, err := OpenWithEmbedder(cfg, newFakeEmbedder())
	if err != nil {
		t.Fatalf("Open failed: %v", err)
	}
	defer s.Close()

	c := &card.Card{
		ID:       card.NewID(),
		Path:     "programming/go/escape-analysis",
		Title:    "Go 内存逃逸分析",
		Keywords: []string{"逃逸分析", "栈分配", "go"},
		Summary:  "解释 Go 中变量从栈逃逸到堆的条件和检测方法。",
		Body:     "# Go 内存逃逸分析\n\n当变量被取地址且逃出当前栈帧时，编译器会将其分配到堆上。",
		Tags:     []string{"go", "performance"},
	}

	if err := s.UpsertCard(c); err != nil {
		t.Fatalf("UpsertCard failed: %v", err)
	}

	// Verify file exists
	filePath := filepath.Join(cfg.CardsDir(), c.FilePath())
	if _, err := os.Stat(filePath); os.IsNotExist(err) {
		t.Errorf("card file not created at %s", filePath)
	}

	// GetCard should return the same data
	got, err := s.GetCard(c.ID)
	if err != nil {
		t.Fatalf("GetCard failed: %v", err)
	}
	if got.Title != c.Title {
		t.Errorf("Title = %q, want %q", got.Title, c.Title)
	}
	if got.Path != c.Path {
		t.Errorf("Path = %q, want %q", got.Path, c.Path)
	}
}

func TestStore_Recall(t *testing.T) {
	cfg := tempConfig(t)
	s, _ := OpenWithEmbedder(cfg, newFakeEmbedder())
	defer s.Close()

	cards := []*card.Card{
		{
			ID: card.NewID(), Path: "go/escape",
			Title: "内存逃逸分析", Keywords: []string{"逃逸"},
			Summary: "Go 变量逃逸到堆", Body: "逃逸分析详细内容",
			Tags: []string{"go"}, Created: time.Now(), Updated: time.Now(),
		},
		{
			ID: card.NewID(), Path: "rust/ownership",
			Title: "所有权模型", Keywords: []string{"所有权"},
			Summary: "Rust 所有权与借用", Body: "所有权详细内容",
			Tags: []string{"rust"}, Created: time.Now(), Updated: time.Now(),
		},
		{
			ID: card.NewID(), Path: "db/redis",
			Title: "Redis 持久化", Keywords: []string{"redis"},
			Summary: "RDB 与 AOF 持久化", Body: "redis persistence details",
			Tags: []string{"database"}, Created: time.Now(), Updated: time.Now(),
		},
	}
	for _, c := range cards {
		if err := s.UpsertCard(c); err != nil {
			t.Fatalf("UpsertCard failed: %v", err)
		}
	}

	// Recall should return results
	results, err := s.Recall("逃逸分析", RecallOpts{TopK: 3})
	if err != nil {
		t.Fatalf("Recall failed: %v", err)
	}
	if len(results) == 0 {
		t.Fatal("expected recall results, got none")
	}
	t.Logf("Recall results for '逃逸分析':")
	for _, r := range results {
		t.Logf("  %s (%s) score=%.4f hit=%s", r.Title, r.ID, r.Score, r.HitType)
	}
}

func TestStore_DeleteCard(t *testing.T) {
	cfg := tempConfig(t)
	s, _ := OpenWithEmbedder(cfg, newFakeEmbedder())
	defer s.Close()

	c := &card.Card{
		ID: card.NewID(), Path: "test/delete",
		Title: "Delete Test", Summary: "S", Body: "B",
		Created: time.Now(), Updated: time.Now(),
	}
	s.UpsertCard(c)

	if err := s.DeleteCard(c.ID); err != nil {
		t.Fatalf("DeleteCard failed: %v", err)
	}

	// File should be gone
	filePath := filepath.Join(cfg.CardsDir(), c.FilePath())
	if _, err := os.Stat(filePath); !os.IsNotExist(err) {
		t.Errorf("card file should be deleted")
	}

	// GetCard should fail
	_, err := s.GetCard(c.ID)
	if err == nil {
		t.Error("expected error getting deleted card")
	}
}

func TestStore_MoveCard(t *testing.T) {
	cfg := tempConfig(t)
	s, _ := OpenWithEmbedder(cfg, newFakeEmbedder())
	defer s.Close()

	c := &card.Card{
		ID: card.NewID(), Path: "old/location",
		Title: "Move Test", Summary: "S", Body: "B",
		Created: time.Now(), Updated: time.Now(),
	}
	s.UpsertCard(c)

	newPath := "new/location"
	if err := s.MoveCard(c.ID, newPath); err != nil {
		t.Fatalf("MoveCard failed: %v", err)
	}

	// Old file gone, new file exists
	oldFile := filepath.Join(cfg.CardsDir(), "old/location.md")
	newFile := filepath.Join(cfg.CardsDir(), "new/location.md")
	if _, err := os.Stat(oldFile); !os.IsNotExist(err) {
		t.Error("old card file should be deleted")
	}
	if _, err := os.Stat(newFile); os.IsNotExist(err) {
		t.Error("new card file should exist")
	}

	// ID should still resolve
	got, err := s.GetCard(c.ID)
	if err != nil {
		t.Fatalf("GetCard after move failed: %v", err)
	}
	if got.Path != newPath {
		t.Errorf("Path = %q, want %q", got.Path, newPath)
	}
}

func TestStore_History(t *testing.T) {
	cfg := tempConfig(t)
	s, _ := OpenWithEmbedder(cfg, newFakeEmbedder())
	defer s.Close()

	c := &card.Card{
		ID: card.NewID(), Path: "history/test",
		Title: "V1", Summary: "S1", Body: "B1",
		Created: time.Now(), Updated: time.Now(),
	}
	s.UpsertCard(c)

	// Modify
	c.Title = "V2"
	c.Body = "B2 modified"
	time.Sleep(time.Millisecond * 10) // ensure different commit time
	s.UpsertCard(c)

	revs, err := s.History(c.ID)
	if err != nil {
		t.Fatalf("History failed: %v", err)
	}
	if len(revs) < 2 {
		t.Errorf("expected at least 2 revisions, got %d", len(revs))
	}
	t.Logf("History for %s:", c.ID)
	for _, r := range revs {
		t.Logf("  %s: %s (%s)", r.Hash[:8], r.Message, r.When.Format(time.RFC3339))
	}
}

func TestStore_Rebuild(t *testing.T) {
	cfg := tempConfig(t)
	s, _ := OpenWithEmbedder(cfg, newFakeEmbedder())
	defer s.Close()

	// Add some cards
	for i := 0; i < 3; i++ {
		s.UpsertCard(&card.Card{
			ID: card.NewID(), Path: fmt.Sprintf("rebuild/card%d", i),
			Title: fmt.Sprintf("Card %d", i), Summary: "S", Body: fmt.Sprintf("Body %d", i),
			Created: time.Now(), Updated: time.Now(),
		})
	}

	// Rebuild should preserve all cards
	if err := s.Rebuild(); err != nil {
		t.Fatalf("Rebuild failed: %v", err)
	}

	results := s.ListCards("")
	if len(results) != 3 {
		t.Errorf("after rebuild, expected 3 cards, got %d", len(results))
	}
}

func TestStore_StandardGitNotVisible(t *testing.T) {
	cfg := tempConfig(t)
	s, _ := OpenWithEmbedder(cfg, newFakeEmbedder())
	defer s.Close()

	c := &card.Card{
		ID: card.NewID(), Path: "git/test",
		Title: "T", Summary: "S", Body: "B",
		Created: time.Now(), Updated: time.Now(),
	}
	s.UpsertCard(c)

	// Standard .git directory should NOT exist in cards dir
	gitDir := filepath.Join(cfg.CardsDir(), ".git")
	if _, err := os.Stat(gitDir); !os.IsNotExist(err) {
		t.Error(".git directory should not exist in cards/ (separated VCS)")
	}

	// VCS directory should exist elsewhere
	vcsDir := cfg.VcsDir()
	if _, err := os.Stat(vcsDir); os.IsNotExist(err) {
		t.Error("_vcs directory should exist")
	}
}

func TestStore_ReferenceFile(t *testing.T) {
	cfg := tempConfig(t)
	s, _ := OpenWithEmbedder(cfg, newFakeEmbedder())
	defer s.Close()

	// Create a temp source file to use as reference
	srcFile := filepath.Join(t.TempDir(), "design.md")
	srcContent := "# Design Doc\n\nDetailed architecture notes..."
	if err := os.WriteFile(srcFile, []byte(srcContent), 0644); err != nil {
		t.Fatal(err)
	}

	c := &card.Card{
		ID:        card.NewID(),
		Path:      "arch/design",
		Title:     "Design", Summary: "S", Body: "B",
		Reference: srcFile,
		Created:   time.Now(), Updated: time.Now(),
	}
	if err := s.UpsertCard(c); err != nil {
		t.Fatalf("UpsertCard with reference failed: %v", err)
	}

	// Reference should now be a KB-relative path
	if c.Reference == srcFile {
		t.Fatal("Reference should be updated to KB-relative path after upsert")
	}
	expectedRef := filepath.ToSlash(filepath.Join("_refs", c.ID, "design.md"))
	if c.Reference != expectedRef {
		t.Errorf("Reference = %q, want %q", c.Reference, expectedRef)
	}

	// Reference file should exist in KB
	refPath := filepath.Join(cfg.CardsDir(), c.Reference)
	data, err := os.ReadFile(refPath)
	if err != nil {
		t.Fatalf("reference file not copied to KB: %v", err)
	}
	if string(data) != srcContent {
		t.Errorf("reference file content mismatch")
	}

	// _refs should not be picked up as cards by loadIndex/Rebuild
	if err := s.Rebuild(); err != nil {
		t.Fatalf("Rebuild failed: %v", err)
	}
	results := s.ListCards("")
	for _, r := range results {
		if r.ID == "" {
			t.Error("found a card with empty ID (likely from _refs)")
		}
	}

	// DeleteCard should remove the reference directory
	if err := s.DeleteCard(c.ID); err != nil {
		t.Fatalf("DeleteCard failed: %v", err)
	}
	refDir := filepath.Join(cfg.CardsDir(), "_refs", c.ID)
	if _, err := os.Stat(refDir); !os.IsNotExist(err) {
		t.Error("reference directory should be deleted with card")
	}
}

func TestStore_ReferenceFileNotFound(t *testing.T) {
	cfg := tempConfig(t)
	s, _ := OpenWithEmbedder(cfg, newFakeEmbedder())
	defer s.Close()

	c := &card.Card{
		ID:        card.NewID(),
		Path:      "test/noref",
		Title:     "T", Summary: "S", Body: "B",
		Reference: "/nonexistent/path/to/file.md",
		Created:   time.Now(), Updated: time.Now(),
	}
	err := s.UpsertCard(c)
	if err == nil {
		t.Fatal("expected error for nonexistent reference file")
	}
	if !strings.Contains(err.Error(), "reference file not found") {
		t.Errorf("error should mention 'reference file not found', got: %v", err)
	}
}

func TestStore_BodyTooLongError(t *testing.T) {
	cfg := tempConfig(t)
	s, _ := OpenWithEmbedder(cfg, newFakeEmbedder())
	defer s.Close()

	// Create body that exceeds MaxBodyTokens via heuristic (len/3)
	longBody := strings.Repeat("a", card.MaxBodyTokens*3+100)
	c := &card.Card{
		ID:      card.NewID(),
		Path:    "test/long",
		Title:   "T", Summary: "S", Body: longBody,
		Created: time.Now(), Updated: time.Now(),
	}
	err := s.UpsertCard(c)
	if err == nil {
		t.Fatal("expected error for oversized body")
	}
	if !strings.Contains(err.Error(), "shorten") || !strings.Contains(err.Error(), "reference") {
		t.Errorf("error should guide to shorten body and use reference, got: %v", err)
	}
}
