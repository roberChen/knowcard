package search

import (
	"testing"
)

func TestTokenizer_English(t *testing.T) {
	tok := NewTokenizer()
	tokens := tok.Tokenize("Hello World Testing 123")
	expected := []string{"hello", "world", "testing", "123"}
	if len(tokens) != len(expected) {
		t.Fatalf("got %v, want %v", tokens, expected)
	}
	for i, tok := range tokens {
		if tok != expected[i] {
			t.Errorf("token[%d] = %q, want %q", i, tok, expected[i])
		}
	}
}

func TestTokenizer_Chinese(t *testing.T) {
	tok := NewTokenizer()
	tokens := tok.Tokenize("内存逃逸分析")
	// Should contain unigrams: 内,存,逃,逸,分,析
	// And bigrams: 内存,存逃,逃逸,逸分,分析
	has := make(map[string]bool)
	for _, tk := range tokens {
		has[tk] = true
	}
	if !has["内存"] {
		t.Errorf("expected bigram '内存' in tokens: %v", tokens)
	}
	if !has["逃逸"] {
		t.Errorf("expected bigram '逃逸' in tokens: %v", tokens)
	}
	if !has["内"] {
		t.Errorf("expected unigram '内' in tokens: %v", tokens)
	}
}

func TestTokenizer_Mixed(t *testing.T) {
	tok := NewTokenizer()
	tokens := tok.Tokenize("Go语言的Goroutine调度")
	has := make(map[string]bool)
	for _, tk := range tokens {
		has[tk] = true
	}
	if !has["go"] {
		t.Errorf("expected 'go' in tokens: %v", tokens)
	}
	if !has["语言"] {
		t.Errorf("expected bigram '语言' in tokens: %v", tokens)
	}
}

func TestBM25_Search(t *testing.T) {
	bm := NewBM25()
	bm.AddDocument("d1", "the quick brown fox jumps over the lazy dog")
	bm.AddDocument("d2", "the lazy dog sleeps all day")
	bm.AddDocument("d3", "quick thinking saves time in programming")

	results := bm.Search("quick fox", 3)
	if len(results) == 0 {
		t.Fatal("expected results")
	}
	// d1 should rank first (contains both "quick" and "fox")
	if results[0].DocID != "d1" {
		t.Errorf("top result = %s, want d1", results[0].DocID)
	}
}

func TestBM25_Chinese(t *testing.T) {
	bm := NewBM25()
	bm.AddDocument("d1", "Go语言的内存逃逸分析与性能优化")
	bm.AddDocument("d2", "Rust的所有权模型与内存安全")
	bm.AddDocument("d3", "Python的垃圾回收机制")

	results := bm.Search("内存逃逸", 3)
	if len(results) == 0 {
		t.Fatal("expected results for Chinese query")
	}
	if results[0].DocID != "d1" {
		t.Errorf("top result = %s, want d1", results[0].DocID)
	}
}

func TestBM25_RemoveDocument(t *testing.T) {
	bm := NewBM25()
	bm.AddDocument("d1", "hello world")
	bm.AddDocument("d2", "hello there")
	if bm.Count() != 2 {
		t.Errorf("Count = %d, want 2", bm.Count())
	}
	bm.RemoveDocument("d1")
	if bm.Count() != 1 {
		t.Errorf("Count = %d, want 1", bm.Count())
	}
	results := bm.Search("hello", 5)
	if len(results) != 1 || results[0].DocID != "d2" {
		t.Errorf("expected only d2, got %v", results)
	}
}

func TestBM25_ReplaceDocument(t *testing.T) {
	bm := NewBM25()
	bm.AddDocument("d1", "apple banana")
	bm.AddDocument("d1", "cherry date") // replace
	if bm.Count() != 1 {
		t.Errorf("Count = %d, want 1", bm.Count())
	}
	results := bm.Search("apple", 5)
	if len(results) != 0 {
		t.Errorf("expected no results for removed content, got %v", results)
	}
	results = bm.Search("cherry", 5)
	if len(results) != 1 {
		t.Errorf("expected 1 result for new content")
	}
}

func TestBM25_KeywordsMatch(t *testing.T) {
	bm := NewBM25()
	bm.AddDocument("d1", "go memory performance goroutine")
	bm.AddDocument("d2", "rust memory safety ownership")
	bm.AddDocument("d3", "go concurrency channels")

	matches := bm.KeywordsMatch([]string{"go", "memory"})
	if len(matches) != 1 || matches[0] != "d1" {
		t.Errorf("expected [d1], got %v", matches)
	}
}

func TestRRF(t *testing.T) {
	semantic := []SearchResult{
		{DocID: "a", Score: 0.9},
		{DocID: "b", Score: 0.8},
		{DocID: "c", Score: 0.7},
	}
	keyword := []SearchResult{
		{DocID: "b", Score: 5.0},
		{DocID: "d", Score: 3.0},
		{DocID: "a", Score: 1.0},
	}

	fused := RRF([][]SearchResult{semantic, keyword}, 60)
	if len(fused) == 0 {
		t.Fatal("expected fused results")
	}
	// 'a' appears at rank 0 in semantic and rank 2 in keyword
	// 'b' appears at rank 1 in semantic and rank 0 in keyword
	// Both 'a' and 'b' should have hit_type "both"
	foundBoth := false
	for _, f := range fused {
		if f.DocID == "a" || f.DocID == "b" {
			if f.HitTypes[0] == "both" {
				foundBoth = true
			}
		}
	}
	if !foundBoth {
		t.Error("expected at least one doc with hit_type 'both'")
	}

	// 'b' should rank higher than 'a' since it's rank 1+rank 0 vs rank 0+rank 2
	// 'b': 1/61 + 1/60 = 0.01639 + 0.01667 = 0.03306
	// 'a': 1/60 + 1/62 = 0.01667 + 0.01613 = 0.03280
	if fused[0].DocID != "b" {
		t.Errorf("expected 'b' to rank first, got %s", fused[0].DocID)
	}
}

func TestRRF_EmptyLanes(t *testing.T) {
	fused := RRF([][]SearchResult{nil, nil}, 60)
	if len(fused) != 0 {
		t.Errorf("expected empty results, got %d", len(fused))
	}
}

func TestBM25_EmptyQuery(t *testing.T) {
	bm := NewBM25()
	bm.AddDocument("d1", "some content")
	results := bm.Search("", 5)
	if len(results) != 0 {
		t.Errorf("expected no results for empty query, got %v", results)
	}
}
