package search

import (
	"math"
	"sort"
	"strings"
	"sync"
)

// BM25 implements an in-memory inverted index with BM25 scoring.
// It is safe for concurrent use.
type BM25 struct {
	mu sync.RWMutex

	// docFreq[token] = number of documents containing this token
	docFreq map[string]int

	// postings[token] = list of {docID, termFrequency}
	postings map[string][]posting

	// docLen[docID] = total token count in the document
	docLen map[string]int

	// docCount = total number of indexed documents
	docCount int

	// avgDocLen = average document length
	avgDocLen float64

	// BM25 parameters
	k1 float64
	b  float64

	tok *Tokenizer
}

type posting struct {
	docID string
	tf    int
}

// SearchResult represents a single BM25 search hit.
type SearchResult struct {
	DocID string
	Score float64
}

// NewBM25 creates a new BM25 index with default parameters (k1=1.5, b=0.75).
func NewBM25() *BM25 {
	return &BM25{
		docFreq:  make(map[string]int),
		postings: make(map[string][]posting),
		docLen:   make(map[string]int),
		k1:       1.5,
		b:        0.75,
		tok:      NewTokenizer(),
	}
}

// SetParams allows tuning BM25 parameters k1 and b.
func (b *BM25) SetParams(k1, bParam float64) {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.k1 = k1
	b.b = bParam
}

// AddDocument indexes a document. If docID already exists, it is replaced.
// The text parameter is tokenized and indexed.
func (b *BM25) AddDocument(docID, text string) {
	b.mu.Lock()
	defer b.mu.Unlock()

	// If doc already exists, remove it first
	if _, exists := b.docLen[docID]; exists {
		b.removeLocked(docID)
	}

	tokens := b.tok.Tokenize(text)
	if len(tokens) == 0 {
		return
	}

	// Count term frequencies
	tfMap := make(map[string]int)
	for _, tok := range tokens {
		tfMap[tok]++
	}

	// Update postings and document frequencies
	for token, freq := range tfMap {
		b.postings[token] = append(b.postings[token], posting{docID: docID, tf: freq})
		b.docFreq[token]++
	}

	// Update document stats
	oldTotal := b.avgDocLen * float64(b.docCount)
	b.docLen[docID] = len(tokens)
	b.docCount++
	b.avgDocLen = (oldTotal + float64(len(tokens))) / float64(b.docCount)
}

// RemoveDocument removes a document from the index.
func (b *BM25) RemoveDocument(docID string) {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.removeLocked(docID)
}

func (b *BM25) removeLocked(docID string) {
	docLen, exists := b.docLen[docID]
	if !exists {
		return
	}

	// Remove from postings
	for token, plist := range b.postings {
		filtered := plist[:0]
		for _, p := range plist {
			if p.docID != docID {
				filtered = append(filtered, p)
			}
		}
		if len(filtered) == 0 {
			delete(b.postings, token)
			delete(b.docFreq, token)
		} else {
			b.postings[token] = filtered
			b.docFreq[token] = len(filtered)
		}
	}

	// Update document stats
	oldTotal := b.avgDocLen * float64(b.docCount)
	delete(b.docLen, docID)
	b.docCount--
	if b.docCount > 0 {
		b.avgDocLen = (oldTotal - float64(docLen)) / float64(b.docCount)
	} else {
		b.avgDocLen = 0
	}
}

// Search returns the top-K documents matching the query, ranked by BM25 score.
// Only documents with a positive score are returned.
func (b *BM25) Search(query string, topK int) []SearchResult {
	b.mu.RLock()
	defer b.mu.RUnlock()

	if b.docCount == 0 {
		return nil
	}

	queryTokens := b.tok.Tokenize(query)
	if len(queryTokens) == 0 {
		return nil
	}

	scores := make(map[string]float64)
	seen := make(map[string]bool) // deduplicate query tokens

	for _, qt := range queryTokens {
		if seen[qt] {
			continue
		}
		seen[qt] = true

		plist, ok := b.postings[qt]
		if !ok {
			continue
		}

		df := b.docFreq[qt]
		idf := math.Log(1.0 + (float64(b.docCount)-float64(df)+0.5)/(float64(df)+0.5))
		if idf <= 0 {
			continue
		}

		for _, p := range plist {
			tfNorm := (float64(p.tf) * (b.k1 + 1)) /
				(float64(p.tf) + b.k1*(1-b.b+b.b*float64(b.docLen[p.docID])/b.avgDocLen))
			scores[p.docID] += idf * tfNorm
		}
	}

	if len(scores) == 0 {
		return nil
	}

	results := make([]SearchResult, 0, len(scores))
	for docID, score := range scores {
		if score > 0 {
			results = append(results, SearchResult{DocID: docID, Score: score})
		}
	}

	sort.Slice(results, func(i, j int) bool {
		return results[i].Score > results[j].Score
	})

	if topK > 0 && len(results) > topK {
		results = results[:topK]
	}

	return results
}

// Count returns the number of indexed documents.
func (b *BM25) Count() int {
	b.mu.RLock()
	defer b.mu.RUnlock()
	return b.docCount
}

// Clear removes all documents from the index.
func (b *BM25) Clear() {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.docFreq = make(map[string]int)
	b.postings = make(map[string][]posting)
	b.docLen = make(map[string]int)
	b.docCount = 0
	b.avgDocLen = 0
}

// KeywordsMatch returns docIDs that contain ALL the given keywords.
// This is used for exact keyword matching (from card front matter keywords).
func (b *BM25) KeywordsMatch(keywords []string) []string {
	b.mu.RLock()
	defer b.mu.RUnlock()

	if len(keywords) == 0 {
		return nil
	}

	// Find intersection of postings for all keywords
	var candidates map[string]bool
	for _, kw := range keywords {
		kw = strings.ToLower(kw)
		plist, ok := b.postings[kw]
		if !ok {
			return nil // keyword not found, no matches
		}
		docs := make(map[string]bool)
		for _, p := range plist {
			docs[p.docID] = true
		}
		if candidates == nil {
			candidates = docs
		} else {
			for doc := range candidates {
				if !docs[doc] {
					delete(candidates, doc)
				}
			}
		}
	}

	result := make([]string, 0, len(candidates))
	for doc := range candidates {
		result = append(result, doc)
	}
	return result
}
