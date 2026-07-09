package embed

import (
	"context"
	"errors"
	"fmt"
	"math"
	"sync"

	"github.com/hybridgroup/yzma/pkg/llama"
)

// LocalEmbedder runs a GGUF embedding model in-process via yzma (llama.cpp binding).
// It is safe for concurrent use (guarded by a mutex, since the underlying
// llama.cpp context is single-threaded per decode call).
type LocalEmbedder struct {
	mu        sync.Mutex
	libLoaded bool
	model     llama.Model
	ctx       llama.Context
	vocab     llama.Vocab
	nEmbd     int32
	pooling   llama.PoolingType
}

// LocalConfig holds parameters for the local embedder.
type LocalConfig struct {
	ModelPath   string
	LibPath     string
	ContextSize uint32
	BatchSize   uint32
	Pooling     string // "mean", "cls", "last" (default "mean")
}

func poolingFromString(s string) llama.PoolingType {
	switch s {
	case "cls":
		return llama.PoolingTypeCLS
	case "last":
		return llama.PoolingTypeLast
	case "none":
		return llama.PoolingTypeNone
	default:
		return llama.PoolingTypeMean
	}
}

// NewLocalEmbedder loads the model and prepares the context for embedding.
func NewLocalEmbedder(cfg LocalConfig) (*LocalEmbedder, error) {
	if cfg.ModelPath == "" {
		return nil, errors.New("model path is required")
	}

	le := &LocalEmbedder{
		pooling: poolingFromString(cfg.Pooling),
	}

	// Load shared library (idempotent in yzma)
	libPath := cfg.LibPath
	if libPath == "" {
		libPath = "llama" // let the dynamic linker find it
	}
	llama.Load(libPath)
	llama.LogSet(llama.LogSilent())
	llama.Init()
	le.libLoaded = true

	// Load model
	mp := llama.ModelDefaultParams()
	model, err := llama.ModelLoadFromFile(cfg.ModelPath, mp)
	if err != nil {
		return nil, fmt.Errorf("loading model %s: %w", cfg.ModelPath, err)
	}
	le.model = model

	// Determine embedding dimension
	le.nEmbd = llama.ModelNEmbd(model)
	if le.nEmbd == 0 {
		return nil, errors.New("model reports 0 embedding dimension")
	}

	// Create context with embeddings enabled
	ctxParams := llama.ContextDefaultParams()
	if cfg.ContextSize > 0 {
		ctxParams.NCtx = cfg.ContextSize
	} else {
		ctxParams.NCtx = 2048
	}
	if cfg.BatchSize > 0 {
		ctxParams.NBatch = cfg.BatchSize
	} else {
		ctxParams.NBatch = 512
	}
	ctxParams.PoolingType = le.pooling
	ctxParams.Embeddings = 1

	ctx, err := llama.InitFromModel(model, ctxParams)
	if err != nil {
		return nil, fmt.Errorf("initializing context: %w", err)
	}
	le.ctx = ctx

	// Get vocabulary for tokenization
	le.vocab = llama.ModelGetVocab(model)

	return le, nil
}

// Embed computes a single embedding vector for the given text.
func (e *LocalEmbedder) Embed(text string) ([]float32, error) {
	e.mu.Lock()
	defer e.mu.Unlock()

	tokens := llama.Tokenize(e.vocab, text, true, true)
	if len(tokens) == 0 {
		return nil, errors.New("tokenization produced no tokens")
	}

	batch := llama.BatchGetOne(tokens)
	if _, err := llama.Decode(e.ctx, batch); err != nil {
		return nil, fmt.Errorf("decoding: %w", err)
	}

	vec, err := llama.GetEmbeddingsSeq(e.ctx, 0, e.nEmbd)
	if err != nil {
		return nil, fmt.Errorf("getting embeddings: %w", err)
	}
	if vec == nil {
		return nil, errors.New("embeddings are nil (check pooling type)")
	}

	return normalize(vec), nil
}

// EmbedBatch computes embeddings for multiple texts sequentially.
// (Concurrent decoding on a single context is not possible; for true parallelism
// multiple contexts would be needed.)
func (e *LocalEmbedder) EmbedBatch(texts []string) ([][]float32, error) {
	result := make([][]float32, len(texts))
	for i, text := range texts {
		emb, err := e.Embed(text)
		if err != nil {
			return nil, fmt.Errorf("embedding text %d: %w", i, err)
		}
		result[i] = emb
	}
	return result, nil
}

// Dim returns the embedding dimension.
func (e *LocalEmbedder) Dim() int {
	return int(e.nEmbd)
}

// CountTokens returns the token count for a text using the model's tokenizer.
func (e *LocalEmbedder) CountTokens(text string) (int, error) {
	e.mu.Lock()
	defer e.mu.Unlock()
	tokens := llama.Tokenize(e.vocab, text, true, true)
	return len(tokens), nil
}

// Close releases model and context resources.
func (e *LocalEmbedder) Close() error {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.ctx != 0 {
		llama.Free(e.ctx)
	}
	if e.model != 0 {
		llama.ModelFree(e.model)
	}
	if e.libLoaded {
		llama.Close()
	}
	return nil
}

// normalize performs L2 normalization on a float32 vector.
func normalize(vec []float32) []float32 {
	var sum float64
	for _, v := range vec {
		sum += float64(v) * float64(v)
	}
	if sum == 0 {
		return vec
	}
	norm := float32(1.0 / math.Sqrt(sum))
	out := make([]float32, len(vec))
	for i, v := range vec {
		out[i] = v * norm
	}
	return out
}

// Compile-time interface check
var _ interface {
	Embed(text string) ([]float32, error)
	EmbedBatch(texts []string) ([][]float32, error)
	Dim() int
	Close() error
} = (*LocalEmbedder)(nil)

// EmbedFunc adapts LocalEmbedder to chromem-go's EmbeddingFunc type.
func (e *LocalEmbedder) EmbedFunc() func(context.Context, string) ([]float32, error) {
	return func(_ context.Context, text string) ([]float32, error) {
		return e.Embed(text)
	}
}
