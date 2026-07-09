package embed

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
)

// Embedder is the common interface for all embedding backends.
type Embedder interface {
	Embed(text string) ([]float32, error)
	EmbedBatch(texts []string) ([][]float32, error)
	Dim() int
	Close() error
}

// TokenCounter is implemented by embedders that have a real tokenizer.
type TokenCounter interface {
	CountTokens(text string) (int, error)
}

// APIEmbedder calls an OpenAI-compatible /v1/embeddings endpoint.
type APIEmbedder struct {
	baseURL string
	apiKey  string
	model   string
	dim     int
	client  *http.Client
}

// APIConfig holds parameters for the API embedder.
type APIConfig struct {
	BaseURL string // e.g. "http://localhost:11434/v1" (Ollama) or "https://api.openai.com/v1"
	APIKey  string
	Model   string // e.g. "nomic-embed-text", "text-embedding-3-small", "bge-m3"
}

// NewAPIEmbedder creates an API-based embedder. The dimension is determined
// lazily on the first Embed call, or can be set manually via SetDim.
func NewAPIEmbedder(cfg APIConfig) (*APIEmbedder, error) {
	if cfg.BaseURL == "" {
		return nil, errors.New("API base URL is required")
	}
	if cfg.Model == "" {
		return nil, errors.New("model name is required")
	}
	return &APIEmbedder{
		baseURL: cfg.BaseURL,
		apiKey:  cfg.APIKey,
		model:   cfg.Model,
		client:  &http.Client{},
	}, nil
}

// SetDim manually sets the embedding dimension (useful to avoid a probe call).
func (a *APIEmbedder) SetDim(d int) { a.dim = d }

type embeddingsRequest struct {
	Model string   `json:"model"`
	Input []string `json:"input"`
}

type embeddingsResponse struct {
	Data []struct {
		Embedding []float32 `json:"embedding"`
	} `json:"data"`
}

func (a *APIEmbedder) embedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	reqBody := embeddingsRequest{
		Model: a.model,
		Input: texts,
	}
	bodyBytes, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("marshaling request: %w", err)
	}

	url := a.baseURL + "/embeddings"
	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(bodyBytes))
	if err != nil {
		return nil, fmt.Errorf("creating request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	if a.apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+a.apiKey)
	}

	resp, err := a.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("API request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("API returned status %d: %s", resp.StatusCode, string(respBody))
	}

	var result embeddingsResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("decoding response: %w", err)
	}

	if len(result.Data) == 0 {
		return nil, errors.New("API returned no embeddings")
	}

	embeddings := make([][]float32, len(result.Data))
	for i, d := range result.Data {
		embeddings[i] = d.Embedding
	}

	if a.dim == 0 {
		a.dim = len(embeddings[0])
	}

	return embeddings, nil
}

func (a *APIEmbedder) Embed(text string) ([]float32, error) {
	embs, err := a.embedBatch(context.Background(), []string{text})
	if err != nil {
		return nil, err
	}
	return embs[0], nil
}

func (a *APIEmbedder) EmbedBatch(texts []string) ([][]float32, error) {
	return a.embedBatch(context.Background(), texts)
}

func (a *APIEmbedder) Dim() int {
	if a.dim > 0 {
		return a.dim
	}
	// Probe to determine dimension
	emb, err := a.Embed("dimension probe")
	if err != nil {
		return 0
	}
	return len(emb)
}

func (a *APIEmbedder) Close() error { return nil }
