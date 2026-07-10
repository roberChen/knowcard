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

// SiliconFlowEmbedder calls SiliconFlow's OpenAI-compatible /v1/embeddings
// endpoint with extended support for Qwen3-VL-Embedding multimodal inputs.
//
// For text-only input it behaves like a standard OpenAI-compatible embedder.
// For VL input, the "input" field accepts objects like {"text": "..."} or
// {"image": "https://..."} instead of plain strings.
type SiliconFlowEmbedder struct {
	apiKey     string
	model      string
	baseURL    string
	dimensions int
	dim        int
	client     *http.Client
}

// SiliconFlowConfig holds parameters for the SiliconFlow embedder.
type SiliconFlowConfig struct {
	APIKey     string
	Model      string // e.g. "Qwen/Qwen3-VL-Embedding-8B"
	BaseURL    string // default: https://api.siliconflow.cn/v1
	Dimensions int    // MRL dimension (0 = model default)
}

// NewSiliconFlowEmbedder creates a SiliconFlow API embedder.
func NewSiliconFlowEmbedder(cfg SiliconFlowConfig) (*SiliconFlowEmbedder, error) {
	if cfg.APIKey == "" {
		return nil, errors.New("SiliconFlow API key is required")
	}
	if cfg.Model == "" {
		return nil, errors.New("model name is required")
	}
	baseURL := cfg.BaseURL
	if baseURL == "" {
		baseURL = "https://api.siliconflow.cn/v1"
	}
	return &SiliconFlowEmbedder{
		apiKey:     cfg.APIKey,
		model:      cfg.Model,
		baseURL:    baseURL,
		dimensions: cfg.Dimensions,
		client:     &http.Client{},
	}, nil
}

// --- Request/Response types ---

// sfRequest is the SiliconFlow embeddings request body.
// The Input field is []interface{} to allow mixed string/object content.
type sfRequest struct {
	Model      string        `json:"model"`
	Input      []interface{} `json:"input"`
	Dimensions int           `json:"dimensions,omitempty"`
}

type sfResponse struct {
	Model string `json:"model"`
	Data  []struct {
		Embedding []float32 `json:"embedding"`
		Index     int       `json:"index"`
	} `json:"data"`
	Usage struct {
		TotalTokens int `json:"total_tokens"`
	} `json:"usage"`
}

func (s *SiliconFlowEmbedder) callAPI(ctx context.Context, input []interface{}) ([][]float32, error) {
	reqBody := sfRequest{
		Model: s.model,
		Input: input,
	}
	if s.dimensions > 0 {
		reqBody.Dimensions = s.dimensions
	}

	bodyBytes, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("marshaling request: %w", err)
	}

	url := s.baseURL + "/embeddings"
	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(bodyBytes))
	if err != nil {
		return nil, fmt.Errorf("creating request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+s.apiKey)

	resp, err := s.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("SiliconFlow API request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("SiliconFlow API returned status %d: %s", resp.StatusCode, string(respBody))
	}

	var result sfResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("decoding response: %w", err)
	}

	if len(result.Data) == 0 {
		return nil, errors.New("SiliconFlow returned no embeddings")
	}

	out := make([][]float32, len(result.Data))
	for _, d := range result.Data {
		out[d.Index] = d.Embedding
	}

	if s.dim == 0 && len(out) > 0 {
		s.dim = len(out[0])
	}

	return out, nil
}

// --- Embedder interface ---

func (s *SiliconFlowEmbedder) Embed(text string) ([]float32, error) {
	results, err := s.callAPI(context.Background(), []interface{}{text})
	if err != nil {
		return nil, err
	}
	return results[0], nil
}

func (s *SiliconFlowEmbedder) EmbedBatch(texts []string) ([][]float32, error) {
	input := make([]interface{}, len(texts))
	for i, t := range texts {
		input[i] = t
	}
	return s.callAPI(context.Background(), input)
}

func (s *SiliconFlowEmbedder) Dim() int {
	if s.dim > 0 {
		return s.dim
	}
	emb, err := s.Embed("dimension probe")
	if err != nil {
		return 0
	}
	return len(emb)
}

func (s *SiliconFlowEmbedder) Close() error { return nil }

// --- MultimodalEmbedder interface ---

func (s *SiliconFlowEmbedder) EmbedVL(contents []VLContent) ([]float32, error) {
	input := make([]interface{}, len(contents))
	for i, c := range contents {
		obj := make(map[string]string)
		if c.Text != "" {
			obj["text"] = c.Text
		}
		if c.Image != "" {
			obj["image"] = c.Image
		}
		input[i] = obj
	}
	results, err := s.callAPI(context.Background(), input)
	if err != nil {
		return nil, err
	}
	if len(results) > 0 {
		return results[0], nil
	}
	return nil, errors.New("no embedding returned")
}

func (s *SiliconFlowEmbedder) EmbedVLBatch(contents []VLContent) ([][]float32, error) {
	input := make([]interface{}, len(contents))
	for i, c := range contents {
		obj := make(map[string]string)
		if c.Text != "" {
			obj["text"] = c.Text
		}
		if c.Image != "" {
			obj["image"] = c.Image
		}
		input[i] = obj
	}
	return s.callAPI(context.Background(), input)
}

// Compile-time interface checks
var _ Embedder = (*SiliconFlowEmbedder)(nil)
var _ MultimodalEmbedder = (*SiliconFlowEmbedder)(nil)
