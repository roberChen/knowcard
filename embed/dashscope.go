package embed

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strings"
)

// VLContent represents a single piece of multimodal content for VL embedding.
// At least one of Text, Image, or Video should be set.
type VLContent struct {
	Text  string `json:"text,omitempty"`
	Image string `json:"image,omitempty"` // URL or base64 data URI ("data:image/...")
	Video string `json:"video,omitempty"` // URL
}

// MultimodalEmbedder extends Embedder with vision-language embedding support.
type MultimodalEmbedder interface {
	Embedder
	// EmbedVL embeds a list of multimodal content items as a single fused vector.
	// When fusion is enabled, all items are combined into one embedding.
	// When fusion is disabled, each item gets its own embedding (use EmbedVLBatch).
	EmbedVL(contents []VLContent) ([]float32, error)
	// EmbedVLBatch embeds each content item independently, returning one vector per item.
	EmbedVLBatch(contents []VLContent) ([][]float32, error)
}

// DashScopeEmbedder calls DashScope native API for Qwen embedding models.
// Supports both text embedding (text-embedding-v4) and multimodal VL embedding
// (qwen3-vl-embedding, tongyi-embedding-vision-plus, etc.).
type DashScopeEmbedder struct {
	apiKey        string
	model         string
	baseURL       string // e.g. "https://dashscope.aliyuncs.com/api/v1"
	dimensions    int
	instruct      string
	isMultimodal  bool   // true = use multimodal endpoint, false = text-only
	enableFusion  bool   // for qwen3-vl-embedding: fuse all inputs into one vector
	dim           int
	client        *http.Client
}

// DashScopeConfig holds parameters for the DashScope embedder.
type DashScopeConfig struct {
	APIKey        string
	Model         string // e.g. "text-embedding-v4", "qwen3-vl-embedding", "tongyi-embedding-vision-plus"
	International bool   // true = intl endpoint, false = domestic
	Dimensions    int    // MRL dimension (0 = model default)
	Instruct      string // task instruction
	EnableFusion  bool   // qwen3-vl-embedding only: fuse all inputs into one vector
}

// known multimodal models — everything else is treated as text-only
var dashscopeMultimodalModels = map[string]bool{
	"qwen3-vl-embedding":                    true,
	"qwen2.5-vl-embedding":                  true,
	"multimodal-embedding-v1":               true,
	"multimodal-embedding-one-peace-v1":     true,
	"tongyi-embedding-vision-plus":          true,
	"tongyi-embedding-vision-flash":         true,
	"tongyi-embedding-vision-plus-2026-03-06":  true,
	"tongyi-embedding-vision-flash-2026-03-06": true,
}

// NewDashScopeEmbedder creates a DashScope API embedder.
func NewDashScopeEmbedder(cfg DashScopeConfig) (*DashScopeEmbedder, error) {
	if cfg.APIKey == "" {
		return nil, errors.New("DashScope API key is required")
	}
	if cfg.Model == "" {
		return nil, errors.New("model name is required")
	}

	baseURL := "https://dashscope.aliyuncs.com/api/v1"
	if cfg.International {
		baseURL = "https://dashscope-intl.aliyuncs.com/api/v1"
	}

	return &DashScopeEmbedder{
		apiKey:       cfg.APIKey,
		model:        cfg.Model,
		baseURL:      baseURL,
		dimensions:   cfg.Dimensions,
		instruct:     cfg.Instruct,
		isMultimodal: dashscopeMultimodalModels[strings.ToLower(cfg.Model)],
		enableFusion: cfg.EnableFusion,
		client:       &http.Client{},
	}, nil
}

// --- Text embedding (native DashScope format) ---

type dsTextRequest struct {
	Model      string           `json:"model"`
	Input      dsTextInput      `json:"input"`
	Parameters *dsTextParams    `json:"parameters,omitempty"`
}

type dsTextInput struct {
	Texts []string `json:"texts"`
}

type dsTextParams struct {
	Dimension  int    `json:"dimension,omitempty"`
	TextType   string `json:"text_type,omitempty"` // "query" or "document"
	OutputType string `json:"output_type,omitempty"` // "dense", "sparse", "dense&sparse"
	Instruct   string `json:"instruct,omitempty"`
}

type dsTextResponse struct {
	StatusCode int    `json:"status_code"`
	Code       string `json:"code"`
	Message    string `json:"message"`
	Output     struct {
		Embeddings []struct {
			Embedding   []float32 `json:"embedding"`
			TextIndex   int       `json:"text_index"`
		} `json:"embeddings"`
	} `json:"output"`
}

func (d *DashScopeEmbedder) embedText(ctx context.Context, texts []string) ([][]float32, error) {
	reqBody := dsTextRequest{
		Model: d.model,
		Input: dsTextInput{Texts: texts},
	}
	if d.dimensions > 0 || d.instruct != "" {
		params := &dsTextParams{}
		if d.dimensions > 0 {
			params.Dimension = d.dimensions
		}
		if d.instruct != "" {
			params.Instruct = d.instruct
		}
		reqBody.Parameters = params
	}

	bodyBytes, _ := json.Marshal(reqBody)
	url := d.baseURL + "/services/embeddings/text-embedding/text-embedding"

	req, _ := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(bodyBytes))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+d.apiKey)

	resp, err := d.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("DashScope text embedding request: %w", err)
	}
	defer resp.Body.Close()

	var result dsTextResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("decoding DashScope response: %w", err)
	}

	if result.StatusCode != 0 && result.StatusCode != 200 {
		return nil, fmt.Errorf("DashScope error %d: %s (code: %s)", result.StatusCode, result.Message, result.Code)
	}

	if len(result.Output.Embeddings) == 0 {
		return nil, errors.New("DashScope returned no embeddings")
	}

	out := make([][]float32, len(result.Output.Embeddings))
	for _, emb := range result.Output.Embeddings {
		out[emb.TextIndex] = emb.Embedding
	}

	if d.dim == 0 && len(out) > 0 {
		d.dim = len(out[0])
	}

	return out, nil
}

// --- Multimodal VL embedding (native DashScope format) ---

type dsVLRequest struct {
	Model      string          `json:"model"`
	Input      dsVLInput       `json:"input"`
	Parameters *dsVLParams     `json:"parameters,omitempty"`
}

type dsVLInput struct {
	Contents []map[string]interface{} `json:"contents"`
}

type dsVLParams struct {
	Dimension     int    `json:"dimension,omitempty"`
	EnableFusion  bool   `json:"enable_fusion,omitempty"`
	Instruct      string `json:"instruct,omitempty"`
}

type dsVLResponse struct {
	Code    string `json:"code"`
	Message string `json:"message"`
	Output  struct {
		Embeddings []struct {
			Index     int       `json:"index"`
			Embedding []float32 `json:"embedding"`
			Type      string    `json:"type"`
		} `json:"embeddings"`
	} `json:"output"`
}

func (d *DashScopeEmbedder) embedVL(ctx context.Context, contents []VLContent) ([][]float32, error) {
	// Convert VLContent to DashScope format
	rawContents := make([]map[string]interface{}, len(contents))
	for i, c := range contents {
		item := make(map[string]interface{})
		if c.Text != "" {
			item["text"] = c.Text
		}
		if c.Image != "" {
			item["image"] = c.Image
		}
		if c.Video != "" {
			item["video"] = c.Video
		}
		rawContents[i] = item
	}

	reqBody := dsVLRequest{
		Model: d.model,
		Input: dsVLInput{Contents: rawContents},
	}

	if d.dimensions > 0 || d.enableFusion || d.instruct != "" {
		params := &dsVLParams{}
		if d.dimensions > 0 {
			params.Dimension = d.dimensions
		}
		if d.enableFusion {
			params.EnableFusion = true
		}
		if d.instruct != "" {
			params.Instruct = d.instruct
		}
		reqBody.Parameters = params
	}

	bodyBytes, _ := json.Marshal(reqBody)
	url := d.baseURL + "/services/embeddings/multimodal-embedding/multimodal-embedding"

	req, _ := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(bodyBytes))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+d.apiKey)

	resp, err := d.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("DashScope VL embedding request: %w", err)
	}
	defer resp.Body.Close()

	var result dsVLResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("decoding DashScope VL response: %w", err)
	}

	if result.Code != "" {
		return nil, fmt.Errorf("DashScope VL error: %s (code: %s)", result.Message, result.Code)
	}

	if len(result.Output.Embeddings) == 0 {
		return nil, errors.New("DashScope VL returned no embeddings")
	}

	out := make([][]float32, len(result.Output.Embeddings))
	for _, emb := range result.Output.Embeddings {
		out[emb.Index] = emb.Embedding
	}

	if d.dim == 0 && len(out) > 0 {
		d.dim = len(out[0])
	}

	return out, nil
}

// --- Embedder interface methods ---

func (d *DashScopeEmbedder) Embed(text string) ([]float32, error) {
	if d.isMultimodal {
		// VL models can still embed text-only via multimodal endpoint
		results, err := d.embedVL(context.Background(), []VLContent{{Text: text}})
		if err != nil {
			return nil, err
		}
		return results[0], nil
	}
	results, err := d.embedText(context.Background(), []string{text})
	if err != nil {
		return nil, err
	}
	return results[0], nil
}

func (d *DashScopeEmbedder) EmbedBatch(texts []string) ([][]float32, error) {
	if d.isMultimodal {
		contents := make([]VLContent, len(texts))
		for i, t := range texts {
			contents[i] = VLContent{Text: t}
		}
		return d.embedVL(context.Background(), contents)
	}
	return d.embedText(context.Background(), texts)
}

func (d *DashScopeEmbedder) Dim() int {
	if d.dim > 0 {
		return d.dim
	}
	// Probe to determine dimension
	emb, err := d.Embed("dimension probe")
	if err != nil {
		return 0
	}
	return len(emb)
}

func (d *DashScopeEmbedder) Close() error { return nil }

// --- MultimodalEmbedder interface methods ---

func (d *DashScopeEmbedder) EmbedVL(contents []VLContent) ([]float32, error) {
	if !d.isMultimodal {
		return nil, fmt.Errorf("model %s does not support multimodal embedding", d.model)
	}
	// With fusion: all contents → one vector
	if d.enableFusion {
		results, err := d.embedVL(context.Background(), contents)
		if err != nil {
			return nil, err
		}
		return results[0], nil
	}
	// Without fusion: just embed the first item
	results, err := d.embedVL(context.Background(), contents)
	if err != nil {
		return nil, err
	}
	if len(results) > 0 {
		return results[0], nil
	}
	return nil, errors.New("no embedding returned")
}

func (d *DashScopeEmbedder) EmbedVLBatch(contents []VLContent) ([][]float32, error) {
	if !d.isMultimodal {
		return nil, fmt.Errorf("model %s does not support multimodal embedding", d.model)
	}
	return d.embedVL(context.Background(), contents)
}

// Compile-time interface checks
var _ Embedder = (*DashScopeEmbedder)(nil)
var _ MultimodalEmbedder = (*DashScopeEmbedder)(nil)
