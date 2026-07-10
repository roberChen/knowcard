package embed

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestDashScopeTextEmbed(t *testing.T) {
	// Mock DashScope text embedding endpoint
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "Bearer test-key" {
			t.Errorf("missing/wrong auth header: %s", r.Header.Get("Authorization"))
		}

		var req dsTextRequest
		json.NewDecoder(r.Body).Decode(&req)
		if req.Model != "text-embedding-v4" {
			t.Errorf("model = %s, want text-embedding-v4", req.Model)
		}
		if len(req.Input.Texts) != 2 {
			t.Errorf("expected 2 texts, got %d", len(req.Input.Texts))
		}
		// Check dimension param
		if req.Parameters == nil || req.Parameters.Dimension != 1024 {
			t.Errorf("expected dimension=1024 in parameters")
		}

		resp := dsTextResponse{StatusCode: 200}
		resp.Output.Embeddings = []struct {
			Embedding []float32 `json:"embedding"`
			TextIndex int       `json:"text_index"`
		}{
			{Embedding: []float32{0.1, 0.2, 0.3}, TextIndex: 0},
			{Embedding: []float32{0.4, 0.5, 0.6}, TextIndex: 1},
		}
		json.NewEncoder(w).Encode(resp)
	}))
	defer srv.Close()

	d := &DashScopeEmbedder{
		apiKey:     "test-key",
		model:      "text-embedding-v4",
		baseURL:    srv.URL,
		dimensions: 1024,
		client:     srv.Client(),
	}

	results, err := d.embedText(context.Background(), []string{"hello", "world"})
	if err != nil {
		t.Fatalf("embedText failed: %v", err)
	}
	if len(results) != 2 {
		t.Fatalf("expected 2 results, got %d", len(results))
	}
	if results[0][0] != 0.1 {
		t.Errorf("result[0][0] = %f, want 0.1", results[0][0])
	}
	if d.Dim() != 3 {
		t.Errorf("Dim() = %d, want 3", d.Dim())
	}
}

func TestDashScopeVLEmbed(t *testing.T) {
	// Mock DashScope multimodal embedding endpoint
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req dsVLRequest
		json.NewDecoder(r.Body).Decode(&req)

		if req.Model != "qwen3-vl-embedding" {
			t.Errorf("model = %s, want qwen3-vl-embedding", req.Model)
		}
		if len(req.Input.Contents) != 2 {
			t.Errorf("expected 2 contents, got %d", len(req.Input.Contents))
		}
		// Check text content
		if req.Input.Contents[0]["text"] != "a photo of a cat" {
			t.Errorf("content[0].text = %v", req.Input.Contents[0]["text"])
		}
		// Check image content
		if req.Input.Contents[1]["image"] != "https://example.com/cat.jpg" {
			t.Errorf("content[1].image = %v", req.Input.Contents[1]["image"])
		}
		// Check fusion param
		if req.Parameters == nil || !req.Parameters.EnableFusion {
			t.Error("expected enable_fusion=true")
		}

		resp := dsVLResponse{}
		resp.Output.Embeddings = []struct {
			Index     int       `json:"index"`
			Embedding []float32 `json:"embedding"`
			Type      string    `json:"type"`
		}{
			{Index: 0, Embedding: []float32{0.1, 0.2, 0.3, 0.4}, Type: "text"},
			{Index: 1, Embedding: []float32{0.5, 0.6, 0.7, 0.8}, Type: "image"},
		}
		json.NewEncoder(w).Encode(resp)
	}))
	defer srv.Close()

	d := &DashScopeEmbedder{
		apiKey:       "test-key",
		model:        "qwen3-vl-embedding",
		baseURL:      srv.URL,
		isMultimodal: true,
		enableFusion: true,
		client:       srv.Client(),
	}

	contents := []VLContent{
		{Text: "a photo of a cat"},
		{Image: "https://example.com/cat.jpg"},
	}
	results, err := d.embedVL(context.Background(), contents)
	if err != nil {
		t.Fatalf("embedVL failed: %v", err)
	}
	if len(results) != 2 {
		t.Fatalf("expected 2 results, got %d", len(results))
	}
	if d.Dim() != 4 {
		t.Errorf("Dim() = %d, want 4", d.Dim())
	}
}

func TestDashScopeEmbedTextViaVLEndpoint(t *testing.T) {
	// VL model should route text through multimodal endpoint
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Verify it hits the multimodal endpoint
		if r.URL.Path != "/services/embeddings/multimodal-embedding/multimodal-embedding" {
			t.Errorf("VL model should use multimodal endpoint, got %s", r.URL.Path)
		}
		resp := dsVLResponse{}
		resp.Output.Embeddings = []struct {
			Index     int       `json:"index"`
			Embedding []float32 `json:"embedding"`
			Type      string    `json:"type"`
		}{
			{Index: 0, Embedding: []float32{0.1, 0.2}, Type: "text"},
		}
		json.NewEncoder(w).Encode(resp)
	}))
	defer srv.Close()

	d := &DashScopeEmbedder{
		apiKey:       "test-key",
		model:        "tongyi-embedding-vision-plus",
		baseURL:      srv.URL,
		isMultimodal: true,
		client:       srv.Client(),
	}

	emb, err := d.Embed("hello")
	if err != nil {
		t.Fatalf("Embed failed: %v", err)
	}
	if len(emb) != 2 {
		t.Errorf("embedding length = %d, want 2", len(emb))
	}
}

func TestDashScopeEmbedVLNonMultimodal(t *testing.T) {
	d := &DashScopeEmbedder{
		apiKey:       "test-key",
		model:        "text-embedding-v4",
		isMultimodal: false,
		client:       &http.Client{},
	}

	// Should error when trying VL on a text-only model
	_, err := d.EmbedVL([]VLContent{{Text: "hello"}})
	if err == nil {
		t.Error("expected error calling EmbedVL on non-multimodal model")
	}
}

func TestNewDashScopeEmbedder(t *testing.T) {
	_, err := NewDashScopeEmbedder(DashScopeConfig{
		APIKey: "",
		Model:  "text-embedding-v4",
	})
	if err == nil {
		t.Error("expected error for missing API key")
	}

	_, err = NewDashScopeEmbedder(DashScopeConfig{
		APIKey: "sk-test",
		Model:  "",
	})
	if err == nil {
		t.Error("expected error for missing model")
	}

	d, err := NewDashScopeEmbedder(DashScopeConfig{
		APIKey:        "sk-test",
		Model:         "qwen3-vl-embedding",
		International: true,
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !d.isMultimodal {
		t.Error("qwen3-vl-embedding should be detected as multimodal")
	}
	if !d.enableFusion {
		// enableFusion defaults to false unless explicitly set, that's fine
	}
	if d.baseURL != "https://dashscope-intl.aliyuncs.com/api/v1" {
		t.Errorf("baseURL = %s, want intl endpoint", d.baseURL)
	}

	d2, _ := NewDashScopeEmbedder(DashScopeConfig{
		APIKey: "sk-test",
		Model:  "text-embedding-v4",
	})
	if d2.isMultimodal {
		t.Error("text-embedding-v4 should NOT be detected as multimodal")
	}
	if d2.baseURL != "https://dashscope.aliyuncs.com/api/v1" {
		t.Errorf("baseURL = %s, want domestic endpoint", d2.baseURL)
	}
}
