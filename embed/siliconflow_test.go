package embed

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestSiliconFlowTextEmbed(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/embeddings" {
			t.Errorf("path = %s, want /embeddings", r.URL.Path)
		}
		if r.Header.Get("Authorization") != "Bearer test-key" {
			t.Errorf("auth header = %s", r.Header.Get("Authorization"))
		}

		var req sfRequest
		json.NewDecoder(r.Body).Decode(&req)
		if req.Model != "Qwen/Qwen3-VL-Embedding-8B" {
			t.Errorf("model = %s", req.Model)
		}
		if len(req.Input) != 2 {
			t.Fatalf("expected 2 inputs, got %d", len(req.Input))
		}
		if req.Input[0] != "hello" {
			t.Errorf("input[0] = %v, want 'hello'", req.Input[0])
		}
		if req.Dimensions != 768 {
			t.Errorf("dimensions = %d, want 768", req.Dimensions)
		}

		resp := sfResponse{}
		resp.Data = []struct {
			Embedding []float32 `json:"embedding"`
			Index     int       `json:"index"`
		}{
			{Embedding: []float32{0.1, 0.2}, Index: 0},
			{Embedding: []float32{0.3, 0.4}, Index: 1},
		}
		json.NewEncoder(w).Encode(resp)
	}))
	defer srv.Close()

	s, err := NewSiliconFlowEmbedder(SiliconFlowConfig{
		APIKey:     "test-key",
		Model:      "Qwen/Qwen3-VL-Embedding-8B",
		BaseURL:    srv.URL,
		Dimensions: 768,
	})
	if err != nil {
		t.Fatal(err)
	}

	results, err := s.EmbedBatch([]string{"hello", "world"})
	if err != nil {
		t.Fatal(err)
	}
	if len(results) != 2 {
		t.Fatalf("got %d results", len(results))
	}
	if s.Dim() != 2 {
		t.Errorf("Dim() = %d, want 2", s.Dim())
	}
}

func TestSiliconFlowVLTextOnly(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req sfRequest
		json.NewDecoder(r.Body).Decode(&req)

		// Single text should be plain string
		if len(req.Input) != 1 {
			t.Fatalf("expected 1 input, got %d", len(req.Input))
		}
		if req.Input[0] != "hello world" {
			t.Errorf("input[0] = %v, want 'hello world'", req.Input[0])
		}

		resp := sfResponse{}
		resp.Data = []struct {
			Embedding []float32 `json:"embedding"`
			Index     int       `json:"index"`
		}{
			{Embedding: []float32{0.1, 0.2}, Index: 0},
		}
		json.NewEncoder(w).Encode(resp)
	}))
	defer srv.Close()

	s, _ := NewSiliconFlowEmbedder(SiliconFlowConfig{
		APIKey:  "test-key",
		Model:   "Qwen/Qwen3-VL-Embedding-8B",
		BaseURL: srv.URL,
	})

	emb, err := s.Embed("hello world")
	if err != nil {
		t.Fatal(err)
	}
	if len(emb) != 2 {
		t.Errorf("embedding length = %d", len(emb))
	}
}

func TestSiliconFlowVLImage(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req sfRequest
		json.NewDecoder(r.Body).Decode(&req)

		// Should have mixed content: text object + image object
		if len(req.Input) != 2 {
			t.Fatalf("expected 2 inputs, got %d", len(req.Input))
		}
		// Check text object
		textObj, ok := req.Input[0].(map[string]interface{})
		if !ok {
			t.Fatalf("input[0] should be object, got %T", req.Input[0])
		}
		if textObj["text"] != "a cat photo" {
			t.Errorf("text = %v", textObj["text"])
		}
		// Check image object
		imgObj, ok := req.Input[1].(map[string]interface{})
		if !ok {
			t.Fatalf("input[1] should be object, got %T", req.Input[1])
		}
		if imgObj["image"] != "https://example.com/cat.jpg" {
			t.Errorf("image = %v", imgObj["image"])
		}

		resp := sfResponse{}
		resp.Data = []struct {
			Embedding []float32 `json:"embedding"`
			Index     int       `json:"index"`
		}{
			{Embedding: []float32{0.5, 0.6}, Index: 0},
			{Embedding: []float32{0.7, 0.8}, Index: 1},
		}
		json.NewEncoder(w).Encode(resp)
	}))
	defer srv.Close()

	s, _ := NewSiliconFlowEmbedder(SiliconFlowConfig{
		APIKey:  "test-key",
		Model:   "Qwen/Qwen3-VL-Embedding-8B",
		BaseURL: srv.URL,
	})

	contents := []VLContent{
		{Text: "a cat photo"},
		{Image: "https://example.com/cat.jpg"},
	}
	results, err := s.EmbedVLBatch(contents)
	if err != nil {
		t.Fatal(err)
	}
	if len(results) != 2 {
		t.Fatalf("expected 2 results, got %d", len(results))
	}
}

func TestNewSiliconFlowEmbedder(t *testing.T) {
	_, err := NewSiliconFlowEmbedder(SiliconFlowConfig{Model: "x"})
	if err == nil {
		t.Error("expected error for missing API key")
	}
	_, err = NewSiliconFlowEmbedder(SiliconFlowConfig{APIKey: "k"})
	if err == nil {
		t.Error("expected error for missing model")
	}
	s, err := NewSiliconFlowEmbedder(SiliconFlowConfig{
		APIKey: "k",
		Model:  "Qwen/Qwen3-VL-Embedding-8B",
	})
	if err != nil {
		t.Fatal(err)
	}
	if s.baseURL != "https://api.siliconflow.cn/v1" {
		t.Errorf("baseURL = %s, want https://api.siliconflow.cn/v1", s.baseURL)
	}
}
