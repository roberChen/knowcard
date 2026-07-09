package embed

import "fmt"

// Config holds parameters for creating an Embedder.
// This struct is independent of the main knowcard.Config to avoid import cycles.
type Config struct {
	Backend    string // "local", "ollama", "openai", "custom"
	ModelPath  string // for local: path to .gguf file
	LibPath    string // for local: path to libllama shared library
	ContextSize uint32
	BatchSize   uint32
	APIBase    string // for API backends
	APIKey     string
	Model      string // model name for API backends
}

// New creates an Embedder from the embed Config.
func New(cfg Config) (Embedder, error) {
	switch cfg.Backend {
	case "local":
		return NewLocalEmbedder(LocalConfig{
			ModelPath:   cfg.ModelPath,
			LibPath:     cfg.LibPath,
			ContextSize: cfg.ContextSize,
			BatchSize:   cfg.BatchSize,
			Pooling:     "mean",
		})
	case "ollama", "openai", "custom":
		base := cfg.APIBase
		if base == "" && cfg.Backend == "ollama" {
			base = "http://localhost:11434/v1"
		}
		return NewAPIEmbedder(APIConfig{
			BaseURL: base,
			APIKey:  cfg.APIKey,
			Model:   cfg.Model,
		})
	default:
		return nil, fmt.Errorf("unknown embed backend: %s", cfg.Backend)
	}
}
