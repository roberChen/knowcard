package embed

import "fmt"

// Config holds parameters for creating an Embedder.
// This struct is independent of the main knowcard.Config to avoid import cycles.
type Config struct {
	Backend    string // "local", "ollama", "openai", "custom", "qwen_cloud"
	ModelPath  string // for local: path to .gguf file
	LibPath    string // for local: path to libllama shared library
	ContextSize uint32
	BatchSize   uint32
	Pooling    string // for local: "mean", "cls", "last" (default: "last")
	APIBase    string // for API backends
	APIKey     string
	Model      string // model name for API backends

	// MRL dimensions (0 = model default)
	Dimensions int

	// DashScope-specific
	DashScopeInternational bool
	Instruct              string
	EnableFusion          bool // qwen3-vl-embedding: fuse all inputs into one vector
}

// New creates an Embedder from the embed Config.
func New(cfg Config) (Embedder, error) {
	switch cfg.Backend {
	case "local":
		pooling := cfg.Pooling
		if pooling == "" {
			pooling = "last" // default to last-pooling (correct for Qwen embedding models)
		}
		return NewLocalEmbedder(LocalConfig{
			ModelPath:   cfg.ModelPath,
			LibPath:     cfg.LibPath,
			ContextSize: cfg.ContextSize,
			BatchSize:   cfg.BatchSize,
			Pooling:     pooling,
		})
	case "ollama", "openai", "custom":
		base := cfg.APIBase
		if base == "" && cfg.Backend == "ollama" {
			base = "http://localhost:11434/v1"
		}
		ae, err := NewAPIEmbedder(APIConfig{
			BaseURL: base,
			APIKey:  cfg.APIKey,
			Model:   cfg.Model,
		})
		if err != nil {
			return nil, err
		}
		if cfg.Dimensions > 0 {
			ae.SetDim(cfg.Dimensions)
		}
		return ae, nil
	case "qwen_cloud":
		return NewDashScopeEmbedder(DashScopeConfig{
			APIKey:        cfg.APIKey,
			Model:         cfg.Model,
			International: cfg.DashScopeInternational,
			Dimensions:    cfg.Dimensions,
			Instruct:      cfg.Instruct,
			EnableFusion:  cfg.EnableFusion,
		})
	case "siliconflow":
		return NewSiliconFlowEmbedder(SiliconFlowConfig{
			APIKey:     cfg.APIKey,
			Model:      cfg.Model,
			BaseURL:    cfg.APIBase,
			Dimensions: cfg.Dimensions,
		})
	default:
		return nil, fmt.Errorf("unknown embed backend: %s", cfg.Backend)
	}
}
