package knowcard

import (
	"fmt"
	"os"
	"path/filepath"

	"gopkg.in/yaml.v3"
)

// EmbedBackend selects how embeddings are computed.
type EmbedBackend string

const (
	EmbedLocal   EmbedBackend = "local"   // yzma + GGUF, CPU inference
	EmbedOllama  EmbedBackend = "ollama"  // Ollama API
	EmbedOpenAI  EmbedBackend = "openai"  // OpenAI API
	EmbedCustom  EmbedBackend = "custom"  // OpenAI-compatible API
)

// Config defines all runtime configuration for knowcard.
type Config struct {
	// Root is the data directory, typically ~/.knowcard
	Root string `yaml:"root"`

	// Embed backend configuration
	Embed EmbedConfig `yaml:"embed"`

	// RRF constant for reciprocal rank fusion (default 60)
	RRFK int `yaml:"rrf_k"`

	// Number of candidates each retrieval lane fetches before fusion (default 30)
	CandidatePool int `yaml:"candidate_pool"`
}

type EmbedConfig struct {
	Backend EmbedBackend `yaml:"backend"`

	// Local (yzma) settings
	ModelPath string `yaml:"model_path,omitempty"` // path to .gguf file
	LibPath   string `yaml:"lib_path,omitempty"`   // path to libllama shared library
	ContextSize uint32 `yaml:"context_size,omitempty"`
	BatchSize   uint32 `yaml:"batch_size,omitempty"`

	// API settings (Ollama / OpenAI / custom)
	APIBase string `yaml:"api_base,omitempty"`
	APIKey  string `yaml:"api_key,omitempty"`
	Model   string `yaml:"model,omitempty"`
}

// DefaultConfig returns a configuration with sensible defaults.
func DefaultConfig() Config {
	home, _ := os.UserHomeDir()
	root := filepath.Join(home, ".knowcard")
	return Config{
		Root: root,
		Embed: EmbedConfig{
			Backend:      EmbedLocal,
			ModelPath:    filepath.Join(root, "models", "bge-m3.Q8_0.gguf"),
			ContextSize:  2048,
			BatchSize:    512,
		},
		RRFK:          60,
		CandidatePool: 30,
	}
}

// CardsDir returns the path to the cards directory.
func (c *Config) CardsDir() string {
	return filepath.Join(c.Root, "cards")
}

// VcsDir returns the path to the git metadata directory (separated from cards).
func (c *Config) VcsDir() string {
	return filepath.Join(c.Root, "_vcs")
}

// IndexDir returns the path to the derived index directory.
func (c *Config) IndexDir() string {
	return filepath.Join(c.Root, "index")
}

// ChromemDir returns the path to the chromem-go persistence directory.
func (c *Config) ChromemDir() string {
	return filepath.Join(c.IndexDir(), "chromem")
}

// ModelsDir returns the path to the model cache directory.
func (c *Config) ModelsDir() string {
	return filepath.Join(c.Root, "models")
}

// ConfigPath returns the path to the config file.
func (c *Config) ConfigPath() string {
	return filepath.Join(c.Root, "knowcard.yaml")
}

// Load reads the config file from disk, falling back to defaults.
func Load(path string) (*Config, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("reading config %s: %w", path, err)
	}
	var cfg Config
	if err := yaml.Unmarshal(data, &cfg); err != nil {
		return nil, fmt.Errorf("parsing config: %w", err)
	}
	if cfg.Root == "" {
		def := DefaultConfig()
		cfg.Root = def.Root
	}
	if cfg.RRFK == 0 {
		cfg.RRFK = 60
	}
	if cfg.CandidatePool == 0 {
		cfg.CandidatePool = 30
	}
	return &cfg, nil
}

// Save writes the config file to disk.
func (c *Config) Save() error {
	data, err := yaml.Marshal(c)
	if err != nil {
		return fmt.Errorf("marshaling config: %w", err)
	}
	if err := os.MkdirAll(c.Root, 0755); err != nil {
		return fmt.Errorf("creating config dir: %w", err)
	}
	return os.WriteFile(c.ConfigPath(), data, 0644)
}

// EnsureDirs creates all required directories.
func (c *Config) EnsureDirs() error {
	dirs := []string{
		c.Root,
		c.CardsDir(),
		c.VcsDir(),
		c.IndexDir(),
		c.ModelsDir(),
	}
	for _, d := range dirs {
		if err := os.MkdirAll(d, 0755); err != nil {
			return fmt.Errorf("creating directory %s: %w", d, err)
		}
	}
	return nil
}
