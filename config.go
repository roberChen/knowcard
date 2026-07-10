package knowcard

import (
	"fmt"
	"os"
	"path/filepath"

	"gopkg.in/yaml.v3"
)

// DirName is the per-project knowledge base directory name.
const DirName = ".knowcard"

// EmbedBackend selects how embeddings are computed.
type EmbedBackend string

const (
	EmbedLocal       EmbedBackend = "local"       // yzma + GGUF, CPU inference
	EmbedOllama      EmbedBackend = "ollama"      // Ollama API
	EmbedOpenAI      EmbedBackend = "openai"      // OpenAI API
	EmbedCustom      EmbedBackend = "custom"      // OpenAI-compatible API
	EmbedQwenCloud   EmbedBackend = "qwen_cloud"  // DashScope native API (text + VL multimodal)
	EmbedSiliconFlow EmbedBackend = "siliconflow" // SiliconFlow OpenAI-compatible API (text + VL)
)

// Config defines all runtime configuration for knowcard.
// Config is global (lives in ~/.config/knowcard/config.yaml), while the
// knowledge base (cards, index, vcs) is per-project (.knowcard/).
type Config struct {
	// Root is the per-project knowledge base directory, discovered at runtime
	// by walking upward from CWD. NOT stored in the config file.
	Root string `yaml:"-"`

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
	ModelPath  string `yaml:"model_path,omitempty"` // path to .gguf file
	LibPath    string `yaml:"lib_path,omitempty"`   // path to libllama shared library
	ContextSize uint32 `yaml:"context_size,omitempty"`
	BatchSize   uint32 `yaml:"batch_size,omitempty"`
	Pooling    string `yaml:"pooling,omitempty"` // "mean", "cls", "last" (default "last" for Qwen models)

	// API settings (Ollama / OpenAI / custom / DashScope / SiliconFlow)
	APIBase string `yaml:"api_base,omitempty"`
	APIKey  string `yaml:"api_key,omitempty"`
	Model   string `yaml:"model,omitempty"`

	// Dimensions for MRL models (Qwen3-Embedding, Qwen3-VL-Embedding, etc.)
	// 0 means use model default
	Dimensions int `yaml:"dimensions,omitempty"`

	// DashScope-specific
	DashScopeInternational bool   `yaml:"dashscope_international,omitempty"` // true = intl endpoint, false = domestic
	Instruct              string `yaml:"instruct,omitempty"`                // task instruction for retrieval
}

// DefaultConfig returns a configuration with sensible defaults.
func DefaultConfig() Config {
	return Config{
		Embed: EmbedConfig{
			Backend:      EmbedLocal,
			ContextSize:  2048,
			BatchSize:    512,
		},
		RRFK:          60,
		CandidatePool: 30,
	}
}

// GlobalConfigPath returns the path to the global config file.
// Search order: $XDG_CONFIG_HOME/knowcard/config.yaml, ~/.config/knowcard/config.yaml
func GlobalConfigPath() string {
	xdg := os.Getenv("XDG_CONFIG_HOME")
	if xdg == "" {
		home, _ := os.UserHomeDir()
		xdg = filepath.Join(home, ".config")
	}
	return filepath.Join(xdg, "knowcard", "config.yaml")
}

// FindRoot walks upward from startDir looking for a .knowcard directory.
// Returns the absolute path to the .knowcard directory, or error if not found.
func FindRoot(startDir string) (string, error) {
	abs, err := filepath.Abs(startDir)
	if err != nil {
		return "", fmt.Errorf("resolving path: %w", err)
	}

	for {
		candidate := filepath.Join(abs, DirName)
		if info, err := os.Stat(candidate); err == nil && info.IsDir() {
			return candidate, nil
		}
		parent := filepath.Dir(abs)
		if parent == abs {
			break
		}
		abs = parent
	}
	return "", fmt.Errorf("no %s directory found (searched from %s upward)", DirName, startDir)
}

// LoadGlobal loads the global config from GlobalConfigPath(), with env var expansion.
func LoadGlobal() (*Config, error) {
	return Load(GlobalConfigPath())
}

// Load reads a config file from the given path, with env var expansion.
func Load(path string) (*Config, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("reading config %s: %w", path, err)
	}
	expanded := os.ExpandEnv(string(data))
	var cfg Config
	if err := yaml.Unmarshal([]byte(expanded), &cfg); err != nil {
		return nil, fmt.Errorf("parsing config: %w", err)
	}
	if cfg.RRFK == 0 {
		cfg.RRFK = 60
	}
	if cfg.CandidatePool == 0 {
		cfg.CandidatePool = 30
	}
	return &cfg, nil
}

// Save writes the config to the given path.
func Save(path string, cfg *Config) error {
	data, err := yaml.Marshal(cfg)
	if err != nil {
		return fmt.Errorf("marshaling config: %w", err)
	}
	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		return fmt.Errorf("creating config dir: %w", err)
	}
	return os.WriteFile(path, data, 0644)
}

// SaveGlobal writes the config to GlobalConfigPath().
func SaveGlobal(cfg *Config) error {
	return Save(GlobalConfigPath(), cfg)
}

// --- Per-project knowledge base paths (derived from Config.Root) ---

func (c *Config) CardsDir() string  { return filepath.Join(c.Root, "cards") }
func (c *Config) VcsDir() string    { return filepath.Join(c.Root, "_vcs") }
func (c *Config) IndexDir() string  { return filepath.Join(c.Root, "index") }
func (c *Config) ChromemDir() string { return filepath.Join(c.IndexDir(), "chromem") }
func (c *Config) ManifestPath() string { return filepath.Join(c.Root, "manifest.json") }

// EnsureDirs creates all required per-project directories.
func (c *Config) EnsureDirs() error {
	dirs := []string{
		c.Root,
		c.CardsDir(),
		c.VcsDir(),
		c.IndexDir(),
	}
	for _, d := range dirs {
		if err := os.MkdirAll(d, 0755); err != nil {
			return fmt.Errorf("creating directory %s: %w", d, err)
		}
	}
	return nil
}
