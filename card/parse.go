package card

import (
	"bytes"
	"fmt"
	"strings"
	"time"

	"gopkg.in/yaml.v3"
)

const frontMatterDelim = "---"

// frontMatter is an internal struct for YAML (un)marshaling.
// Unlike Card, it includes Body and Path so the full file can be round-tripped.
type frontMatter struct {
	ID        string    `yaml:"id"`
	Title     string    `yaml:"title"`
	Keywords  []string  `yaml:"keywords"`
	Summary   string    `yaml:"summary"`
	Reference string    `yaml:"reference,omitempty"`
	Tags      []string  `yaml:"tags,omitempty"`
	Created   time.Time `yaml:"created"`
	Updated   time.Time `yaml:"updated"`
}

// Parse parses a markdown string with YAML front matter into a Card.
// Expected format:
//   ---
//   id: ...
//   title: ...
//   ---
//   # Body content...
func Parse(raw string) (*Card, error) {
	// Normalise leading whitespace/BOM
	raw = strings.TrimLeft(raw, "\xef\xbb\xbf")
	raw = strings.TrimLeft(raw, "\n\r\t ")

	lines := strings.Split(raw, "\n")

	// Find the opening delimiter
	if len(lines) == 0 || strings.TrimSpace(lines[0]) != frontMatterDelim {
		return nil, fmt.Errorf("missing opening front matter delimiter '---'")
	}

	// Find the closing delimiter
	closeIdx := -1
	for i := 1; i < len(lines); i++ {
		if strings.TrimSpace(lines[i]) == frontMatterDelim {
			closeIdx = i
			break
		}
	}
	if closeIdx == -1 {
		return nil, fmt.Errorf("missing closing front matter delimiter '---'")
	}

	yamlContent := strings.Join(lines[1:closeIdx], "\n")
	body := strings.TrimSpace(strings.Join(lines[closeIdx+1:], "\n"))

	var fm frontMatter
	if err := yaml.Unmarshal([]byte(yamlContent), &fm); err != nil {
		return nil, fmt.Errorf("parsing YAML front matter: %w", err)
	}

	return &Card{
		ID:        fm.ID,
		Title:     fm.Title,
		Keywords:  fm.Keywords,
		Summary:   fm.Summary,
		Reference: fm.Reference,
		Tags:      fm.Tags,
		Created:   fm.Created,
		Updated:   fm.Updated,
		Body:      body,
	}, nil
}

// Serialize converts a Card into a markdown string with YAML front matter.
func Serialize(c *Card) (string, error) {
	fm := frontMatter{
		ID:        c.ID,
		Title:     c.Title,
		Keywords:  c.Keywords,
		Summary:   c.Summary,
		Reference: c.Reference,
		Tags:      c.Tags,
		Created:   c.Created,
		Updated:   c.Updated,
	}

	var buf bytes.Buffer

	// Write opening delimiter
	buf.WriteString(frontMatterDelim + "\n")

	// Marshal front matter to YAML
	yamlData, err := yaml.Marshal(&fm)
	if err != nil {
		return "", fmt.Errorf("marshaling YAML front matter: %w", err)
	}

	// Remove trailing newline from yaml.Marshal to avoid double newlines
	yamlStr := strings.TrimRight(string(yamlData), "\n")
	buf.WriteString(yamlStr)
	buf.WriteString("\n")

	// Write closing delimiter
	buf.WriteString(frontMatterDelim + "\n\n")

	// Write body
	buf.WriteString(strings.TrimSpace(c.Body))
	buf.WriteString("\n")

	return buf.String(), nil
}
