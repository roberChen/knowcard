package card

import (
	"crypto/rand"
	"encoding/hex"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"time"
)

const MaxBodyTokens = 4096

var (
	ErrEmptyTitle   = errors.New("card title must not be empty")
	ErrEmptySummary = errors.New("card summary must not be empty")
	ErrEmptyBody    = errors.New("card body must not be empty")
	ErrBodyTooLong  = fmt.Errorf("card body exceeds %d tokens — shorten the body to a concise summary and attach the detailed content as a reference document (use the 'reference' parameter in upsert_card)", MaxBodyTokens)
	ErrEmptyID      = errors.New("card id must not be empty")
	ErrEmptyPath    = errors.New("card path must not be empty")
)

// Card represents a single knowledge card.
// Path is a semantic relative path (e.g. "programming/go/escape-analysis")
// that determines where the .md file lives on disk. ID is the immutable
// primary key stored in the YAML front matter.
type Card struct {
	ID        string    `yaml:"id"`
	Path      string    `yaml:"-"`          // not stored in front matter; derived from file location
	Title     string    `yaml:"title"`
	Keywords  []string  `yaml:"keywords"`
	Summary   string    `yaml:"summary"`
	Reference string    `yaml:"reference,omitempty"`
	Tags      []string  `yaml:"tags,omitempty"`
	Created   time.Time `yaml:"created"`
	Updated   time.Time `yaml:"updated"`
	Body      string    `yaml:"-"`
}

// NewID generates a random 16-byte hex ID (32 characters).
func NewID() string {
	b := make([]byte, 16)
	rand.Read(b)
	return hex.EncodeToString(b)
}

// Validate checks that the card has all required fields and sensible values.
// tokenCounter is an optional function for precise token counting; if nil,
// a character-based heuristic is used (1 token ≈ 3.5 chars for mixed CJK/English).
func (c *Card) Validate(tokenCounter func(string) (int, error)) error {
	if c.ID == "" {
		return ErrEmptyID
	}
	if c.Path == "" {
		return ErrEmptyPath
	}
	if c.Title == "" {
		return ErrEmptyTitle
	}
	if c.Summary == "" {
		return ErrEmptySummary
	}
	if c.Body == "" {
		return ErrEmptyBody
	}
	if tokenCounter != nil {
		n, err := tokenCounter(c.Body)
		if err != nil {
			return fmt.Errorf("token counting failed: %w", err)
		}
		if n > MaxBodyTokens {
			return ErrBodyTooLong
		}
	} else {
		heuristic := len(c.Body) / 3
		if heuristic > MaxBodyTokens {
			return ErrBodyTooLong
		}
	}
	return nil
}

// FilePath returns the full .md file path relative to the cards root.
func (c *Card) FilePath() string {
	return c.Path + ".md"
}

// PathFromID is used as a fallback when no semantic path is provided.
// It produces "unsorted/<first-2-chars>/<id>".
func PathFromID(id string) string {
	if len(id) < 2 {
		return filepath.Join("unsorted", id)
	}
	return filepath.Join("unsorted", id[:2], id)
}

// pathRegexp matches the allowed characters in a card path component.
// Allows letters, digits, hyphens, underscores, and dots. No spaces.
var pathRegexp = regexp.MustCompile(`^[a-zA-Z0-9][a-zA-Z0-9\-_./]*$`)

// ValidatePath checks that a path is a valid relative path with no
// leading/trailing slashes, no "..", and only URL-safe characters.
func ValidatePath(path string) error {
	path = strings.TrimSpace(path)
	if path == "" {
		return ErrEmptyPath
	}
	if strings.HasPrefix(path, "/") || strings.HasSuffix(path, "/") {
		return fmt.Errorf("path must not start or end with '/': %s", path)
	}
	if strings.Contains(path, "..") {
		return fmt.Errorf("path must not contain '..': %s", path)
	}
	if !pathRegexp.MatchString(path) {
		return fmt.Errorf("path contains invalid characters: %s", path)
	}
	return nil
}

// readCardFile reads a card from a .md file on disk, deriving Path from
// the file's location relative to cardsRoot.
func ReadCard(filePath, cardsRoot string) (*Card, error) {
	data, err := os.ReadFile(filePath)
	if err != nil {
		return nil, fmt.Errorf("reading card file %s: %w", filePath, err)
	}
	card, err := Parse(string(data))
	if err != nil {
		return nil, fmt.Errorf("parsing card file %s: %w", filePath, err)
	}
	rel, err := filepath.Rel(cardsRoot, filePath)
	if err != nil {
		return nil, fmt.Errorf("computing relative path: %w", err)
	}
	card.Path = strings.TrimSuffix(rel, ".md")
	// Normalise OS-specific separators to forward slashes
	card.Path = filepath.ToSlash(card.Path)
	return card, nil
}

// WriteCard writes a card to disk as a .md file inside cardsRoot.
// The file location is determined by card.FilePath().
func WriteCard(c *Card, cardsRoot string) error {
	if err := ValidatePath(c.Path); err != nil {
		return err
	}
	content, err := Serialize(c)
	if err != nil {
		return err
	}
	fullPath := filepath.Join(cardsRoot, c.FilePath())
	if err := os.MkdirAll(filepath.Dir(fullPath), 0755); err != nil {
		return fmt.Errorf("creating card directory: %w", err)
	}
	return os.WriteFile(fullPath, []byte(content), 0644)
}

// DeleteCardFile removes the .md file for the given path from disk.
func DeleteCardFile(path, cardsRoot string) error {
	fullPath := filepath.Join(cardsRoot, path+".md")
	return os.Remove(fullPath)
}

// MoveCardFile renames the .md file from oldPath to newPath within cardsRoot.
func MoveCardFile(oldPath, newPath, cardsRoot string) error {
	if err := ValidatePath(newPath); err != nil {
		return err
	}
	oldFull := filepath.Join(cardsRoot, oldPath+".md")
	newFull := filepath.Join(cardsRoot, newPath+".md")
	if err := os.MkdirAll(filepath.Dir(newFull), 0755); err != nil {
		return fmt.Errorf("creating target directory: %w", err)
	}
	return os.Rename(oldFull, newFull)
}
