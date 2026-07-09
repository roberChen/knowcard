package card

import (
	"strings"
	"testing"
	"time"
)

func mustParseTime(t *testing.T, s string) time.Time {
	t.Helper()
	ts, err := time.Parse(time.RFC3339, s)
	if err != nil {
		t.Fatalf("parsing time %q: %v", s, err)
	}
	return ts
}

func TestParse(t *testing.T) {
	input := `---
id: abc123
title: Test Card
keywords:
    - go
    - memory
summary: A test card about Go memory.
reference: /docs/go/test.md
tags:
    - programming
    - go
created: 2026-07-10T10:00:00Z
updated: 2026-07-10T10:00:00Z
---

# Test Card

This is the body content.
It has multiple lines.
`
	card, err := Parse(input)
	if err != nil {
		t.Fatalf("Parse failed: %v", err)
	}
	if card.ID != "abc123" {
		t.Errorf("ID = %q, want %q", card.ID, "abc123")
	}
	if card.Title != "Test Card" {
		t.Errorf("Title = %q, want %q", card.Title, "Test Card")
	}
	if len(card.Keywords) != 2 || card.Keywords[0] != "go" || card.Keywords[1] != "memory" {
		t.Errorf("Keywords = %v, want [go memory]", card.Keywords)
	}
	if card.Summary != "A test card about Go memory." {
		t.Errorf("Summary = %q", card.Summary)
	}
	if card.Reference != "/docs/go/test.md" {
		t.Errorf("Reference = %q", card.Reference)
	}
	if len(card.Tags) != 2 {
		t.Errorf("Tags = %v", card.Tags)
	}
	expectedBody := "# Test Card\n\nThis is the body content.\nIt has multiple lines."
	if card.Body != expectedBody {
		t.Errorf("Body = %q, want %q", card.Body, expectedBody)
	}
}

func TestParse_NoFrontMatter(t *testing.T) {
	_, err := Parse("just some text")
	if err == nil {
		t.Fatal("expected error for missing front matter")
	}
}

func TestParse_MissingCloseDelimiter(t *testing.T) {
	input := "---\nid: abc\ntitle: Test\n"
	_, err := Parse(input)
	if err == nil {
		t.Fatal("expected error for missing close delimiter")
	}
}

func TestSerialize_RoundTrip(t *testing.T) {
	original := &Card{
		ID:        "deadbeef",
		Title:     "Round Trip",
		Keywords:  []string{"test", "serialize"},
		Summary:   "Testing serialize and parse round trip.",
		Reference: "/docs/test.md",
		Tags:      []string{"unit"},
		Created:   mustParseTime(t, "2026-07-10T10:00:00Z"),
		Updated:   mustParseTime(t, "2026-07-10T12:00:00Z"),
		Body:      "# Round Trip\n\nSome content here.",
	}

	data, err := Serialize(original)
	if err != nil {
		t.Fatalf("Serialize failed: %v", err)
	}

	parsed, err := Parse(data)
	if err != nil {
		t.Fatalf("Parse of serialized data failed: %v", err)
	}

	if parsed.ID != original.ID {
		t.Errorf("ID mismatch: %q vs %q", parsed.ID, original.ID)
	}
	if parsed.Title != original.Title {
		t.Errorf("Title mismatch: %q vs %q", parsed.Title, original.Title)
	}
	if len(parsed.Keywords) != len(original.Keywords) {
		t.Errorf("Keywords length mismatch")
	}
	if parsed.Summary != original.Summary {
		t.Errorf("Summary mismatch")
	}
	if parsed.Body != original.Body {
		t.Errorf("Body mismatch: %q vs %q", parsed.Body, original.Body)
	}
	if !parsed.Created.Equal(original.Created) {
		t.Errorf("Created mismatch: %v vs %v", parsed.Created, original.Created)
	}
	if !parsed.Updated.Equal(original.Updated) {
		t.Errorf("Updated mismatch: %v vs %v", parsed.Updated, original.Updated)
	}
}

func TestSerialize_NoReference(t *testing.T) {
	card := &Card{
		ID:       "x",
		Title:    "No Ref",
		Summary:  "No reference field.",
		Body:     "body",
		Created:  time.Now(),
		Updated:  time.Now(),
	}
	data, err := Serialize(card)
	if err != nil {
		t.Fatalf("Serialize failed: %v", err)
	}
	// reference should not appear in output when empty
	if strings.Contains(data, "reference:") {
		t.Errorf("serialized output should not contain 'reference:' when empty:\n%s", data)
	}
}

func TestValidate(t *testing.T) {
	tests := []struct {
		name    string
		card    *Card
		wantErr error
	}{
		{
			name: "valid card",
			card: &Card{
				ID: "abc", Path: "test/card", Title: "T", Summary: "S", Body: "B",
				Created: time.Now(), Updated: time.Now(),
			},
			wantErr: nil,
		},
		{
			name: "empty ID",
			card: &Card{Path: "test/card", Title: "T", Summary: "S", Body: "B"},
			wantErr: ErrEmptyID,
		},
		{
			name: "empty path",
			card: &Card{ID: "abc", Title: "T", Summary: "S", Body: "B"},
			wantErr: ErrEmptyPath,
		},
		{
			name: "empty title",
			card: &Card{ID: "abc", Path: "test/card", Summary: "S", Body: "B"},
			wantErr: ErrEmptyTitle,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.card.Validate(nil)
			if tt.wantErr == nil && err != nil {
				t.Errorf("unexpected error: %v", err)
			}
			if tt.wantErr != nil && err != tt.wantErr {
				t.Errorf("error = %v, want %v", err, tt.wantErr)
			}
		})
	}
}

func TestValidatePath(t *testing.T) {
	valid := []string{"programming/go/test", "a/b/c", "concept-1", "databases/redis_001"}
	for _, p := range valid {
		if err := ValidatePath(p); err != nil {
			t.Errorf("ValidatePath(%q) = %v, want nil", p, err)
		}
	}
	invalid := []string{"", "/leading", "trailing/", "../escape", "has space", "中文路径"}
	for _, p := range invalid {
		if err := ValidatePath(p); err == nil {
			t.Errorf("ValidatePath(%q) = nil, want error", p)
		}
	}
}

func TestNewID(t *testing.T) {
	id1 := NewID()
	id2 := NewID()
	if len(id1) != 32 {
		t.Errorf("NewID() length = %d, want 32", len(id1))
	}
	if id1 == id2 {
		t.Error("NewID() returned duplicate IDs")
	}
}
