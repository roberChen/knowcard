package search

import (
	"strings"
	"unicode"
	"unicode/utf8"
)

// Tokenizer splits text into search tokens.
// For English/Latin text: lowercase, split on non-alphanumeric.
// For CJK text: extracts character bigrams (plus unigrams) which is a
// proven technique for Chinese/Japanese/Korean IR without word segmentation.
type Tokenizer struct{}

func NewTokenizer() *Tokenizer { return &Tokenizer{} }

// Tokenize returns a slice of lowercase tokens extracted from text.
// Each CJK bigram is a token, each CJK unigram is a token,
// and each maximal Latin alphanumeric sequence is a token.
func (t *Tokenizer) Tokenize(text string) []string {
	text = strings.ToLower(text)
	var tokens []string
	var latin strings.Builder

 runes := []rune(text)
	for i := 0; i < len(runes); i++ {
		r := runes[i]

		if isCJK(r) {
			// Flush any pending Latin word
			if latin.Len() > 0 {
				tokens = append(tokens, latin.String())
				latin.Reset()
			}
			// CJK unigram
			tokens = append(tokens, string(r))
			// CJK bigram with next char if also CJK
			if i+1 < len(runes) && isCJK(runes[i+1]) {
				tokens = append(tokens, string(r)+string(runes[i+1]))
			}
			continue
		}

		if (r >= 'a' && r <= 'z') || (r >= '0' && r <= '9') || r == '-' || r == '_' {
			latin.WriteRune(r)
		} else {
			if latin.Len() > 0 {
				tokens = append(tokens, latin.String())
				latin.Reset()
			}
			// Ignore other characters (punctuation, spaces, etc.)
		}
	}
	if latin.Len() > 0 {
		tokens = append(tokens, latin.String())
	}
	return tokens
}

// isCJK reports whether r is a CJK Unified Ideograph or related range.
func isCJK(r rune) bool {
	return (r >= 0x4E00 && r <= 0x9FFF) ||  // CJK Unified Ideographs
		(r >= 0x3400 && r <= 0x4DBF) ||  // CJK Extension A
		(r >= 0x20000 && r <= 0x2A6DF) || // CJK Extension B
		(r >= 0x3040 && r <= 0x309F) ||  // Hiragana
		(r >= 0x30A0 && r <= 0x30FF) ||  // Katakana
		(r >= 0xAC00 && r <= 0xD7AF)     // Hangul Syllables
}

// IsValidUTF8 is a convenience wrapper exported for testing.
func IsValidUTF8(s string) bool { return utf8.ValidString(s) }

// IsASCIILetter checks if a rune is a basic ASCII letter (for testing).
func IsASCIILetter(r rune) bool {
	return unicode.IsLetter(r) && r < 128
}
