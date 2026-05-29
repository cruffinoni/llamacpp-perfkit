// Package tokenize counts tokens in text using tiktoken.
package tokenize

import (
	"fmt"
	"os"

	"github.com/pkoukk/tiktoken-go"
)

// Tokenizer counts tokens in text using a specific model's tokenizer.
type Tokenizer interface {
	CountTokens(text string) (int, error)
}

// GPTTokenizer wraps tiktoken for OpenAI model families.
type GPTTokenizer struct {
	enc *tiktoken.Tiktoken
}

// NewGPTTokenizer creates a GPTTokenizer for the given model name or encoding.
func NewGPTTokenizer(model string) (*GPTTokenizer, error) {
	enc, err := tiktoken.EncodingForModel(model)
	if err != nil {
		enc, err = tiktoken.GetEncoding(model)
		if err != nil {
			return nil, fmt.Errorf("get encoding for %q: %w", model, err)
		}
	}
	return &GPTTokenizer{enc: enc}, nil
}

// CountTokens returns the number of tokens in the given text.
func (g *GPTTokenizer) CountTokens(text string) (int, error) {
	tokens := g.enc.Encode(text, nil, nil)
	return len(tokens), nil
}

// CountTokensFromFile counts tokens in a file.
func CountTokensFromFile(path string, t Tokenizer) (int, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return 0, fmt.Errorf("read file: %w", err)
	}
	return t.CountTokens(string(data))
}
