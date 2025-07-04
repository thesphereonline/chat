package tokenizer

import (
	"strings"
)

type Tokenizer struct {
	Vocab     map[string]int
	InvVocab  map[int]string
	VocabSize int
}

func NewTokenizer() *Tokenizer {
	return &Tokenizer{
		Vocab:    make(map[string]int),
		InvVocab: make(map[int]string),
	}
}

func (t *Tokenizer) Train(corpus []string, vocabLimit int) {
	freq := map[string]int{}
	for _, line := range corpus {
		for _, word := range strings.Fields(line) {
			for _, ch := range word {
				freq[string(ch)]++
			}
		}
	}
	i := 0
	for k := range freq {
		t.Vocab[k] = i
		t.InvVocab[i] = k
		i++
		if i >= vocabLimit {
			break
		}
	}
	t.VocabSize = len(t.Vocab)
}

func (t *Tokenizer) Encode(text string) []int {
	tokens := []int{}
	for _, word := range strings.Fields(text) {
		for _, ch := range word {
			if id, ok := t.Vocab[string(ch)]; ok {
				tokens = append(tokens, id)
			} else {
				// fallback to unknown token (or skip)
				tokens = append(tokens, 0)
			}
		}
	}
	return tokens
}

func (t *Tokenizer) Decode(tokens []int) string {
	var sb strings.Builder
	for _, id := range tokens {
		sb.WriteString(t.InvVocab[id])
	}
	return sb.String()
}
