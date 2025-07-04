package main

import (
	"fmt"

	"github.com/thesphereonline/chat/backend/llm"
	"github.com/thesphereonline/chat/backend/tokenizer"
)

func main() {
	// Init tokenizer and train minimal vocab
	tok := tokenizer.NewTokenizer()
	tok.Train([]string{
		"hello world",
		"this is sphere chat",
		"custom LLM in Go",
	}, 100)

	// Example input
	input := "hello sphere"
	tokens := tok.Encode(input)

	fmt.Println("🧠 Input Tokens:", tokens)

	// Init transformer model
	model := llm.NewTransformer(llm.Config{
		VocabSize:    len(tok.Vocab),
		EmbeddingDim: 32,
		NumHeads:     2,
		NumLayers:    2,
		SeqLen:       16,
	})

	fmt.Println("Token count:", len(tokens))
	fmt.Println("Embedding dim:", model.Config.EmbeddingDim)
	fmt.Println("Vocab size:", model.Config.VocabSize)
	// Run forward pass
	output := model.Forward(tokens)

	fmt.Println("🧾 Output Vector:", output)
}
