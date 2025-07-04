package main

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/thesphereonline/chat/backend/llm"
	"github.com/thesphereonline/chat/backend/tokenizer"
)

func main() {
	rand.Seed(time.Now().UnixNano())

	// 1. Initialize tokenizer and train vocab
	tok := tokenizer.NewTokenizer()
	corpus := []string{
		"hello world",
		"custom llm in go",
		"sphere chat ai",
	}
	tok.Train(corpus, 50)

	// 2. Init model config
	cfg := llm.Config{
		VocabSize:    len(tok.Vocab),
		EmbeddingDim: 32,
		NumHeads:     2,
		NumLayers:    2,
		SeqLen:       16,
	}

	model := llm.NewTransformer(cfg)

	// 3. Training data: input → target token
	type Sample struct {
		Input  []int
		Target int
	}
	trainingData := []Sample{}
	for _, text := range corpus {
		tokens := tok.Encode(text)
		for i := 0; i < len(tokens)-1; i++ {
			trainingData = append(trainingData, Sample{
				Input:  tokens[:i+1],
				Target: tokens[i+1],
			})
		}
	}

	fmt.Println("📚 Training on", len(trainingData), "samples")

	// 4. Train for N epochs
	epochs := 50
	learningRate := 0.01

	for epoch := 1; epoch <= epochs; epoch++ {
		totalLoss := 0.0
		for _, sample := range trainingData {
			// Forward pass
			output := model.Forward(sample.Input)

			// Predict next token as the one with max dot product
			logits := llm.DotWithEmbeddings(output, model.Embeddings)

			// Calculate loss (mean squared error)
			loss := llm.MSELoss(logits, sample.Target)
			totalLoss += loss

			// Very naive backprop: subtract scaled gradient
			for i := range logits {
				grad := logits[i]
				if i == sample.Target {
					grad -= 1
				}
				for j := 0; j < model.Config.EmbeddingDim; j++ {
					model.Embeddings[i][j] -= learningRate * grad * output[j]
				}
			}

		}

		fmt.Printf("🧠 Epoch %d | Loss: %.4f\n", epoch, totalLoss/float64(len(trainingData)))
	}
}
