package llm

import (
	"math"
	"math/rand"
)

// dotWithEmbeddings computes dot product between output vector and every token embedding.
// This acts like a final classification layer (logits).
func DotWithEmbeddings(vec []float64, embeds [][]float64) []float64 {
	scores := make([]float64, len(embeds))
	for i := range embeds {
		scores[i] = dot(vec, embeds[i])
	}
	return softmax(scores)
}

// mseLoss calculates mean squared error between softmax(pred) and one-hot(target)
func MSELoss(pred []float64, target int) float64 {
	loss := 0.0
	for i := range pred {
		diff := 0.0
		if i == target {
			diff = pred[i] - 1.0
		} else {
			diff = pred[i]
		}
		loss += diff * diff
	}
	return loss / float64(len(pred))
}

// predictTop1 returns the index of the highest-scoring token
func PredictTop1(logits []float64) int {
	maxIdx := 0
	maxVal := logits[0]
	for i, v := range logits {
		if v > maxVal {
			maxVal = v
			maxIdx = i
		}
	}
	return maxIdx
}

// sampleTopK picks a token from the top-k logits (with temperature-based randomness)
func SampleTopK(logits []float64, k int, temperature float64) int {
	type pair struct {
		Index int
		Score float64
	}
	top := make([]pair, len(logits))
	for i, score := range logits {
		top[i] = pair{i, score}
	}
	// sort descending
	for i := 0; i < len(top); i++ {
		for j := i + 1; j < len(top); j++ {
			if top[j].Score > top[i].Score {
				top[i], top[j] = top[j], top[i]
			}
		}
	}
	if k > len(top) {
		k = len(top)
	}
	probs := make([]float64, k)
	sum := 0.0
	for i := 0; i < k; i++ {
		probs[i] = math.Exp(top[i].Score / temperature)
		sum += probs[i]
	}
	for i := range probs {
		probs[i] /= sum
	}
	r := rand.Float64()
	acc := 0.0
	for i := 0; i < k; i++ {
		acc += probs[i]
		if r <= acc {
			return top[i].Index
		}
	}
	return top[0].Index
}
