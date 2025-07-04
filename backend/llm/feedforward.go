package llm

type FeedForward struct {
	W1 [][]float64
	W2 [][]float64
}

func NewFeedForward(dim int) FeedForward {
	hiddenDim := dim * 4
	return FeedForward{
		W1: randMatrix(hiddenDim, dim),
		W2: randMatrix(hiddenDim, dim),
	}
}

func (f *FeedForward) Apply(x [][]float64) [][]float64 {
	seqLen := len(x)
	out := make([][]float64, seqLen)
	for i := 0; i < seqLen; i++ {
		hidden := relu(matVecMulTransposed(x[i], f.W1))
		out[i] = matVecMul(f.W2, hidden)
	}
	return out
}
