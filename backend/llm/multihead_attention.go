package llm

type MultiHeadAttention struct {
	NumHeads int
	Dim      int
	HeadDim  int

	Wq [][][]float64
	Wk [][][]float64
	Wv [][][]float64
	Wo [][]float64
}

func NewMultiHeadAttention(dim, numHeads int) MultiHeadAttention {
	headDim := dim / numHeads
	mha := MultiHeadAttention{
		NumHeads: numHeads,
		Dim:      dim,
		HeadDim:  headDim,
		Wq:       make([][][]float64, numHeads),
		Wk:       make([][][]float64, numHeads),
		Wv:       make([][][]float64, numHeads),
		Wo:       make([][]float64, dim),
	}
	for h := 0; h < numHeads; h++ {
		mha.Wq[h] = randMatrix(dim, headDim)
		mha.Wk[h] = randMatrix(dim, headDim)
		mha.Wv[h] = randMatrix(dim, headDim)
	}
	for i := 0; i < dim; i++ {
		mha.Wo[i] = randVector(dim)
	}
	return mha
}

func (m *MultiHeadAttention) Apply(x [][]float64) [][]float64 {
	seqLen := len(x)
	output := make([][]float64, seqLen)

	// Concatenate all head outputs
	allHeads := make([][][]float64, m.NumHeads)
	for h := 0; h < m.NumHeads; h++ {
		Q := matMul(x, m.Wq[h])
		K := matMul(x, m.Wk[h])
		V := matMul(x, m.Wv[h])

		// Attention = softmax(QKᵀ / √d) * V
		attnScores := scaledDotProduct(Q, K, m.HeadDim)
		attnProbs := softmax2D(attnScores)
		attnOut := matMul2D(attnProbs, V)
		allHeads[h] = attnOut
	}

	// Concatenate all heads
	for t := 0; t < seqLen; t++ {
		concat := []float64{}
		for h := 0; h < m.NumHeads; h++ {
			concat = append(concat, allHeads[h][t]...)
		}
		output[t] = matVecMul(m.Wo, concat)
	}
	return output
}
