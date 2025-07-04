package llm

// ----------- Config -----------

type Config struct {
	VocabSize    int
	EmbeddingDim int
	NumHeads     int
	NumLayers    int
	SeqLen       int
}

type Transformer struct {
	Embeddings      [][]float64
	PositionEmbeds  [][]float64
	AttentionLayers []MultiHeadAttention
	FeedForwards    []FeedForward
	NormLayers      []LayerNorm
	Config          Config
}

// ----------- Model Init -----------

func NewTransformer(cfg Config) *Transformer {
	model := &Transformer{
		Embeddings:      make([][]float64, cfg.VocabSize),
		PositionEmbeds:  make([][]float64, cfg.SeqLen),
		AttentionLayers: make([]MultiHeadAttention, cfg.NumLayers),
		FeedForwards:    make([]FeedForward, cfg.NumLayers),
		NormLayers:      make([]LayerNorm, cfg.NumLayers),
		Config:          cfg,
	}
	// Token embeddings
	for i := 0; i < cfg.VocabSize; i++ {
		model.Embeddings[i] = randVector(cfg.EmbeddingDim)
	}
	// Positional encoding
	for i := 0; i < cfg.SeqLen; i++ {
		model.PositionEmbeds[i] = sinusoidalPosEncoding(i, cfg.EmbeddingDim)
	}
	// Layers
	for l := 0; l < cfg.NumLayers; l++ {
		model.AttentionLayers[l] = NewMultiHeadAttention(cfg.EmbeddingDim, cfg.NumHeads)
		model.FeedForwards[l] = NewFeedForward(cfg.EmbeddingDim)
		model.NormLayers[l] = NewLayerNorm(cfg.EmbeddingDim)
	}
	return model
}

// ----------- Forward Pass -----------

func (m *Transformer) Forward(input []int) []float64 {
	seqLen := len(input)
	if seqLen > m.Config.SeqLen {
		panic("Input too long")
	}

	// Step 1: Embed input
	x := make([][]float64, seqLen)
	for i, tokenID := range input {
		x[i] = vecAdd(m.Embeddings[tokenID], m.PositionEmbeds[i])
	}

	// Step 2: Transformer layers
	for l := 0; l < m.Config.NumLayers; l++ {
		attn := m.AttentionLayers[l].Apply(x)
		norm1 := m.NormLayers[l].Normalize(addVectors(x, attn))

		ff := m.FeedForwards[l].Apply(norm1)
		x = m.NormLayers[l].Normalize(addVectors(norm1, ff))
	}

	// Step 3: Final token logits (softmax-ready)
	return meanPool(x)
}
