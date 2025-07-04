package llm

import "math"

type LayerNorm struct {
	Eps float64
}

func NewLayerNorm(dim int) LayerNorm {
	return LayerNorm{Eps: 1e-5}
}

func (ln *LayerNorm) Normalize(x [][]float64) [][]float64 {
	normed := make([][]float64, len(x))
	for i := range x {
		normed[i] = layerNormVec(x[i], ln.Eps)
	}
	return normed
}

func layerNormVec(x []float64, eps float64) []float64 {
	mean := 0.0
	for _, v := range x {
		mean += v
	}
	mean /= float64(len(x))

	variance := 0.0
	for _, v := range x {
		diff := v - mean
		variance += diff * diff
	}
	variance /= float64(len(x))

	normed := make([]float64, len(x))
	for i, v := range x {
		normed[i] = (v - mean) / math.Sqrt(variance+eps)
	}
	return normed
}
