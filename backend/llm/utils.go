package llm

import (
	"fmt"
	"math"
	"math/rand/v2"
)

func randVector(size int) []float64 {
	v := make([]float64, size)
	for i := range v {
		v[i] = rand.NormFloat64() * 0.02
	}
	return v
}

func randMatrix(rows, cols int) [][]float64 {
	m := make([][]float64, rows)
	for i := range m {
		m[i] = randVector(cols)
	}
	return m
}

func vecAdd(a, b []float64) []float64 {
	out := make([]float64, len(a))
	for i := range a {
		out[i] = a[i] + b[i]
	}
	return out
}

func addVectors(a, b [][]float64) [][]float64 {
	out := make([][]float64, len(a))
	for i := range a {
		out[i] = vecAdd(a[i], b[i])
	}
	return out
}

func matVecMul(mat [][]float64, vec []float64) []float64 {
	if len(mat[0]) != len(vec) {
		panic(fmt.Sprintf("matVecMul size mismatch: mat cols = %d, vec = %d", len(mat[0]), len(vec)))
	}

	out := make([]float64, len(mat))
	for i := range mat {
		sum := 0.0
		for j := range mat[i] {
			sum += mat[i][j] * vec[j]
		}
		out[i] = sum
	}
	return out
}

func matMul(x [][]float64, w [][]float64) [][]float64 {
	out := make([][]float64, len(x))
	for i := range x {
		out[i] = matVecMulTransposed(x[i], w)
	}
	return out
}

func matVecMulTransposed(vec []float64, mat [][]float64) []float64 {
	out := make([]float64, len(mat[0]))
	for i := range mat[0] {
		sum := 0.0
		for j := range vec {
			sum += vec[j] * mat[j][i]
		}
		out[i] = sum
	}
	return out
}

func dot(a, b []float64) float64 {
	sum := 0.0
	for i := range a {
		sum += a[i] * b[i]
	}
	return sum
}

func scaledDotProduct(Q, K [][]float64, d int) [][]float64 {
	out := make([][]float64, len(Q))
	scale := 1.0 / math.Sqrt(float64(d))
	for i := range Q {
		out[i] = make([]float64, len(K))
		for j := range K {
			out[i][j] = dot(Q[i], K[j]) * scale
		}
	}
	return out
}

func softmax2D(x [][]float64) [][]float64 {
	out := make([][]float64, len(x))
	for i := range x {
		out[i] = softmax(x[i])
	}
	return out
}

func softmax(x []float64) []float64 {
	maxVal := x[0]
	for _, v := range x {
		if v > maxVal {
			maxVal = v
		}
	}
	exps := make([]float64, len(x))
	sum := 0.0
	for i, v := range x {
		exps[i] = math.Exp(v - maxVal)
		sum += exps[i]
	}
	for i := range exps {
		exps[i] /= sum
	}
	return exps
}

func relu(x []float64) []float64 {
	out := make([]float64, len(x))
	for i, v := range x {
		if v > 0 {
			out[i] = v
		}
	}
	return out
}

func meanPool(x [][]float64) []float64 {
	out := make([]float64, len(x[0]))
	for _, vec := range x {
		for i := range vec {
			out[i] += vec[i]
		}
	}
	for i := range out {
		out[i] /= float64(len(x))
	}
	return out
}

func sinusoidalPosEncoding(pos, dim int) []float64 {
	out := make([]float64, dim)
	for i := 0; i < dim; i++ {
		angle := float64(pos) / math.Pow(10000, float64(2*(i/2))/float64(dim))
		if i%2 == 0 {
			out[i] = math.Sin(angle)
		} else {
			out[i] = math.Cos(angle)
		}
	}
	return out
}

func matMul2D(a, b [][]float64) [][]float64 {
	rows := len(a)
	cols := len(b[0])
	shared := len(b)

	out := make([][]float64, rows)
	for i := 0; i < rows; i++ {
		out[i] = make([]float64, cols)
		for j := 0; j < cols; j++ {
			sum := 0.0
			for k := 0; k < shared; k++ {
				sum += a[i][k] * b[k][j]
			}
			out[i][j] = sum
		}
	}
	return out
}
