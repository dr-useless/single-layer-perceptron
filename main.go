package main

import (
	"math"
	"math/rand"
	"time"
)

func main() {
	goPerceptron := Perceptron{
		input:        [][]float64{{0, 0, 1}, {1, 1, 1}, {1, 0, 1}, {0, 1, 0}}, // Input data
		actualOutput: []float64{0, 1, 1, 0},
		epochs:       10000,
	}
	goPerceptron.initRandom()
	goPerceptron.train()
	//Make Predictions
	print(goPerceptron.propogateForward([]float64{0, 1, 0}), "\n")
	print(goPerceptron.propogateForward([]float64{1, 0, 1}), "\n")
}

// Perceptron (neuron) object
type Perceptron struct {
	input        [][]float64
	actualOutput []float64
	weights      []float64
	bias         float64
	epochs       int
}

// Random initialization
func (a *Perceptron) initRandom() {
	rand.Seed(time.Now().UnixNano())
	a.bias = 0.0
	a.weights = make([]float64, len(a.input[0]))
	for i := 0; i < len(a.input[0]); i++ {
		a.weights[i] = rand.Float64()
	}
}

// Dot product of 2 vectors of same size
func dotProduct(v1, v2 []float64) float64 {
	dot := 0.0
	for i := 0; i < len(v1); i++ {
		dot += v1[i] * v2[i]
	}
	return dot
}

// Addition of 2 vectors of same size
func vecAdd(v1, v2 []float64) []float64 {
	add := make([]float64, len(v1))
	for i := 0; i < len(v1); i++ {
		add[i] = v1[i] + v2[i]
	}
	return add
}

// Multiplication of a vector & matrix
func scalarMatMul(s float64, mat []float64) []float64 {
	result := make([]float64, len(mat))
	for i := 0; i < len(mat); i++ {
		result[i] += s * mat[i]
	}
	return result
}

// Sigmoid activation function
func (a *Perceptron) sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

// Forward pass
func (a *Perceptron) propogateForward(x []float64) (sum float64) {
	return a.sigmoid(dotProduct(a.weights, x) + a.bias)
}

// Calculate gradients of weights
func (a *Perceptron) gradW(x []float64, y float64) []float64 {
	pred := a.propogateForward(x)
	return scalarMatMul(-(pred-y)*pred*(1-pred), x)
}

// Calculate gradients of bias
func (a *Perceptron) gradB(x []float64, y float64) float64 {
	pred := a.propogateForward(x)
	return -(pred - y) * pred * (1 - pred)
}

//Train the perceptron for n epochs
func (a *Perceptron) train() {
	for i := 0; i < a.epochs; i++ {
		dw := make([]float64, len(a.input[0]))
		db := 0.0
		for length, val := range a.input {
			dw = vecAdd(dw, a.gradW(val, a.actualOutput[length]))
			db += a.gradB(val, a.actualOutput[length])
		}
		dw = scalarMatMul(2/float64(len(a.actualOutput)), dw)
		a.weights = vecAdd(a.weights, dw)
		a.bias += db * 2 / float64(len(a.actualOutput))
	}
}
