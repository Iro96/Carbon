#pragma once
#include "tensor.hpp"

struct Linear {
    Tensor W, b;
    Tensor x_cache;

    Linear(int in, int out)
        : W(in, out), b(1, out) {
        W.randomize();
        b.randomize();
    }

    Tensor forward(const Tensor& x) {
        x_cache = x;  // cache input for backward
        Tensor y = Tensor::matmul(x, W);

        // add bias
        for (int i = 0; i < y.rows; ++i)
            for (int j = 0; j < y.cols; ++j)
                y(i, j) += b.val[j];

        return y;
    }

    Tensor backward(const Tensor& grad_out) {
        // grad_out.val holds upstream gradient (dL/dY)

        Tensor grad_in(x_cache.rows, W.rows);

        // dL/dW and dL/dX
        for (int i = 0; i < x_cache.rows; ++i) {
            for (int j = 0; j < W.cols; ++j) {
                float go = grad_out.val[i * grad_out.cols + j];
                for (int k = 0; k < W.rows; ++k) {
                    W.grad[k * W.cols + j] += x_cache(i, k) * go;
                    grad_in.val[i * W.rows + k] += go * W.val[k * W.cols + j];
                }
            }
        }

        // dL/db
        for (int j = 0; j < b.cols; ++j) {
            float sum = 0.0f;
            for (int i = 0; i < grad_out.rows; ++i)
                sum += grad_out.val[i * grad_out.cols + j];
            b.grad[j] += sum;
        }

        return grad_in;
    }

    void step(float lr) {
        for (size_t i = 0; i < W.val.size(); ++i) {
            W.val[i] -= lr * W.grad[i];
            W.grad[i] = 0.0f;
        }
        for (size_t i = 0; i < b.val.size(); ++i) {
            b.val[i] -= lr * b.grad[i];
            b.grad[i] = 0.0f;
        }
    }

    void save(std::ofstream& f) const {
        W.save(f);
        b.save(f);
    }

    void load(std::ifstream& f) {
        W.load(f);
        b.load(f);
    }
};
