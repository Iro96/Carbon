#pragma once
#include "embedding.hpp"
#include "transformer_block.hpp"
#include "layers.hpp"
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>

Tensor softmax(const Tensor& x) {
    Tensor y(x.rows, x.cols);
    for (int i = 0; i < x.rows; i++) {
        float maxv = -1e9, sum = 0;
        for (int j = 0; j < x.cols; j++) maxv = std::max(maxv, x(i, j));
        for (int j = 0; j < x.cols; j++) { y(i, j) = exp(x(i, j) - maxv); sum += y(i, j); }
        for (int j = 0; j < x.cols; j++) y(i, j) /= sum;
    }
    return y;
}

float cross_entropy(const Tensor& pred, const std::vector<int>& target, Tensor& grad_out) {
    float loss = 0;
    for (int i = 0; i < pred.rows; i++) {
        int t = target[i];
        for (int j = 0; j < pred.cols; j++) {
            float p = pred(i, j);
            grad_out.grad[i * pred.cols + j] = p - (j == t ? 1.0f : 0.0f);
        }
        loss += -log(std::max(pred(i, t), 1e-9f));
    }
    return loss / pred.rows;
}

struct Model {
    Embedding emb;
    std::vector<TransformerBlock> blocks;
    Linear lm_head;

    Model(int vocab, int dim, int hidden, int layers, int heads)
        : emb(vocab, dim), lm_head(dim, vocab) {
        for (int i = 0; i < layers; i++) blocks.emplace_back(dim, hidden, heads);
    }

    Tensor forward(const std::vector<int>& tokens) {
        Tensor x = emb.forward(tokens);
        for (auto& b : blocks) x = b.forward(x);
        Tensor logits = lm_head.forward(x);
        return softmax(logits);
    }

    void step(float lr) {
        emb.step(lr);
        for (auto& b : blocks) b.step(lr);
        lm_head.step(lr);
    }

    void save(const std::string& path) {
        std::ofstream f(path, std::ios::binary);
        emb.save(f);
        for (auto& b : blocks) b.save(f);
        lm_head.save(f);
        f.close();
    }

    void load(const std::string& path) {
        std::ifstream f(path, std::ios::binary);
        emb.load(f);
        for (auto& b : blocks) b.load(f);
        lm_head.load(f);
        f.close();
    }

    // Optional: inference helper
    int predict_next(const std::vector<int>& tokens) {
        Tensor probs = forward(tokens);
        int last_row = probs.rows - 1;
        float best = -1e9;
        int idx = 0;
        for (int j = 0; j < probs.cols; j++) {
            if (probs(last_row, j) > best) {
                best = probs(last_row, j);
                idx = j;
            }
        }
        return idx;
    }
};
