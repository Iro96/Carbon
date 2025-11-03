# Carbon [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

**Carbon** is a pure C++ Transformer framework inspired by GPT, featuring SIMD-optimized tensor math, multi-head attention, feedforward networks, and Byte Pair Encoding (BPE) tokenization. It’s a fully self-contained system for training and running language models — no external ML libraries required.

---

## Features

* **Pure C++ Implementation** — No dependencies on TensorFlow, PyTorch, or Eigen.
* **SIMD-Optimized Math** — Fast tensor operations using AVX instructions.
* **Transformer Architecture** — Implements GPT-style blocks with LayerNorm, MHA, and FeedForward layers.
* **Custom BPE Tokenizer** — Train and apply subword tokenization directly in C++.
* **Train & Save Models** — Full forward, backward, and training loops included.
* **Lightweight & Educational** — Easy to read and modify for research or learning purposes.

---

## Model Overview

| Component                            | Description                                             |
| ------------------------------------ | ------------------------------------------------------- |
| **Tensor**                           | Core data structure for matrix operations and gradients |
| **Linear / LayerNorm / FeedForward** | Standard Transformer components                         |
| **MultiHeadAttention**               | Implements scaled dot-product attention                 |
| **TransformerBlock**                 | Combines attention, normalization, and MLP layers       |
| **Embedding + LM Head**              | Token and output embeddings                             |
| **BPETokenizer**                     | Pure C++ byte-pair encoding tokenizer                   |

---

## Example Configuration

Default model in `main.cpp`:

```cpp
const int vocab   = 50000;
const int dim     = 1024;
const int hidden  = 4096;
const int layers  = 24;
const int heads   = 16;
Model model(vocab, dim, hidden, layers, heads);
```

You can adjust these parameters freely — **no code refactor required**.

---

## Build Instructions

### Requirements

* C++17 or later
* AVX2-capable CPU
* `g++` or `clang++` compiler

### Compile

```bash
g++ -O3 -mavx2 -std=c++17 main.cpp -o carbon
```

### Run

```bash
./carbon
```

This will train a small Transformer model on randomly generated token data and periodically print training loss.

---

## Saving & Loading

The model automatically saves its trained weights to:

```
MiniLLM_250M.cb
```

and can later be reloaded using:

```cpp
model.load("MiniLLM_250M.cb");
```

---

## Tokenizer

Train and apply BPE tokenization directly:

```cpp
BPETokenizer tok;
tok.train("dataset.txt", 50000);
auto tokens = tok.encode("Hello world");
auto text = tok.decode(tokens);
```

---

## License

MIT License © 2025 — Open Source and free for research and educational use.
