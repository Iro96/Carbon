#include "model.hpp"
#include "tokenizer_bpe.hpp"
#include <iostream>
#include <random>

int main() {
    // --- Initialize tokenizer and train or load it ---
    BPETokenizer tokenizer;

    // Option A: Train from a text corpus (do once)
    // tokenizer.train("data/corpus.txt", 50000);
    // tokenizer.save_token("tokenizer.model");

    // Option B: Load pre-trained tokenizer
    tokenizer.load_token("tokenizer.model");

    // --- Initialize model ---
    const int vocab = tokenizer.vocab.size(), dim = 1024, hidden = 4096, layers = 24, heads = 16;
    Model model(vocab, dim, hidden, layers, heads);

    // --- Example text input ---
    std::string text = "Hello world, this is a small GPT model.";
    std::vector<int> tokens = tokenizer.encode(text);

    // For demonstration, set the target as "next token"
    std::vector<int> target(tokens.size());
    for (size_t i = 0; i < tokens.size(); i++)
        target[i] = (tokens[i] + 1) % vocab;

    float lr = 1e-4f;

    // --- Training loop ---
    for (int epoch = 0; epoch < 100; epoch++) {
        Tensor pred = model.forward(tokens);
        Tensor grad(pred.rows, pred.cols);
        float loss = cross_entropy(pred, target, grad);
        model.step(lr);

        if (epoch % 10 == 0)
            std::cout << "Epoch " << epoch << " | Loss=" << loss << "\n";
    }

    model.save("./models/CarbonLLM_250M.cb");
    std::cout << "Saved weights to CarbonLLM_250M.cb\n";

    // --- Decode some output for fun ---
    std::string decoded = tokenizer.decode(tokens);
    std::cout << "Decoded tokens: " << decoded << "\n";

    return 0;
}

