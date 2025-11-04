#include "model.hpp"
#include "tokenizer_bpe.hpp"
#include <iostream>
#include <random>

int main() {
    // --- Initialize tokenizer ---
    BPETokenizer tokenizer;

    // Train the tokenizer directly in code using a small text sample
    std::string tiny_corpus =
        "Hello world! This is a minimal tokenizer example. "
        "We are training a small BPE tokenizer inline.";
    tokenizer.train_from_string(tiny_corpus, 500); // assumes you added a helper for inline training
    // If BPETokenizer only supports train(file), you can write to a tmp file first.

    // --- Initialize model parameters ---
    const int vocab = tokenizer.vocab.size();
    const int dim = 256;
    const int hidden = 512;
    const int layers = 4;
    const int heads = 4;

    Model model(vocab, dim, hidden, layers, heads);

    // --- Example text input ---
    std::string text = "Hello world, this is a tiny GPT model.";
    std::vector<int> tokens = tokenizer.encode(text);

    // Create dummy target: "next token" prediction
    std::vector<int> target(tokens.size());
    for (size_t i = 0; i < tokens.size(); i++)
        target[i] = (tokens[i] + 1) % vocab;

    float lr = 1e-4f;

    // --- Training loop ---
    for (int epoch = 0; epoch < 50; epoch++) {
        Tensor pred = model.forward(tokens);
        Tensor grad(pred.rows, pred.cols);
        float loss = cross_entropy(pred, target, grad);
        model.step(lr);

        if (epoch % 10 == 0)
            std::cout << "Epoch " << epoch << " | Loss=" << loss << "\n";
    }

    // Optionally save weights
    model.save("./models/CarbonLLM_250M.cb");
    std::cout << "Saved weights to CarbonLLM_250M.cb\n";

    // --- Decode some output for fun ---
    std::string decoded = tokenizer.decode(tokens);
    std::cout << "Decoded tokens: " << decoded << "\n";

    return 0;
}
