#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <map>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <iostream>
#include <set>
#include <iomanip>
#include <iterator>

struct BPETokenizer {
    // ===== Core Data =====
    std::unordered_map<std::string, int> vocab;
    std::vector<std::string> rev_vocab;
    std::map<std::pair<std::string, std::string>, int> pair_freq;
    std::vector<std::pair<std::string, std::string>> merges;
    std::unordered_map<std::string, int> freq;

    // ===== Special Tokens =====
    const std::vector<std::string> special_tokens = {"<unk>", "<pad>", "<bos>", "<eos>"};

    // ===== UTF-8 Aware Split =====
    static std::vector<std::string> split_chars(const std::string& text) {
        std::vector<std::string> out;
        for (size_t i = 0; i < text.size();) {
            unsigned char c = static_cast<unsigned char>(text[i]);
            size_t len = 1;
            if      ((c & 0xE0) == 0xC0) len = 2;
            else if ((c & 0xF0) == 0xE0) len = 3;
            else if ((c & 0xF8) == 0xF0) len = 4;
            out.emplace_back(text.substr(i, len));
            i += len;
        }
        return out;
    }

    // Convert vector of tokens to string
    static std::string word_to_string(const std::vector<std::string>& w) {
        std::string s;
        for (const auto& x : w) s += x;
        return s;
    }

    // ===== Count Frequencies of Symbol Pairs =====
    void count_pairs(const std::vector<std::vector<std::string>>& corpus) {
        pair_freq.clear();
        for (const auto& word : corpus) {
            if (word.size() < 2) continue;
            const auto key = word_to_string(word);
            const int weight = freq.count(key) ? freq.at(key) : 1;
            for (size_t i = 0; i + 1 < word.size(); ++i)
                pair_freq[{word[i], word[i + 1]}] += weight;
        }
    }

    // ===== Find Most Frequent Pair =====
    [[nodiscard]] std::pair<std::string, std::string> most_frequent_pair() const {
        std::pair<std::string, std::string> best_pair;
        int best_count = 0;
        for (const auto& [pair, count] : pair_freq) {
            if (count > best_count) {
                best_count = count;
                best_pair = pair;
            }
        }
        return best_pair;
    }

    // ===== Merge a Pair in Corpus =====
    static void merge_pair(std::vector<std::vector<std::string>>& corpus,
                           const std::pair<std::string, std::string>& pair) {
        const std::string merged = pair.first + pair.second;
        for (auto& word : corpus) {
            std::vector<std::string> new_word;
            for (size_t i = 0; i < word.size();) {
                if (i + 1 < word.size() && word[i] == pair.first && word[i + 1] == pair.second) {
                    new_word.push_back(merged);
                    i += 2;
                } else {
                    new_word.push_back(word[i++]);
                }
            }
            word.swap(new_word);
        }
    }

    // ===== Train BPE Tokenizer =====
    void train(const std::string& text_path, int vocab_size = 50000, bool verbose = true) {
        std::ifstream f(text_path);
        if (!f) {
            std::cerr << "[BPE] Error: cannot open file: " << text_path << "\n";
            return;
        }

        freq.clear();
        std::string word;
        while (f >> word)
            freq["▁" + word]++;

        std::vector<std::vector<std::string>> corpus;
        corpus.reserve(freq.size());
        for (const auto& [w, _] : freq)
            corpus.push_back(split_chars(w));

        // ===== Initialize Vocab =====
        vocab.clear();
        rev_vocab.clear();
        for (const auto& t : special_tokens) {
            vocab[t] = static_cast<int>(vocab.size());
            rev_vocab.push_back(t);
        }

        std::set<std::string> symbols;
        for (const auto& [w, _] : freq)
            for (const auto& c : split_chars(w))
                symbols.insert(c);

        for (const auto& s : symbols) {
            vocab[s] = static_cast<int>(vocab.size());
            rev_vocab.push_back(s);
        }

        // ===== BPE Merging Loop =====
        while (static_cast<int>(vocab.size()) < vocab_size) {
            count_pairs(corpus);
            auto best_pair = most_frequent_pair();
            if (pair_freq.empty() || pair_freq[best_pair] < 2) break;

            merge_pair(corpus, best_pair);
            const std::string merged = best_pair.first + best_pair.second;
            vocab[merged] = static_cast<int>(vocab.size());
            rev_vocab.push_back(merged);
            merges.push_back(best_pair);

            if (verbose && vocab.size() % 1000 == 0)
                std::cout << "[BPE] Merges: " << vocab.size() << "\n";
        }

        if (verbose)
            std::cout << "[BPE] Training complete. Final vocab size = " << vocab.size() << "\n";
    }

    // ===== Apply Merges =====
    [[nodiscard]] std::vector<std::string> apply_bpe(std::vector<std::string> tokens) const {
        for (const auto& [a, b] : merges) {
            for (size_t i = 0; i + 1 < tokens.size();) {
                if (tokens[i] == a && tokens[i + 1] == b) {
                    tokens[i] = a + b;
                    tokens.erase(tokens.begin() + i + 1);
                } else ++i;
            }
        }
        return tokens;
    }

    // ===== Encode / Decode =====
    [[nodiscard]] std::vector<int> encode(const std::string& text) const {
        std::istringstream iss(text);
        std::string word;
        std::vector<int> out;
        while (iss >> word) {
            auto chars = split_chars("▁" + word);
            auto merged = apply_bpe(std::move(chars));
            for (const auto& t : merged) {
                if (auto it = vocab.find(t); it != vocab.end())
                    out.push_back(it->second);
                else
                    out.push_back(vocab.at("<unk>"));
            }
        }
        return out;
    }

    [[nodiscard]] std::string decode(const std::vector<int>& tokens) const {
        std::string out;
        for (int t : tokens) {
            if (t >= 0 && t < static_cast<int>(rev_vocab.size()))
                out += rev_vocab[t];
        }
        std::replace(out.begin(), out.end(), '▁', ' ');
        return out;
    }

    // ===== Save / Load =====
    void save_token(const std::string& path) const {
        std::ofstream f(path);
        if (!f) {
            std::cerr << "[BPE] Error: cannot write to file: " << path << "\n";
            return;
        }
        f << "#vocab " << vocab.size() << "\n";
        for (const auto& [token, id] : vocab)
            f << token << " " << id << "\n";

        f << "#merges " << merges.size() << "\n";
        for (const auto& [a, b] : merges)
            f << a << " " << b << "\n";
    }

    void load_token(const std::string& path) {
        vocab.clear();
        rev_vocab.clear();
        merges.clear();

        std::ifstream f(path);
        if (!f) {
            std::cerr << "[BPE] Error: cannot open token file: " << path << "\n";
            return;
        }

        std::string header;
        while (f >> header) {
            if (header == "#vocab") {
                size_t n; f >> n;
                for (size_t i = 0; i < n; ++i) {
                    std::string token; int id;
                    f >> token >> id;
                    vocab[token] = id;
                }
                rev_vocab.resize(vocab.size());
                for (const auto& [tok, id] : vocab)
                    if (id >= 0 && id < static_cast<int>(rev_vocab.size()))
                        rev_vocab[id] = tok;
            } else if (header == "#merges") {
                size_t m; f >> m;
                merges.reserve(m);
                for (size_t i = 0; i < m; ++i) {
                    std::string a, b;
                    f >> a >> b;
                    merges.emplace_back(a, b);
                }
            }
        }
    }
};
