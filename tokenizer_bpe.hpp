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

struct BPETokenizer {
    std::unordered_map<std::string,int> vocab;
    std::vector<std::string> rev_vocab;
    std::map<std::pair<std::string,std::string>,int> pair_freq;
    std::vector<std::pair<std::string,std::string>> merges;

    // Split text into characters
    static std::vector<std::string> split_chars(const std::string& text) {
        std::vector<std::string> out;
        for(char c : text) out.push_back(std::string(1,c));
        return out;
    }

    // Count frequencies of symbol pairs
    void count_pairs(const std::vector<std::vector<std::string>>& corpus) {
        pair_freq.clear();
        for(auto& word : corpus) {
            for(size_t i=0;i+1<word.size();++i)
                pair_freq[{word[i], word[i+1]}]++;
        }
    }

    // Find the most frequent pair
    std::pair<std::string,std::string> most_frequent_pair() {
        int best=0; std::pair<std::string,std::string> bp;
        for(auto &p:pair_freq) {
            if(p.second>best) { best=p.second; bp=p.first; }
        }
        return bp;
    }

    // Merge all occurrences of pair in corpus
    void merge_pair(std::vector<std::vector<std::string>>& corpus,
                    const std::pair<std::string,std::string>& pair) {
        std::string merged = pair.first + pair.second;
        for(auto &word:corpus){
            std::vector<std::string> new_word;
            for(size_t i=0;i<word.size();) {
                if(i+1<word.size() && word[i]==pair.first && word[i+1]==pair.second) {
                    new_word.push_back(merged);
                    i+=2;
                } else {
                    new_word.push_back(word[i]);
                    i++;
                }
            }
            word.swap(new_word);
        }
    }

    void train(const std::string& text_path, int vocab_size=50000, int verbose=1) {
        std::ifstream f(text_path);
        if(!f.is_open()){ std::cerr<<"Tokenizer: cannot open "<<text_path<<"\n"; return;}
        std::string word;
        std::unordered_map<std::string,int> freq;
        while(f>>word) freq[word]++;
        std::vector<std::vector<std::string>> corpus;
        for(auto &p:freq){
            auto chars = split_chars(p.first);
            corpus.push_back(chars);
        }

        // initial vocab: single characters
        std::set<std::string> symbols;
        for(auto &p:freq){
            for(char c: p.first) symbols.insert(std::string(1,c));
        }
        for(auto &s:symbols) vocab[s]=(int)vocab.size();
        for(auto &p:vocab) rev_vocab.push_back(p.first);

        while((int)vocab.size() < vocab_size) {
            count_pairs(corpus);
            auto best_pair = most_frequent_pair();
            if(pair_freq[best_pair] < 2) break; // no useful merges
            merge_pair(corpus,best_pair);
            std::string merged = best_pair.first + best_pair.second;
            vocab[merged]=(int)vocab.size();
            rev_vocab.push_back(merged);
            merges.push_back(best_pair);
            if(verbose && vocab.size()%1000==0)
                std::cout << "[BPE] Merges: " << vocab.size() << "\n";
        }
        std::cout << "BPE training done. vocab="<<vocab.size()<<"\n";
    }

    std::vector<std::string> apply_bpe(const std::vector<std::string>& chars) const {
        std::vector<std::string> tokens = chars;
        bool changed = true;
        while(changed) {
            changed = false;
            for(auto &pair: merges) {
                for(size_t i=0;i+1<tokens.size();) {
                    if(tokens[i]==pair.first && tokens[i+1]==pair.second) {
                        tokens[i]=pair.first+pair.second;
                        tokens.erase(tokens.begin()+i+1);
                        changed=true;
                    } else ++i;
                }
            }
        }
        return tokens;
    }

    std::vector<int> encode(const std::string& text) const {
        std::istringstream iss(text);
        std::string word;
        std::vector<int> out;
        while(iss>>word){
            auto chars=split_chars(word);
            auto merged=apply_bpe(chars);
            for(auto&t:merged){
                auto it=vocab.find(t);
                out.push_back(it==vocab.end()?0:it->second);
            }
        }
        return out;
    }

    std::string decode(const std::vector<int>& tokens) const {
        std::string out;
        for(int t:tokens) out += rev_vocab[t];
        return out;
    }

    void save(const std::string& path) const {
        std::ofstream f(path);
        for(auto &p:merges)
            f << p.first << " " << p.second << "\n";
    }

    void load(const std::string& path) {
        merges.clear();
        std::ifstream f(path);
        std::string a,b;
        while(f>>a>>b)
            merges.push_back({a,b});
    }
};