#pragma once
#include "tensor.hpp"

struct Embedding {
    Tensor table;
    std::vector<int> last_tokens;
    Embedding(int vocab,int dim):table(vocab,dim){table.randomize();}
    Tensor forward(const std::vector<int>&tokens){
        last_tokens=tokens;
        Tensor out(tokens.size(),table.cols);
        for(size_t i=0;i<tokens.size();i++)
            for(int j=0;j<table.cols;j++)
                out(i,j)=table(tokens[i],j);
        return out;
    }
    Tensor backward(const Tensor&grad_out){
        for(size_t i=0;i<last_tokens.size();i++)
            for(int j=0;j<table.cols;j++)
                table.grad[last_tokens[i]*table.cols+j]+=grad_out.grad[i*grad_out.cols+j];
        return grad_out;
    }
    void step(float lr){
        for(size_t i=0;i<table.val.size();i++){
            table.val[i]-=lr*table.grad[i];
            table.grad[i]=0;
        }
    }
    void save(std::ofstream&f)const{table.save(f);}
    void load(std::ifstream&f){table.load(f);}
};
