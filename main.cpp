#include "model.hpp"
#include <iostream>
#include <random>

int main(){
    const int vocab=50000, dim=1024, hidden=4096, layers=24, heads=16;
    Model model(vocab,dim,hidden,layers,heads);
    std::mt19937 gen(1);
    std::uniform_int_distribution<int>d(0,vocab-1);
    std::vector<int> tokens(32), target(32);
    float lr=1e-4f;

    for(int epoch=0;epoch<1000;epoch++){
        for(size_t i=0;i<tokens.size();i++){ tokens[i]=d(gen); target[i]=(tokens[i]+1)%vocab; }
        Tensor pred=model.forward(tokens);
        Tensor grad(pred.rows,pred.cols);
        float loss=cross_entropy(pred,target,grad);
        model.step(lr);
        if(epoch%10==0) std::cout<<"Epoch "<<epoch<<" | Loss="<<loss<<"\n";
    }

    model.save("./models/CarbonLLM_250M.cb");
    std::cout<<"Saved weights to CarbonLLM_250M.cb\n";
    return 0;
}