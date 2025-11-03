#pragma once
#include "layernorm.hpp"
#include "attention.hpp"
#include "feedforward.hpp"

struct TransformerBlock {
    LayerNorm ln1,ln2; MultiHeadAttention attn; FeedForward ff;
    TransformerBlock(int d,int h,int heads):ln1(d),ln2(d),attn(d,heads),ff(d,h){}
    Tensor forward(const Tensor&x){
        Tensor norm1=ln1.forward(x);
        Tensor attn_out=attn.forward(norm1);
        Tensor res1(x.rows,x.cols);
        for(size_t i=0;i<res1.val.size();i++) res1.val[i]=x.val[i]+attn_out.val[i];
        Tensor norm2=ln2.forward(res1);
        Tensor ff_out=ff.forward(norm2);
        Tensor out(res1.rows,res1.cols);
        for(size_t i=0;i<out.val.size();i++) out.val[i]=res1.val[i]+ff_out.val[i];
        return out;
    }
    void step(float lr){ln1.step(lr);ln2.step(lr);attn.step(lr);ff.step(lr);}
    void save(std::ofstream&f)const{ln1.save(f);ln2.save(f);attn.save(f);ff.save(f);}
    void load(std::ifstream&f){ln1.load(f);ln2.load(f);attn.load(f);ff.load(f);}
};
