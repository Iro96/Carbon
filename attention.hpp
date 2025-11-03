#pragma once
#include "layers.hpp"

struct MultiHeadAttention {
    int dim, heads, head_dim;
    Linear q_proj,k_proj,v_proj,o_proj;
    Tensor Q,K,V;
    MultiHeadAttention(int d,int h):dim(d),heads(h),head_dim(d/h),
        q_proj(d,d),k_proj(d,d),v_proj(d,d),o_proj(d,d){}
    Tensor forward(const Tensor&x){
        Q=q_proj.forward(x); K=k_proj.forward(x); V=v_proj.forward(x);
        Tensor out(x.rows,dim);
        for(int head=0;head<heads;head++){
            int offset=head*head_dim;
            Tensor scores(x.rows,x.rows);
            for(int i=0;i<x.rows;i++)
                for(int j=0;j<x.rows;j++)
                    scores(i,j)=Tensor::dot_simd(&Q.val[i*dim+offset],&K.val[j*dim+offset],head_dim)/std::sqrt((float)head_dim);
            for(int i=0;i<scores.rows;i++){
                float maxv=-1e9,sum=0;
                for(int j=0;j<scores.cols;j++) maxv=std::max(maxv,scores(i,j));
                for(int j=0;j<scores.cols;j++){scores(i,j)=exp(scores(i,j)-maxv);sum+=scores(i,j);}
                for(int j=0;j<scores.cols;j++) scores(i,j)/=sum;
            }
            for(int i=0;i<x.rows;i++)
                for(int j=0;j<head_dim;j++){
                    float s=0;
                    for(int k=0;k<x.rows;k++) s+=scores(i,k)*V(k,offset+j);
                    out(i,offset+j)=s;
                }
        }
        return o_proj.forward(out);
    }
    Tensor backward(const Tensor&grad_out){return o_proj.backward(grad_out);}
    void step(float lr){q_proj.step(lr);k_proj.step(lr);v_proj.step(lr);o_proj.step(lr);}
    void save(std::ofstream&f)const{q_proj.save(f);k_proj.save(f);v_proj.save(f);o_proj.save(f);}
    void load(std::ifstream&f){q_proj.load(f);k_proj.load(f);v_proj.load(f);o_proj.load(f);}
};
