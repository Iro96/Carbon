#pragma once
#include "layers.hpp"

struct FeedForward {
    Linear l1,l2; Tensor a_cache;
    FeedForward(int d,int h):l1(d,h),l2(h,d){}
    Tensor forward(const Tensor&x){
        a_cache=l1.forward(x);
        Tensor y(a_cache.rows,a_cache.cols);
        for(int i=0;i<a_cache.rows;i++)
            for(int j=0;j<a_cache.cols;j++) y(i,j)=std::max(0.0f,a_cache(i,j));
        return l2.forward(y);
    }
    Tensor backward(const Tensor&grad_out){
        Tensor grad_y=l2.backward(grad_out);
        for(int i=0;i<a_cache.rows;i++)
            for(int j=0;j<a_cache.cols;j++)
                grad_y.grad[i*a_cache.cols+j]*=(a_cache(i,j)>0)?1.0f:0.0f;
        return l1.backward(grad_y);
    }
    void step(float lr){l1.step(lr);l2.step(lr);}
    void save(std::ofstream&f)const{l1.save(f);l2.save(f);}
    void load(std::ifstream&f){l1.load(f);l2.load(f);}
};
