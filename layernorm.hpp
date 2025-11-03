#pragma once
#include "tensor.hpp"

struct LayerNorm {
    Tensor gamma,beta,x_cache,mean_cache,var_cache;
    int dim;
    LayerNorm(int d):gamma(1,d),beta(1,d),dim(d){
        for(int i=0;i<d;i++){gamma.val[i]=1.0f;beta.val[i]=0.0f;}
    }
    Tensor forward(const Tensor&x){
        x_cache=x;
        Tensor y(x.rows,x.cols);
        mean_cache=Tensor(1,x.cols);
        var_cache=Tensor(1,x.cols);
        for(int j=0;j<x.cols;j++){
            float mean=0;
            for(int i=0;i<x.rows;i++) mean+=x(i,j);
            mean/=x.rows;
            float var=0;
            for(int i=0;i<x.rows;i++) var+=(x(i,j)-mean)*(x(i,j)-mean);
            var/=x.rows;
            mean_cache(0,j)=mean;var_cache(0,j)=var;
            float inv_std=1.0f/std::sqrt(var+1e-5f);
            for(int i=0;i<x.rows;i++)
                y(i,j)=(x(i,j)-mean)*inv_std*gamma.val[j]+beta.val[j];
        }
        return y;
    }
    Tensor backward(const Tensor&grad_out){
        Tensor grad_in(x_cache.rows,x_cache.cols);
        for(int j=0;j<x_cache.cols;j++){
            float mean=mean_cache(0,j),var=var_cache(0,j);
            float inv_std=1.0f/std::sqrt(var+1e-5f);
            float dgamma=0,dbeta=0;
            for(int i=0;i<x_cache.rows;i++){
                float xmu=x_cache(i,j)-mean;
                dgamma+=grad_out.grad[i*x_cache.cols+j]*xmu*inv_std;
                dbeta+=grad_out.grad[i*x_cache.cols+j];
            }
            gamma.grad[j]+=dgamma;
            beta.grad[j]+=dbeta;
            for(int i=0;i<x_cache.rows;i++){
                float N=x_cache.rows;
                float dxhat=grad_out.grad[i*x_cache.cols+j]*gamma.val[j];
                float xmu=x_cache(i,j)-mean;
                grad_in.grad[i*x_cache.cols+j]=inv_std*(dxhat - (1.0f/N)*dbeta - (xmu*inv_std*inv_std/N)*dgamma);
            }
        }
        return grad_in;
    }
    void step(float lr){
        for(size_t i=0;i<gamma.val.size();i++){
            gamma.val[i]-=lr*gamma.grad[i];gamma.grad[i]=0;
            beta.val[i]-=lr*beta.grad[i];beta.grad[i]=0;
        }
    }
    void save(std::ofstream&f)const{gamma.save(f);beta.save(f);}
    void load(std::ifstream&f){gamma.load(f);beta.load(f);}
};
