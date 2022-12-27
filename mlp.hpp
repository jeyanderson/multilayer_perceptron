#pragma once
#include "matrix.hpp"
#include <random>
#include <utility>
#include <cassert>
using namespace linalg;
namespace nn{
    template<typename T>
    class MLP{
        public:
        std::vector<size_t> unitsPerLayer;
        std::vector<Matrix<T>> biasVectors;
        std::vector<Matrix<T>> weightMatrices;
        std::vector<Matrix<T>> activations;
        float lr;
        explicit MLP(std::vector<size_t> unitsPerLayer,float lr=.001f):unitsPerLayer(unitsPerLayer),biasVectors(),weightMatrices(),lr(lr){
            for(size_t i=0;i<unitsPerLayer.size()-1;++i){
                size_t inChannels=unitsPerLayer[i],outChannels=unitsPerLayer[i+1];
                Matrix w=linalg::mtx<T>::randn(outChannels,inChannels);
                weightMatrices.push_back(w);
                Matrix b=linalg::mtx<T>::randn(outChannels,1);
                biasVectors.push_back(b);
                activations.resize(unitsPerLayer.size());}}
        static float sigmoid(float x){
            return 1.0f/(1+exp(-x));}
        static float dSigmoid(float x){
            return (x*(1-x));}
        Matrix<T> forward(Matrix<T> x){
            assert(get<0>(x.shape)==unitsPerLayer[0]&&get<1>(x.shape));
            activations[0]=x;
            Matrix prev(x);
            for(int i=0;i<unitsPerLayer.size()-1;++i){
                Matrix y=weightMatrices[i].matmul(prev);
                y=y+biasVectors[i];
                y=y.applyFunction(sigmoid);
                activations[i+1]=y;
                prev=y;}
            return prev;}
        void backprop(Matrix<T> target){
            assert(get<0>(target.shape)==unitsPerLayer.back());
            Matrix y=target;
            Matrix yHat=activations.back();
            Matrix error=target-yHat;
            for(int i=weightMatrices.size()-1;i>=0;--i){
                Matrix Wt=weightMatrices[i].T();
                Matrix prevErrors=Wt.matmul(error);
                Matrix dOutputs=activations[i+1].applyFunction(dSigmoid);
                Matrix gradients=error.multiplyElementwise(dOutputs);
                gradients=gradients.multiplyScalar(lr);
                Matrix At=activations[i].T();
                Matrix weightGradients=gradients.matmul(At);
                biasVectors[i]=biasVectors[i].add(gradients);
                weightMatrices[i]=weightMatrices[i].add(weightGradients);
                error=prevErrors;}}};}
