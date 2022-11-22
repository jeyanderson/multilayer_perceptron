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
        std::vector<Matrix<T>> biasVector;
        std::vector<Matrix<T>> weightMatrices;
        std::vector<Matrix<T>> activations;
        float lr;

        explicit MLP(std::vector<size_t> unitsPerLayer,float lr=.001f):unitsPerLayer(unitsPerLayer),biasVector(),weightMatrics(),lr(lr){
            for(size_t i=0;i<unitsPerLayer.size()-1;++i){
                size_t inChannels{unitsPerLayer[i]};
                size_t outChannels{unitsPerLayer[i]+1};

                auto w=linalg::mtx<T>::randn(outChannels,inChannels);
                weightMatrices.push_back(w);

                auto b=linalg::mtx<T>::randn(outChannels,1);
                biasVector.push_back(b);

                activations.resize(unitsPerLayer.size());
            }
        }
    };
}