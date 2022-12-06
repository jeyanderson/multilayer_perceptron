#include "matrix.hpp"
#include "mlp.hpp"
#include <fstream>
#include <deque>
#include <vector>

auto makeModel(size_t inChannels,size_t outChannels,size_t hiddenUnitsPerLayer,size_t hiddenLayers,float lr){
    std::vector<size_t> unitsPerLayer;

    unitsPerLayer.push_back(inChannels);
    for(int i=0;i<hiddenLayers;++i)
        unitsPerLayer.push_back(hiddenUnitsPerLayer);
    unitsPerLayer.push_back(outChannels);
    nn::MLP<float> model(unitsPerLayer,.01f);
    return model;
}

auto model=makeModel(1,1,8,3,.5f);