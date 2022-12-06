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

void log(auto &file, const auto &x, const auto &y, const auto &y_hat){
  auto mse=(y.data[0]-y_hat.data[0]);
  mse=mse*mse;

  file<<mse<<" "<<x.data[0]<<" "<<y.data[0]<<" "<<y_hat.data[0]<<" \n";
}

int main(){
    int inChannels=1,outChannels=1,hiddenUnitsPerLayer=8,hiddenLayers=3;
    float lr=.5f;
    auto model=makeModel(
        inChannels,
        outChannels,
        hiddenUnitsPerLayer,
        hiddenLayers,
        lr);
    std::ofstream file;
    file.open ("data.txt");

    int maxIter=10000;
    float mse;

    const float PI=3.141592;
    for(int i=1;i<=maxIter;++i){
        auto x=mtx<float>::randn(inChannels,1).multiplyScalar(PI);
        auto y=x.applyFunction([](float v)->float{return sin(v)*sin(v);});
        auto yHat=model.forward(x);
        model.backprop(y);
        if ((i+1)%1500==0)
            log(file,x,y,yHat);
    }
    file.close();
}