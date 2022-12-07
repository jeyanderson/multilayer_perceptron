#include "matrix.hpp"
#include "mlp.hpp"
#include <fstream>
#include <deque>
//#include <iomanip>

// helper to initialize multi-layer perceptron with n hidden layers each w/ same num hidden units
auto makeModel(size_t inChannels,size_t outChannels,size_t hiddenUnitsPerLayer,int hiddenLayers,float lr){
  std::vector<size_t> unitsPerLayer;

  unitsPerLayer.push_back(inChannels);

  for (int i=0;i<hiddenLayers;++i)
    unitsPerLayer.push_back(hiddenUnitsPerLayer);

  unitsPerLayer.push_back(outChannels);

  nn::MLP<float> model(unitsPerLayer,.01f);
  return model;
}

void log(std::ofstream &file,const auto &x,const auto &y,const auto &yHat){
  float mse=(y.data[0]-yHat.data[0]);
  mse=mse*mse;
  file<<mse<<" "<<x.data[0]<< " "<<y.data[0]<<" "<<yHat.data[0]<<" \n";
}

int main(){
//  test_matrix();

  std::srand(42069);

  // init model
  int inChannels=1,outChannels=1,hiddenUnitsPerLayer=8,hiddenLayers=3;
  float lr=.5f;
  nn::MLP model=makeModel(
      inChannels=1,
      outChannels=1,
      hiddenUnitsPerLayer=8,
      hiddenLayers=3,
      lr=.5f);
  // train
  std::ofstream file;
  file.open ("data.txt");
  int maxIter=1000000,printEvery=500;
  float mse;
  std::deque deque=std::deque<float>(printEvery);
  for(int i=1;i<=maxIter;++i){
    // generate (x, y) training data
    Matrix x=mtx<float>::rand(inChannels,1).multiplyScalar(3.);
    Matrix y=x.applyFunction([](float v)->float{return sin(v)*sin(v);});

    Matrix yHat=model.forward(x); // forward pass
    model.backprop(y);// backward pass

    // compute and print error
    mse=(y-yHat).square().data[0];
    deque.push_back(mse);
    if ((i+1)%printEvery==0){
      log(file,x,y,yHat);
    }
  }
  file.close();

}