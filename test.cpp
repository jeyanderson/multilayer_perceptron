#include "matrix.hpp"

int main(){
  Matrix M = mtx<float>::randn(2,2); // init randn matrix

  int i =0;
  M.printShape();
  M.print(); // print the OG matrix

  (M-M).print();  // print M minus itself

  (M+M).print();  // print its sum
  std::cout<<"by scalar"<<std::endl;
  (M.multiply_scalar(2.f)).print();  // print 2x itself
  std::cout<<"by itself"<<std::endl;
  (M.multiply_elementwise(M)).print(); // mult M w itself
  std::cout<<"transpose"<<std::endl;
  auto MT = M.T(); // transpose the matrix
  MT.print();
    std::cout<<"matmul"<<std::endl;
  (MT.matmul(M)).print();  // form symm. pos. def. matrix

  (M.applyFunction([](int x){return x-x;} )).print(); // apply fun
}