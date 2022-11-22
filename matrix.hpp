// matrix.hpp
#pragma once
#include <vector>
#include <cmath>
#include <tuple>
#include <cassert>
#include <iostream>
#include <tuple>
#include <functional>
#include <random>

template<typename Type>
class Matrix{
    size_t cols;
    size_t rows;

    public:
        std::vector<Type> data;
        std::tuple<size_t,size_t> shape;
        int numel=rows*cols;

    Matrix(size_t rows,size_t cols):rows(rows),cols(cols),data({}){
        data.resize(rows*cols,Type());
        shape={rows,cols};
    }
    Matrix():cols(0),rows(0),data({}){shape={rows,cols};};

    void printShape(){
        std::cout<<"Matrix shape(["<<rows<<","<<cols<<"])"<<std::endl;
    }

    void print(){
        for(size_t r=0;r<rows;r++){
            for(size_t c=0;c<cols;c++){
                std::cout<<(*this)(r,c)<<' ';
            }
            std::cout<<std::endl;
        }
        std::cout<<std::endl;
    }

    Type& operator()(size_t row,size_t col){
        return data[row*cols+col];
    }

    Matrix matmul(Matrix &target){
        assert(cols==target.rows);
        Matrix output(rows,target.cols);
        for(size_t r=0;r<output.rows;r++){
            for(size_t c=0;c<output.cols;c++){
                for(size_t k=0;k<target.rows;k++){
                    output(r,c)+=(*this)(r,k)*target(k,c);
                }
            }
        }
        return output;
    }
};

// 1 2 3
// 1 2 3
// 1 2 3

//1 2 3 4 5
//1 2 3 4 5
//1 2 3 4 5

//6 12 18 24 30
//0 0 0 0 0
//0 0 0 0 0

// 1
// 2
// 3
// 1
// 2
// 3
// 1
// 2
// 3