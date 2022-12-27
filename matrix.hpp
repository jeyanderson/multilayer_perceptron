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
namespace linalg{
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
        shape=std::make_tuple(rows,cols);}
    Matrix():cols(0),rows(0),data({}){shape={rows,cols};};
    void printShape(){
        std::cout<<"Matrix shape(["<<rows<<","<<cols<<"])"<<std::endl;}
    void print(){
        for(size_t r=0;r<rows;++r){
            for(size_t c=0;c<cols;++c){
                std::cout<<(*this)(r,c)<<' ';}
            std::cout<<std::endl;}
        std::cout<<std::endl;}
    Type& operator()(size_t row,size_t col){
        assert (0 <= row && row < rows);
        assert (0 <= col && col < cols);
        return data[row*cols+col];}
    Matrix matmul(Matrix &target) {
        assert(cols == target.rows);
        Matrix output(rows, target.cols);
        for (size_t r = 0; r < output.rows; ++r) {
        for (size_t c = 0; c < output.cols; ++c) {
            for (size_t k = 0; k < target.rows; ++k)
            output(r, c) += (*this)(r, k) * target(k, c);}}
        return output;}
    Matrix multiplyElementwise(Matrix &target){
        assert(shape==target.shape);
        Matrix output((*this));
        for(size_t r=0;r<rows;++r){
            for(size_t c=0;c<cols;++c){
                output(r,c)*=target(r,c);}}
        return output;}
    Matrix square(){
        return multiplyElementwise((*this));}
    Matrix multiplyScalar(Type scalar){
        Matrix output((*this));
        for(size_t r=0;r<rows;++r){
            for(size_t c=0;c<cols;++c){
                output(r,c)*=scalar;}}
        return output;}
    Matrix add(Matrix &target){
        assert(shape==target.shape);
        Matrix output((*this));
        for (size_t r=0;r<rows;++r){
            for(size_t c=0;c<cols;++c){
                output(r,c)+=target(r,c);}}
        return output;}
    Matrix operator+(Matrix &target){
        return add(target);}
    Matrix operator-(){
        Matrix output((*this));
        for(size_t r=0;r<rows;++r){
            for(size_t c=0;c<cols;++c){
                output(r,c)*=-1;}}
        return output;}
    Matrix sub(Matrix &target){
        Matrix negtarget(-target);
        return add(negtarget);}
    Matrix operator-(Matrix &target){
        return sub(target);}
    Matrix transpose(){
        Matrix transposed(cols,rows);
        for(size_t r=0;r<cols;++r){
            for(size_t c=0;c<rows;++c){
                transposed(r,c)=(*this)(c,r);}}
        return transposed;}
    Matrix T(){
        return transpose();}
    Matrix applyFunction(const std::function<Type(const Type &)> &function){
        Matrix output((*this));
        for(size_t r=0;r<rows;++r){
            for(size_t c=0;c<cols;++c){
                output(r,c)=function((*this)(r,c));}}
        return output;}};
template<typename T>
struct mtx{
    static Matrix<T> randn(size_t rows,size_t cols){
        Matrix<T> M(rows,cols);
        std::random_device rd{};
        std::mt19937 gen{rd()};
        T n(M.numel);
        T stdev{1/sqrt(n)};
        std::normal_distribution<T> d{0,stdev};
        for(size_t r=0;r<rows;++r){
            for(size_t c=0;c<cols;++c){
                M(r,c)=d(gen);}}
        return M;}
    static Matrix<T> rand(size_t rows, size_t cols) {
    Matrix<T> M{rows, cols};
    std::random_device rd{};
    std::mt19937 gen{rd()};
    std::uniform_real_distribution<T> d{0, 1};
    for (size_t r = 0; r < rows; ++r) {
      for (int c = 0; c < cols; ++c) {
        M(r, c) = d(gen);}}
    return M;}};}
