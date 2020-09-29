#ifndef DISTMATRIX_MATRIX_H
#define DISTMATRIX_MATRIX_H

#include <memory>

template<class ValueType>
struct Matrix {
    int nlocal;
    int nrows;
    int ncols;
    std::shared_ptr<ValueType[]> array;
    Matrix(int nrows, int ncols);
    const ValueType &operator()(int i, int j);
    void copy_to(Matrix<ValueType> &A);
    void set(int i, int j, const ValueType x);

    void operator=(const Matrix<ValueType> &other);
    void operator=(const ValueType x);
    void operator=(std::initializer_list<ValueType> x);

    void operator*=(const ValueType x);
    void operator+=(const ValueType x);
    void operator+=(const Matrix<ValueType>& A);

    Matrix<bool> operator==(const ValueType x);
    Matrix<bool> operator==(const Matrix<ValueType> &x);
    bool all_equal(const ValueType x);
    bool all_equal(const Matrix<ValueType> &A);

    ValueType sum();
    void print();

private:
    inline int flatten(int i, int j);
};

#endif//DISTMATRIX_MATRIX_H
