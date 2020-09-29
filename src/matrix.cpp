#include "matrix.h"

#include <algorithm>
#include <complex>
#include <iostream>
#include <numeric>

template<class ValueType>
Matrix<ValueType>::Matrix(int nrows, int ncols) : nrows(nrows), ncols(ncols), nlocal(nrows * ncols), array(new ValueType[nrows * ncols]) {
    // array = std::make_shared<ValueType[]>(nlocal);
    ;
}

template<class ValueType>
int Matrix<ValueType>::flatten(int i, int j) {
    return j * nrows + i;
}

template<class ValueType>
void Matrix<ValueType>::set(int i, int j, const ValueType x) {
    array[flatten(i, j)] = x;
}

template<class ValueType>
const ValueType &Matrix<ValueType>::operator()(int i, int j) {
    return array[flatten(i, j)];
}

template<class ValueType>
void Matrix<ValueType>::print() {
    for (int i = 0; i < nrows; i++) {
        for (int j = 0; j < nrows; j++) {
            std::cout << array[flatten(i, j)] << " ";
        }
        std::cout << "\n";
    }
}

template<class ValueType>
void Matrix<ValueType>::operator=(const Matrix<ValueType> &other) {
    nrows = other.nrows;
    ncols = other.ncols;
    nlocal = other.nlocal;
    array = other.array;
}

template<class ValueType>
void Matrix<ValueType>::operator=(const ValueType x) {
    for (int i = 0; i < nlocal; i++) {
        array[i] = x;
    }
}

template<class ValueType>
void Matrix<ValueType>::operator=(std::initializer_list<ValueType> x) {
    for (int i = 0; i < nlocal; i++) {
        array[i] = *(x.begin() + i);
    }
}

template<class ValueType>
void Matrix<ValueType>::copy_to(Matrix<ValueType> &A) {
    for (int i = 0; i < nlocal; i++) {
        A.array[i] = array[i];
    }
}

template<class ValueType>
void Matrix<ValueType>::operator*=(const ValueType x) {
    for (int i = 0; i < nlocal; i++) {
        array[i] *= x;
    }
}
template<class ValueType>
void Matrix<ValueType>::operator+=(const ValueType x) {
    for (int i = 0; i < nlocal; i++) {
        array[i] += x;
    }
}

template<class ValueType>
void Matrix<ValueType>::operator+=(const Matrix<ValueType> &A) {
    for (int i = 0; i < nlocal; i++) {
        array[i] += A.array[i];
    }
}

template<class ValueType>
ValueType Matrix<ValueType>::sum() {
    return std::accumulate(array.get(), array.get() + nlocal, ValueType(0));
}

template<class ValueType>
Matrix<bool> Matrix<ValueType>::operator==(const ValueType x) {
    Matrix<bool> A(nrows, ncols);
    for (int i = 0; i < nlocal; i++) {
        A.array[i] = array[i] == x;
    }
    return A;
}

template<class ValueType>
bool Matrix<ValueType>::all_equal(const ValueType x) {
    bool result = true;
    for (int i = 0; i < nlocal; i++) {
        result = result && (array[i] == x);
    }
    return result;
}

template<class ValueType>
Matrix<bool> Matrix<ValueType>::operator==(const Matrix<ValueType> &A) {
    Matrix<bool> result(nrows, ncols);
    for (int i = 0; i < nlocal; i++) {
        result.array[i] = array[i] == A.array[i];
    }
    return result;
}

template<class ValueType>
bool Matrix<ValueType>::all_equal(const Matrix<ValueType> &A) {
    bool result = true;
    for (int i = 0; i < nlocal; i++) {
        result = result && (array[i] == A.array[i]);
    }
    return result;
}


template class Matrix<bool>;
template class Matrix<int>;
template class Matrix<float>;
template class Matrix<double>;
template class Matrix<std::complex<float>>;
template class Matrix<std::complex<double>>;