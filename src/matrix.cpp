#include <matrix.h>

#include <algorithm>
#include <complex>
#include <exception>
#include <functional>
#include <iostream>
#include <numeric>
#include <type_traits>

#ifdef USE_MKL
#include <mkl.h>
#else
#include <cblas.h>
#include <lapacke.h>
#endif

template<class ValueType>
Matrix<ValueType>::Matrix(int nrows, int ncols) : nlocalrows(nrows), nlocalcols(ncols),
                                                  nlocal(nrows * ncols), array(new ValueType[nrows * ncols]),
                                                  nrows(nrows), ncols(ncols) {
    // array = std::make_shared<ValueType[]>(nlocal);
    ;
}

template<class ValueType>
int Matrix<ValueType>::flatten(int i, int j) {
    return j * nlocalrows + i;
}
template<class ValueType>
std::pair<int, int> Matrix<ValueType>::unflatten(int idx) {
    int j = idx / nlocalrows;
    int i = idx - j * nlocalrows;
    return {i, j};
}

template<class ValueType>
void Matrix<ValueType>::set(int i, int j, const ValueType x) {
    array[flatten(i, j)] = x;
}

template<class ValueType>
ValueType Matrix<ValueType>::operator()(int i, int j) {
    return array[flatten(i, j)];
}

template<class ValueType>
void Matrix<ValueType>::print() {
    for (int i = 0; i < nrows; i++) {
        for (int j = 0; j < ncols; j++) {
            std::cout << operator()(i, j) << " ";
        }
        std::cout << "\n";
    }
}

/*template<class ValueType>
void Matrix<ValueType>::operator=(const Matrix<ValueType> &other) {
    nlocalrows = other.nlocalrows;
    nlocalcols = other.nlocalcols;
    nlocal = other.nlocal;
    array = other.array;
}*/

template<class ValueType>
void Matrix<ValueType>::operator=(const ValueType x) {
    for (int i = 0; i < nlocal; i++) {
        array[i] = x;
    }
}

template<class ValueType>
void Matrix<ValueType>::operator=(std::initializer_list<ValueType> x) {
    for (int k = 0; k < nlocal; k++) {
        auto [i, j] = unflatten(k);
        int idx = j * nrows + i;
        set(i, j, *(x.begin() + idx));
    }
    std::cout << std::endl;
}

template<class ValueType>
void Matrix<ValueType>::operator=(std::function<ValueType(int, int)> f) {
    for (int idx = 0; idx < nlocal; idx++) {
        auto [i, j] = unflatten(idx);
        array[idx] = f(i, j);
    }
}

template<class ValueType>
void Matrix<ValueType>::operator=(std::function<ValueType(ValueType, int, int)> f) {
    for (int idx = 0; idx < nlocal; idx++) {
        auto [i, j] = unflatten(idx);
        array[idx] = f(array[idx], i, j);
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
bool Matrix<ValueType>::operator==(const ValueType x) {
    bool result = true;
    for (int i = 0; i < nlocal; i++) {
        result = result && (array[i] == x);
    }
    return result;
}


template<class ValueType>
bool Matrix<ValueType>::operator==(const Matrix<ValueType> &A) {
    bool result = true;
    for (int i = 0; i < nlocal; i++) {
        result = result && (array[i] == A.array[i]);
    }
    return result;
}

template<class ValueType>
Matrix<ValueType> Matrix<ValueType>::matmul(const Matrix<ValueType> &B, const ValueType alpha) {
    Matrix<ValueType> C(nlocalrows, B.nlocalcols);
    ValueType beta(0);
    ValueType *A_ptr = array.get(), *B_ptr = B.array.get(), *C_ptr = C.array.get();
    int m = nlocalrows, k = nlocalcols, n = B.nlocalcols;

    if constexpr (std::is_same_v<ValueType, float>) {
        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, A_ptr, m, B_ptr, k, beta, C_ptr, m);
    } else if constexpr (std::is_same_v<ValueType, double>) {
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, A_ptr, m, B_ptr, k, beta, C_ptr, m);
    } else if constexpr (std::is_same_v<ValueType, std::complex<float>>) {
        cblas_cgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, k, &alpha, A_ptr, m, B_ptr, k, &beta, C_ptr, m);
    } else if constexpr (std::is_same_v<ValueType, std::complex<double>>) {
        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, k, &alpha, A_ptr, m, B_ptr, k, &beta, C_ptr, m);
    } else {
        throw std::logic_error("matmul called with unsupported type");
    }
    return C;
}

template<class ValueType>
std::pair<std::vector<typename Matrix<ValueType>::ComplexValueType>, Matrix<typename Matrix<ValueType>::ComplexValueType>> Matrix<ValueType>::diagonalize() {
    std::vector<ComplexValueType> eigvals(nlocalrows);
    Matrix<ComplexValueType> eigvecs(nlocalrows, nlocalrows);
    Matrix<ValueType> Acopy(nlocalrows, nlocalrows);
    copy_to(Acopy);
    int info;

    if constexpr (std::is_same_v<ValueType, std::complex<float>>) {
        info = LAPACKE_cgeev(LAPACK_COL_MAJOR, 'N', 'V', nlocalrows, (lapack_complex_float *) Acopy.array.get(), nlocalrows, (lapack_complex_float *) eigvals.data(), nullptr, 1, (lapack_complex_float *) eigvecs.array.get(), nlocalrows);
    } else if constexpr (std::is_same_v<ValueType, std::complex<double>>) {
        info = LAPACKE_zgeev(LAPACK_COL_MAJOR, 'N', 'V', nlocalrows, (lapack_complex_double *) Acopy.array.get(), nlocalrows, (lapack_complex_double *) eigvals.data(), nullptr, 1, (lapack_complex_double *) eigvecs.array.get(), nlocalrows);
    } else if constexpr (std::is_same_v<ValueType, float> || std::is_same_v<ValueType, double>) {
        std::vector<ValueType> eigvals_re(nlocalrows), eigvals_im(nlocalrows);
        Matrix<ValueType> tmp_eigvecs(nlocalrows, nlocalrows);
        if constexpr (std::is_same_v<ValueType, float>) {
            info = LAPACKE_sgeev(LAPACK_COL_MAJOR, 'N', 'V', nlocalrows, Acopy.array.get(), nlocalrows, eigvals_re.data(), eigvals_im.data(), nullptr, 1, tmp_eigvecs.array.get(), nlocalrows);
        } else if constexpr (std::is_same_v<ValueType, double>) {
            info = LAPACKE_dgeev(LAPACK_COL_MAJOR, 'N', 'V', nlocalrows, Acopy.array.get(), nlocalrows, eigvals_re.data(), eigvals_im.data(), nullptr, 1, tmp_eigvecs.array.get(), nlocalrows);
        }
        for (int i = 0; i < nlocalrows; i++) {
            eigvals[i] = std::complex<ValueType>(eigvals_re[i], eigvals_im[i]);
        }
        for (int j = 0; j < nlocalrows; j++) {
            // unwrap potentially complex eigenvectors stores as reals according to DGEEV docs
            if (std::norm(eigvals[j] - std::conj(eigvals[j])) < 1e-15) {// real eigenvalue
                for (int i = 0; i < nlocalrows; i++) {
                    eigvecs.set(i, j, tmp_eigvecs(i, j));
                }
            } else if (std::norm(eigvals[j] - std::conj(eigvals[j + 1])) < 1e-15) {
                for (int i = 0; i < nlocalrows; i++) {
                    std::complex<ValueType> tmp(tmp_eigvecs(i, j), tmp_eigvecs(i, j + 1));
                    eigvecs.set(i, j, tmp);
                    eigvecs.set(i, j + 1, std::conj(tmp));
                }
            }
        }
    } else {
        throw std::logic_error("this should only be called for complex<float/double>");
    }
    if (info) {
        std::cerr << "info = " << info << std::endl;
        throw std::runtime_error("error in LAPACKE_?geev");
    }
    return {eigvals, eigvecs};
}


template class Matrix<bool>;
template class Matrix<int>;
template class Matrix<float>;
template class Matrix<double>;
template class Matrix<std::complex<float>>;
template class Matrix<std::complex<double>>;