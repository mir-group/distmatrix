#ifndef DISTMATRIX_MATRIX_H
#define DISTMATRIX_MATRIX_H

#include <complex>
#include <functional>
#include <memory>

template<class ValueType>
class Matrix {
public:
    int nlocal;
    int nrows, ncols;
    int nlocalrows;
    int nlocalcols;
    std::shared_ptr<ValueType[]> array;
    Matrix(int nrows, int ncols);

    virtual ValueType operator()(int i, int j);
    virtual void set(int i, int j, const ValueType x);

    void copy_to(Matrix<ValueType> &A);

    //    void operator=(const Matrix<ValueType> &other);
    void operator=(const ValueType x);
    void operator=(std::initializer_list<ValueType> x);
    void operator=(std::function<ValueType(int, int)> f);
    void operator=(std::function<ValueType(ValueType, int, int)> f);

    void operator*=(const ValueType x);
    void operator+=(const ValueType x);
    void operator+=(const Matrix<ValueType> &A);

    /**
        Matrix product
    */
    Matrix<ValueType> matmul(const Matrix<ValueType> &B, const ValueType alpha = ValueType(1));

    typedef typename std::conditional_t<
            std::is_same_v<ValueType, std::complex<float>> || std::is_same_v<ValueType, std::complex<double>>,
            ValueType,
            std::complex<ValueType>>
            ComplexValueType;
    std::pair<std::vector<ComplexValueType>, Matrix<ComplexValueType>> diagonalize();

    virtual bool operator==(const ValueType x);
    virtual bool operator==(const Matrix<ValueType> &x);


    virtual ValueType sum();
    void print();
    virtual void fence() { ; };

protected:
    virtual int flatten(int i, int j);
    virtual std::pair<int, int> unflatten(int idx);
};

#endif//DISTMATRIX_MATRIX_H
