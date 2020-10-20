#ifndef DISTMATRIX_MATRIX_H
#define DISTMATRIX_MATRIX_H

#include <complex>
#include <functional>
#include <memory>

/**
 * A serial matrix class, mostly for  completion.
 * The DistMatrix class inherits from this and overrides the necessary functionality.
 * Most of the methods are written in such a way that they don't need overriding.
 * @tparam ValueType The type of values to be stored, e.g. `int`, `float`, `std::complex<double>`.
 */
template<class ValueType>
class Matrix {
public:
    int nlocal;                        ///< The number of local elements.
    int nrows;                         ///< The total number of rows.
    int ncols;                         ///< The total number of columns.
    int nlocalrows;                    ///< The local number of rows.
    int nlocalcols;                    ///< The local number of columns.
    std::shared_ptr<ValueType[]> array;///< The underlying array.

    /**
     * Construct a matrix with the given number of rows and columns.
     * @param nrows The number of rows.
     * @param ncols The number of columns.
     */
    Matrix(int nrows, int ncols);

    /**
     * Get an element of the matrix.
     * @param i The row index.
     * @param j The column index.
     * @return `A(i,j)`.
     */
    virtual ValueType operator()(int i, int j, bool lock = false);

    /**
     * Set an element of the matrix.
     * @param i The row index.
     * @param j The column index.
     * @param x The value to be set.
     * @param lock Ignored for serial matrix.
     */
    virtual void set(int i, int j, const ValueType x, bool lock = false);

    /**
     *  \brief Perform a deep copy to the matrix `A`
     *
     *  Note: The usual assignment operator has reference semantics,
     *  so `A = B` makes `A` point to the underlying data of `B`.
     * @param A The destination matrix.
     * @param lock Ignored for serial matrix.
     */
    void copy_to(Matrix<ValueType> &A);

    /**
     * Set all the elements equal to `x`.
     */
    void operator=(const ValueType x);

    /**
     * \brief Set the elements equal to `x`, in a column-major fashion.
     *
     * Example:
     * ```cpp
     * Matrix(2,3) A;
     * A = {1,2,3,4,5,6};
     * ```
     * which results in
     * ```cpp
     * A = 1 3 5
     *     2 4 6
     * ```
     * @param x A flattened list of values.
     */
    void operator=(std::initializer_list<ValueType> x);

    /**
     * \brief Element-wise initialization.
     *
     * Set `A(i,j) = f(i,j)`.
     * @param f A scalar function taking the indices as arguments. Typically a lambda,
     * which can capture whatever you want. For example, you can do matrix subtraction with
     * ```cpp
     * Matrix<double> A(3,4), B(3,4),C(3,4);
     * ...
     * A = [&B, &C] (int i, int j){
     *      return B(i,j) - C(i,j);
     * };
     * ```
     */
    void operator=(std::function<ValueType(int, int)> f);

    /**
     * \brief Element-wise transformation.
     *
     * Set `A(i,j) = f(A(i,j), i, j)`.
     * @param f A scalar function taking a matrix element and the indices as arguments. Typically a lambda,
     * which can capture whatever you want. For example, you can implement `A = alpha*A + beta*B` with
     * ```cpp
     * Matrix<double> A(3,4), B(3,4);
     * ...
     * A = [&B, alpha, beta] (double Aij, int i, int j){
     *      return alpha*Aij + beta*B(i,j);
     * };
     * ```
     */
    void operator=(std::function<ValueType(ValueType, int, int)> f);

    /**
     * Element-wise multiplication by `x`.
     */
    void operator*=(ValueType x);

    /**
     * Element-wise multiplication by `x`.
     */
    void operator+=(ValueType x);

    /**
     * Matrix addition.
     */
    void operator+=(const Matrix<ValueType> &A);

    /**
     * Matrix product, `C = alpha*A*B`.
     * @param B Matrix with which to multiply.
     * @param alpha Scalar prefactor. Defaults to 1.
     * @return The matrix `C`.
     */
    Matrix<ValueType> matmul(const Matrix<ValueType> &B, ValueType alpha = ValueType(1));

    /**
     * `ValueType` wrapped in `std::complex<>` if not already `std::complex<float>`
     * or `std::complex<double>`.
     */
    typedef typename std::conditional_t<
            std::is_same_v<ValueType, std::complex<float>> || std::is_same_v<ValueType, std::complex<double>>,
            ValueType,
            std::complex<ValueType>>
            ComplexValueType;

    /**
     * Diagonalize the matrix.
     *
     * *Note:* This does not exist for DistMatrix, since ScaLAPACK only implements symmetric eigensolvers.
     * @return A vector of eigenvalues and the matrix whose columns are eigenvectors.
     */
    std::pair<std::vector<ComplexValueType>, Matrix<ComplexValueType>> diagonalize();

    /**
     * Check if all elements are exactly equal to `x`.
     * @return `true` if all elements are equal to `x`, otherwise `false`.
     */
    virtual bool operator==(ValueType x);
    /**
     * Check if the matrices are element-wise equal.
     * @return `true` if all elements are equal, otherwise `false`.
     */
    virtual bool operator==(const Matrix<ValueType> &B);

    /**
     * Sum all the elements.
     * @return The sum.
     */
    virtual ValueType sum();

    /**
     * Print the matrix to `std::cout`.
     */
    void print();

    /**
     * Matrix inversion, using QR factorization for extra stability.
     *
     * *Note:* Only supported for `float` and `double`, as it uses `?ormqr`.
     * @return The inverse matrix.
     */
    Matrix<ValueType> qr_invert();

    virtual void fence(){;};

protected:
    /**
     * Convert 2D-index to flattened index in the range `[0,nlocal)`.
     * @param i The row index.
     * @param j The column index.
     * @return Flattened index in `[0,nlocal)`.
     */
    virtual int flatten(int i, int j);
    /**
     * Convert flat index to row and column index.
     * @param idx Flat index in the range `[0,nlocal)`.
     * @return Pair of row and column index.
     */
    virtual std::pair<int, int> unflatten(int idx);
};

#endif//DISTMATRIX_MATRIX_H
