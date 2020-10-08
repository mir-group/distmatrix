#ifndef DISTMATRIX_DISTMATRIX_H
#define DISTMATRIX_DISTMATRIX_H
#include <functional>
#include <matrix.h>
#include <mpi.h>

/**
 * Distributed matrix class.
 * The data is distributed in a block-cyclic fashion via ScaLAPACK/BLACS calls.
 * MPI Remote Memory Access allow for seamless setting and getting of elements on other MPI ranks.
 * All functionality from Matrix should be available, with the exception of the non-symmetric diagonalization.
 * @tparam ValueType The type of values to be stored, e.g. `int`, `float`, `std::complex<double>`.
 */
template<class ValueType>
class DistMatrix : public Matrix<ValueType> {
public:
    std::shared_ptr<MPI_Win> mpiwindow;///< The MPI window used for remote memory access.

    int nrowsperblock;///< The number of rows per block.
    int ncolsperblock;///< The number of columns per block.

    int desc[9];///< The ScaLAPACK matrix descriptor.

    /**
     * Construct a distributed matrix.
     * If `nrowsperblock` or `ncolsperblock` is less than 1, we guess a semi-sensible value.
     * @param ndistrows The total number of rows.
     * @param ndistcols The total number of columns.
     * @param nrowsperblock The number of rows per block (*not* per processor, see block-cyclic distribution).
     * @param ncolsperblock The number of columns per block (*not* per processor).
     */
    DistMatrix(int ndistrows, int ndistcols, int nrowsperblock = -1, int ncolsperblock = -1);

    /**
     * Get an element from the matrix.
     * If it is not a local element, `MPI_Win_(un)lock` and `MPI_Get` will be used.
     * @param i The row index.
     * @param j The column index.
     * @return A(i,j), whether local or remote.
     */
    ValueType operator()(int i, int j) override;

    /**
     * Set an element of the matrix.
     * If it is not a local element, `MPI_Win_(un)lock` and `MPI_Put` will be used.
     * @param i The row index.
     * @param j The column index.
     * @param x The new value of A(i,j).
     */
    void set(int i, int j, ValueType x) override;

    /**
     * The MPI_Datatype, e.g. `MPI_CXX_BOOL`, `MPI_DOUBLE`, `MPI_COMPLEX_FLOAT`.
     * We've programmed the common ones, add more to the constructor if desired.
     */
    MPI_Datatype mpitype;

    /**
     * Determine if an element resides on this rank.
     * @param i The row index.
     * @param j The column index.
     * @return `true` if `A(i,j)` resides on this MPI rank, otherwise `false`.
     */
    bool islocal(int i, int j);

    // assignment operator isn't inherited by default;
    using Matrix<ValueType>::operator=;

    /**
     * Check if all elements are exactly equal to `x`.
     * Must be called on all processors.
     * @return `true` if all elements are equal to `x`, otherwise `false`.
     */
    bool operator==(const ValueType x) override;
    /**
     * Check if the matrices are element-wise equal.
     * Must be called on all processors.
     * @return `true` if all elements are equal, otherwise `false`.
     */
    bool operator==(const Matrix<ValueType> &x) override;

    /**
     * Sum all the elements.
     * Must be called on all processors.
     * @return The sum.
     */
    ValueType sum() override;

    /**
     * Matrix product, `C = alpha*A*B`.
     * @param B The matrix with which to multiply.
     * @param alpha The scalar prefactor. Defaults to 1.
     * @return The matrix `C`.
     */
    DistMatrix<ValueType> matmul(const DistMatrix<ValueType> &B, const ValueType alpha = ValueType(1));

protected:
    /**
     * Convert flat local index to local row and column index.
     * @param idx Flat index in the range `[0,nlocal)`.
     * @return Pair of row and column index in ranges `[0,nlocalrows/cols)`.
     */
    std::pair<int, int> unflatten_local(int idx);
    /**
     * Convert flat local index to global row and column index.
     * @param idx Flat index in the range `[0,nlocal)`.
     * @return Pair of row and column index in ranges `[0,nrows/cols)`.
     */
    std::pair<int, int> unflatten(int idx) override;
    /**
     * Convert local indices to global indices.
     * @param ilocal Local row index, 0<=ilocal<nlocalrows.
     * @param jlocal Local column index, 0<=jlocal<nlocalcols.
     * @return Global indices in ranges `[0,nrows/cols)`.
     */
    std::pair<int, int> l2g(int ilocal, int jlocal);
    /**
     * Convert global indices to local indices.
     * @param i Global row index, 0<=i<nrows.
     * @param j Global column index, 0<=j<ncols.
     * @return Local indices in ranges `[0,nlocalrows/cols)`.
     */
    std::pair<int, int> g2l(int i, int j);
    /**
     * Find the indices of the processor on which a global element resides.
     * @param i Global row index, 0<=i<nrows.
     * @param j Global column index, 0<=j<ncols.
     * @return Processor indices in ranges `[0,blacs::nprows/cols)`.
     */
    std::pair<int, int> g2p(int i, int j);
    /**
     * Find local, flat index of an element, no matter where it resides.
     * @param i Global row index, 0<=i<nrows.
     * @param j Global column index, 0<=j<ncols.
     * @return The flattened index, i.e. the offset from `array[0]`, on whatever process has element i,j.
     */
    int flatten(int i, int j) override;
    /**
     * Get the size of the local matrix on a process.
     * @param ip The processor row index.
     * @param jp The processor column index.
     * @return The size of the local matrix stored on processor ip,jp.
     */
    std::pair<int, int> getlocalsizes(int ip, int jp);
};


#endif//DISTMATRIX_DISTMATRIX_H
