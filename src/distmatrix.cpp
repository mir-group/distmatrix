#include <blacs.h>
#include <cstring>
#include <distmatrix.h>
#include <exception>
#include <extern_blacs.h>
#include <iostream>
#include <mpi.h>

void check_info(int info, const std::string &name) {
    if (info) {
        std::cerr << "info = " << info << std::endl;
        throw std::runtime_error("error in " + name);
    }
};

void delete_window(MPI_Win *window) {
    MPI_Win_fence(0, *window);
    MPI_Win_free(window);
    delete window;
}

template<class ValueType>
DistMatrix<ValueType>::DistMatrix(int ndistrows, int ndistcols, int nrowsperblock, int ncolsperblock) : nrowsperblock(nrowsperblock), ncolsperblock(ncolsperblock),
                                                                                                        mpiwindow(new MPI_Win, delete_window) {
//    if (blacs::nprows > ndistrows || blacs::npcols > ndistcols) {
//        throw std::logic_error("process grid is larger than matrix - TODO");
//    }
    if (nrowsperblock < 1) {
        this->nrowsperblock = nrowsperblock = std::max(1, ndistrows / blacs::nprows / 4);
    }
    if (ncolsperblock < 1) {
        this->ncolsperblock = ncolsperblock = std::max(1, ndistcols / blacs::npcols / 4);
    }
    this->nrows = ndistrows;
    this->ncols = ndistcols;

    int zero = 0;
    int info;
    std::tie(this->nlocalrows, this->nlocalcols) = getlocalsizes(blacs::myprow, blacs::mypcol);

    this->nlocal = (this->nlocalrows) * (this->nlocalcols);
    this->array = std::shared_ptr<ValueType[]>(new ValueType[this->nlocal]);

    //printf("ip = %d, jp = %d, ndistrows = %d, ndistcols = %d, nlocalrows = %d\n",
    //     blacs::myprow, blacs::mypcol, ndistrows, ndistcols, this->nlocalrows);
    descinit_(&desc[0], &ndistrows, &ndistcols, &nrowsperblock, &ncolsperblock, &zero, &zero, &blacs::blacscontext, &this->nlocalrows, &info);
    if (info != 0) {
        std::cerr << "info = " << info << std::endl;
        throw std::runtime_error("DESCINIT error");
    }
    MPI_Win_create((this->array).get(), this->nlocal * sizeof(ValueType), sizeof(ValueType), MPI_INFO_NULL, MPI_COMM_WORLD, mpiwindow.get());
    fence();

    if (std::is_same_v<ValueType, bool>) {
        mpitype = MPI_CXX_BOOL;
    } else if (std::is_same_v<ValueType, int>) {
        mpitype = MPI_INT;
    } else if (std::is_same_v<ValueType, float>) {
        mpitype = MPI_FLOAT;
    } else if (std::is_same_v<ValueType, double>) {
        mpitype = MPI_DOUBLE;
    } else if (std::is_same_v<ValueType, std::complex<float>>) {
        mpitype = MPI_CXX_FLOAT_COMPLEX;
    } else if (std::is_same_v<ValueType, std::complex<double>>) {
        mpitype = MPI_CXX_DOUBLE_COMPLEX;
    } else {
        mpitype = MPI_DATATYPE_NULL;
    }
}

template<class ValueType>
ValueType DistMatrix<ValueType>::operator()(int i, int j, bool lock) {
    if (islocal(i, j)) {
        return Matrix<ValueType>::operator()(i, j);
    } else {
        int remoteidx = flatten(i, j);
        auto [ip, jp] = g2p(i, j);
        int remoterank = ip * blacs::npcols + jp;
        ValueType result;
        if (lock) {
            MPI_Win_lock(MPI_LOCK_SHARED, remoterank, 0, *mpiwindow);
            MPI_Get(&result, 1, mpitype, remoterank, remoteidx, 1, mpitype, *mpiwindow);
            MPI_Win_unlock(remoterank, *mpiwindow);
        } else {
            MPI_Request request;
            MPI_Rget(&result, 1, mpitype, remoterank, remoteidx, 1, mpitype, *mpiwindow, &request);
            MPI_Wait(&request, MPI_STATUS_IGNORE);
        }

        return result;
    }
}

template<class ValueType>
void DistMatrix<ValueType>::set(int i, int j, const ValueType x, bool lock) {
    if (islocal(i, j)) {
        Matrix<ValueType>::set(i, j, x);
    } else {
        int remoteidx = flatten(i, j);// something wrong with this, returns 0
        auto [ip, jp] = g2p(i, j);
        int remoterank = ip * blacs::npcols + jp;
        if (remoterank < 0) {
            printf(" i = %d, j = %d, mb = %d, nb = %d, ip = %d, jp = %d, npcols = %d, nprows = %d, rr = %d\n",
                   i, j, nrowsperblock, ncolsperblock, ip, jp, blacs::npcols, blacs::nprows, remoterank);
        }
        if (lock) MPI_Win_lock(MPI_LOCK_EXCLUSIVE, remoterank, 0, *mpiwindow);
        MPI_Put(&x, 1, mpitype, remoterank, remoteidx, 1, mpitype, *mpiwindow);
        if (lock) MPI_Win_unlock(remoterank, *mpiwindow);
    }
}

template<class ValueType>
std::pair<int, int> DistMatrix<ValueType>::unflatten_local(int idx) {
    return Matrix<ValueType>::unflatten(idx);
}
template<class ValueType>
std::pair<int, int> DistMatrix<ValueType>::unflatten(int idx) {
    auto [ilocal, jlocal] = unflatten_local(idx);
    return l2g(ilocal, jlocal);
}
template<class ValueType>
std::pair<int, int> DistMatrix<ValueType>::l2g(int ilocal, int jlocal) {
    ilocal++;
    jlocal++;
    int zero = 0;
    int i = indxl2g_(&ilocal, &nrowsperblock, &blacs::myprow, &zero, &blacs::nprows) - 1;
    int j = indxl2g_(&jlocal, &ncolsperblock, &blacs::mypcol, &zero, &blacs::npcols) - 1;
    return {i, j};
}
template<class ValueType>
std::pair<int, int> DistMatrix<ValueType>::g2l(int i, int j) {
    i++;
    j++;
    int zero = 0;
    int ilocal = indxg2l_(&i, &nrowsperblock, &blacs::myprow, &zero, &blacs::nprows) - 1;
    int jlocal = indxg2l_(&j, &ncolsperblock, &blacs::mypcol, &zero, &blacs::npcols) - 1;
    return {ilocal, jlocal};
}

template<class ValueType>
std::pair<int, int> DistMatrix<ValueType>::g2p(int i, int j) {
    i++;
    j++;
    int zero = 0;
    int ip = indxg2p_(&i, &nrowsperblock, &blacs::myprow, &zero, &blacs::nprows);
    int jp = indxg2p_(&j, &ncolsperblock, &blacs::mypcol, &zero, &blacs::npcols);
    return {ip, jp};
}
template<class ValueType>
bool DistMatrix<ValueType>::islocal(int i, int j) {
    auto [ip, jp] = g2p(i, j);
    return ip == blacs::myprow && jp == blacs::mypcol;
}
template<class ValueType>
int DistMatrix<ValueType>::flatten(int i, int j) {
    if (islocal(i, j)) {
        auto [ilocal, jlocal] = g2l(i, j);
        return Matrix<ValueType>::flatten(ilocal, jlocal);
    }
    auto [ip, jp] = g2p(i, j);
    auto [ilocal, jlocal] = g2l(i, j);
    auto [remoterows, remotecols] = getlocalsizes(ip, jp);
    return jlocal * remoterows + ilocal;
}
template<class ValueType>
std::pair<int, int> DistMatrix<ValueType>::getlocalsizes(int ip, int jp) {
    int zero = 0;
    int nlocalrows = numroc_(&(this->nrows), &nrowsperblock, &ip, &zero, &blacs::nprows);
    int nlocalcols = numroc_(&(this->ncols), &ncolsperblock, &jp, &zero, &blacs::npcols);
    //printf("ip = %d, jp = %d, nrows = %d, ncols = %d, nrowsperblock = %d, ncolsperblock = %d, nprows = %d, npcols = %d, nlocalrows = %d, nlocalcols = %d\n",
    //     ip, jp, this->nrows, this->ncols, nrowsperblock, ncolsperblock, blacs::nprows, blacs::npcols, nlocalrows, nlocalcols);
    return {nlocalrows, nlocalcols};
}

template<class ValueType>
bool DistMatrix<ValueType>::operator==(const ValueType x) {
    bool local_equal = Matrix<ValueType>::operator==(x);
    bool result;
    MPI_Allreduce(&local_equal, &result, 1, MPI_CXX_BOOL, MPI_LAND, MPI_COMM_WORLD);
    return result;
}
template<class ValueType>
bool DistMatrix<ValueType>::operator==(const Matrix<ValueType> &x) {
    bool local_equal = Matrix<ValueType>::operator==(x);
    bool result;
    MPI_Allreduce(&local_equal, &result, 1, MPI_CXX_BOOL, MPI_LAND, MPI_COMM_WORLD);
    return result;
}
template<class ValueType>
ValueType DistMatrix<ValueType>::sum() {
    ValueType local_sum = Matrix<ValueType>::sum();
    ValueType result;
    MPI_Allreduce(&local_sum, &result, 1, mpitype, MPI_SUM, MPI_COMM_WORLD);
    return result;
}
template<class ValueType>
DistMatrix<ValueType> DistMatrix<ValueType>::matmul(const DistMatrix<ValueType> &B, const ValueType alpha, const char transA, const char transB) {
    int m, k, n, l;
    int Crowsperblock, Ccolsperblock;
    if (transA == 'N' || transA == 'n') {
        m = this->nrows;
        k = this->ncols;
        Crowsperblock = this->nrowsperblock;
    } else {
        m = this->ncols;
        k = this->nrows;
        Crowsperblock = this->ncolsperblock;
    }
    if (transB == 'N' || transB == 'n') {
        n = B.ncols;
        l = B.nrows;
        Ccolsperblock = B.ncolsperblock;
    } else {
        n = B.nrows;
        l = B.ncols;
        Ccolsperblock = B.nrowsperblock;
    }

    // Detect shape mismatch of the two matrices
    if (k != l) {
        std::stringstream errmsg;
        errmsg << "A(" << this->nrows << ", " << this->ncols << ") does not match B(" << B.nrows << ", " << B.ncols << ")";
        throw std::logic_error(errmsg.str());
    }

    DistMatrix<ValueType> C(m, n, Crowsperblock, Ccolsperblock);
    ValueType beta(0);
    ValueType *A_ptr = this->array.get(), *B_ptr = B.array.get(), *C_ptr = C.array.get();
    int one = 1;
    if constexpr (std::is_same_v<ValueType, float>) {
        psgemm_(&transA, &transB, &m, &n, &k, &alpha, A_ptr, &one, &one, &desc[0],
                B_ptr, &one, &one, &(B.desc[0]), &beta,
                C_ptr, &one, &one, &(C.desc[0]));
    } else if constexpr (std::is_same_v<ValueType, double>) {
        pdgemm_(&transA, &transB, &m, &n, &k, &alpha, A_ptr, &one, &one, &desc[0],
                B_ptr, &one, &one, &(B.desc[0]), &beta,
                C_ptr, &one, &one, &(C.desc[0]));

    } else if constexpr (std::is_same_v<ValueType, std::complex<float>>) {
        pcgemm_(&transA, &transB, &m, &n, &k, &alpha, A_ptr, &one, &one, &desc[0],
                B_ptr, &one, &one, &(B.desc[0]), &beta,
                C_ptr, &one, &one, &(C.desc[0]));

    } else if constexpr (std::is_same_v<ValueType, std::complex<double>>) {
        pzgemm_(&transA, &transB, &m, &n, &k, &alpha, A_ptr, &one, &one, &desc[0],
                B_ptr, &one, &one, &(B.desc[0]), &beta,
                C_ptr, &one, &one, &(C.desc[0]));
    } else {
        throw std::logic_error("matmul called with unsupported type");
    }
    return C;
}
template<class ValueType>
void DistMatrix<ValueType>::fence() {
    MPI_Win_fence(0, *mpiwindow);
    blacs::barrier();
}
template<class ValueType>
std::tuple<DistMatrix<ValueType>, std::vector<ValueType>> DistMatrix<ValueType>::qr() {
    /*
     * Upper triangular part of QR will contain R, lower triangular
     * will be Q as represented by Householder reflections.
     */

    if (this->nrows < this->ncols) {
        throw std::runtime_error("The matrix for QR should have nrows >= ncols !");
    }

    blacs::barrier();
    printf("creating QR\n");
    DistMatrix<ValueType> QR(this->nrows, this->ncols, this->nrowsperblock, this->ncolsperblock);
    blacs::barrier();
    printf("barrier\n");
    this->copy_to(QR);
    printf("copied\n");
    std::vector<ValueType> tau(this->nlocalrows);

    int info, lwork = -1, one = 1;
    ValueType worktmp;
    int m = this->nrows, n = this->ncols;
    char U = 'U', N = 'N', R = 'R', T = 'T';

    if constexpr (std::is_same_v<ValueType, float>) {
        psgeqrf_(&m, &n, QR.array.get(), &one, &one, &(QR.desc[0]), tau.data(), &worktmp, &lwork, &info);
        check_info(info, "geqrf work query");
        lwork = worktmp;
        std::vector<ValueType> work(lwork);
        psgeqrf_(&m, &n, QR.array.get(), &one, &one, &(QR.desc[0]), tau.data(), work.data(), &lwork, &info);
        check_info(info, "geqrf");
    } else if constexpr (std::is_same_v<ValueType, double>) {
        pdgeqrf_(&m, &n, QR.array.get(), &one, &one, &(QR.desc[0]), tau.data(), &worktmp, &lwork, &info);
        check_info(info, "geqrf work query");
        lwork = worktmp;
        std::vector<ValueType> work(lwork);
        pdgeqrf_(&m, &n, QR.array.get(), &one, &one, &(QR.desc[0]), tau.data(), work.data(), &lwork, &info);
        check_info(info, "geqrf");
    } else {
        throw std::logic_error("qr called with unsupported type!");
    }

    return std::make_tuple(QR, tau);
}
template<class ValueType>
DistMatrix<ValueType> DistMatrix<ValueType>::Q_matmul(DistMatrix<ValueType> &b, std::vector<ValueType> &tau, char sideQ, char transQ) {
    int l = this->nrows;
    int k = this->ncols;
    int m = b.nrows;
    int n = b.ncols;
    int r, c;

    // Detect shape mismatch of the two matrices
    bool mismatch = false;
    if (sideQ == 'L' && transQ == 'N') {
        r = l;
        c = n;
        if (m != k) mismatch = true;
    } else if (sideQ == 'L' && transQ == 'T') {
        r = k;
        c = n; 
        if (m != l) mismatch = true; 
    } else if (sideQ == 'R' && transQ == 'N') {
        r = m;
        c = k;
        if (n != l) mismatch = true; 
    } else if (sideQ == 'R' && transQ == 'T') {
        r = m;
        c = l;
        if (n != k) mismatch = true; 
    }

    // Detect shape mismatch of the two matrices
    if (mismatch) {
        std::stringstream errmsg;
        errmsg << "Q(" << this->nrows << ", " << this->ncols << ") does not match b(" << b.nrows << ", " << b.ncols << ")";
        throw std::logic_error(errmsg.str());
    }

    DistMatrix<ValueType> Q_b(m, n, b.nrowsperblock, b.ncolsperblock);
    blacs::barrier();
    b.copy_to(Q_b);

    int info, lwork = -1, one = 1;
    ValueType worktmp;

    if constexpr (std::is_same_v<ValueType, float>) {
        lwork = -1;
        psormqr_(&sideQ, &transQ, &m, &n, &k, (this->array).get(), &one, &one, &(this->desc[0]), tau.data(), Q_b.array.get(), &one, &one, &(Q_b.desc[0]), &worktmp, &lwork, &info);
        check_info(info, "ormqr work query");
        lwork = worktmp;
        std::vector<ValueType> work(lwork);
        work.resize(lwork);
        psormqr_(&sideQ, &transQ, &m, &n, &k, (this->array).get(), &one, &one, &(this->desc[0]), tau.data(), Q_b.array.get(), &one, &one, &(Q_b.desc[0]), work.data(), &lwork, &info);
        check_info(info, "ormqr");
    } else if constexpr (std::is_same_v<ValueType, double>) {
        lwork = -1;
        pdormqr_(&sideQ, &transQ, &m, &n, &k, (this->array).get(), &one, &one, &(this->desc[0]), tau.data(), Q_b.array.get(), &one, &one, &(Q_b.desc[0]), &worktmp, &lwork, &info);
        check_info(info, "ormqr work query");
        lwork = worktmp;
        std::vector<ValueType> work(lwork);
        work.resize(lwork);
        pdormqr_(&sideQ, &transQ, &m, &n, &k, (this->array).get(), &one, &one, &(this->desc[0]), tau.data(), Q_b.array.get(), &one, &one, &(Q_b.desc[0]), work.data(), &lwork, &info);
        check_info(info, "ormqr");
    } else {
        throw std::logic_error("Q_matmul called with unsupported type!");
    }

    // change shape of Q_b, it originally has the same shape as b
    if (Q_b.nrows == r && Q_b.ncols == c) {
        return Q_b;
    } else {
        DistMatrix<ValueType> C(r, c);

        // TODO: simplify it using "collect"
        int syscontext, ptrcontext, bigcontext;
        int zero = 0;
        blacs_get_(&zero, &zero, &syscontext);
        bigcontext = syscontext;
        blacs_gridinit_(&bigcontext, &blacs::blacslayout, &blacs::nprows, &blacs::npcols);

        if constexpr (std::is_same_v<ValueType, float>) {
            psgemr2d_(&r, &c, Q_b.array.get(), &one, &one, &(Q_b.desc[0]),
                      C.array.get(), &one, &one, &(C.desc[0]), &bigcontext);
        } else if constexpr (std::is_same_v<ValueType, double>) {
            pdgemr2d_(&r, &c, Q_b.array.get(), &one, &one, &(Q_b.desc[0]),
                      C.array.get(), &one, &one, &(C.desc[0]), &bigcontext);
        }

        return C;
    }
}

template<class ValueType>
DistMatrix<ValueType> DistMatrix<ValueType>::qr_invert() {
    if (this->nrows != this->ncols) {
        throw std::runtime_error("The matrix for QR should have nrows == ncols !");
    }

    blacs::barrier();
    printf("creating QR\n");
    DistMatrix<ValueType> QR(this->nrows, this->ncols, this->nrowsperblock, this->ncolsperblock);
    blacs::barrier();
    printf("creating Ainv\n");
    blacs::barrier();
    //DistMatrix<ValueType> Ainv(this->nrows, this->ncols, this->nrowsperblock, this->ncolsperblock);
    DistMatrix<ValueType> Ainv(this->ncols, this->nrows, this->ncolsperblock, this->nrowsperblock);
    blacs::barrier();
    printf("copying A to QR\n");
    this->copy_to(QR);
    std::vector<ValueType> tau(this->nlocalrows);

    int info, lwork = -1, one = 1;
    ValueType worktmp;
    int m = this->nrows, n = this->ncols;
    char U = 'U', N = 'N', R = 'R', T = 'T';

    if constexpr (std::is_same_v<ValueType, float>) {
        /*
         * Upper triangular part of QR will contain R, lower triangular
         * will be Q as represented by Householder reflections.
         */
        psgeqrf_(&m, &n, QR.array.get(), &one, &one, &(QR.desc[0]), tau.data(), &worktmp, &lwork, &info);
        check_info(info, "geqrf work query");
        lwork = worktmp;
        std::vector<ValueType> work(lwork);
        psgeqrf_(&m, &n, QR.array.get(), &one, &one, &(QR.desc[0]), tau.data(), work.data(), &lwork, &info);
        check_info(info, "geqrf");
        QR.copy_to(Ainv);

        /*
         * Upper triangular part of Ainv will be the inverse of R.
         */
        //pstrtri_(&U, &N, &n, Ainv.array.get(), &one, &one, &(Ainv.desc[0]), &info);
        pstrtri_(&U, &N, &m, Ainv.array.get(), &one, &one, &(Ainv.desc[0]), &info);
        check_info(info, "trtri");

        /*
         * Set lower triangular part of Ainv to zero.
         */
        Ainv = [](ValueType Ainvij, int i, int j) {
            return i > j ? 0 : Ainvij;
        };

        /*
         * Calculate A^-1 = R^-1 Q^T
         */
        lwork = -1;
        psormqr_(&R, &T, &m, &n, &m, QR.array.get(), &one, &one, &(QR.desc[0]), tau.data(), Ainv.array.get(), &one, &one, &(Ainv.desc[0]), &worktmp, &lwork, &info);
        check_info(info, "ormqr work query");
        lwork = worktmp;
        work.resize(lwork);
        psormqr_(&R, &T, &m, &n, &m, QR.array.get(), &one, &one, &(QR.desc[0]), tau.data(), Ainv.array.get(), &one, &one, &(Ainv.desc[0]), work.data(), &lwork, &info);
        check_info(info, "ormqr");
    } else if constexpr (std::is_same_v<ValueType, double>) {
        printf("begin pdgeqrf\n");
        pdgeqrf_(&m, &n, QR.array.get(), &one, &one, &(QR.desc[0]), tau.data(), &worktmp, &lwork, &info);
        check_info(info, "geqrf work query");
        lwork = worktmp;
        std::vector<ValueType> work(lwork);
        pdgeqrf_(&m, &n, QR.array.get(), &one, &one, &(QR.desc[0]), tau.data(), work.data(), &lwork, &info);
        check_info(info, "geqrf");
        QR.copy_to(Ainv);
        printf("Done pdgeqrf\n");

        //pdtrtri_(&U, &N, &n, Ainv.array.get(), &one, &one, &(Ainv.desc[0]), &info);
        pdtrtri_(&U, &N, &m, Ainv.array.get(), &one, &one, &(Ainv.desc[0]), &info);
        check_info(info, "trtri");
        printf("Done invert tri\n");

        /*
         * Set lower triangular part of Ainv to zero.
         */
        Ainv = [](ValueType Ainvij, int i, int j) {
            return i > j ? 0 : Ainvij;
        };
        printf("set lower tri of A\n");

        lwork = -1;
        pdormqr_(&R, &T, &m, &n, &m, QR.array.get(), &one, &one, &(QR.desc[0]), tau.data(), Ainv.array.get(), &one, &one, &(Ainv.desc[0]), &worktmp, &lwork, &info);
        check_info(info, "ormqr work query");
        lwork = worktmp;
        work.resize(lwork);
        pdormqr_(&R, &T, &m, &n, &m, QR.array.get(), &one, &one, &(QR.desc[0]), tau.data(), Ainv.array.get(), &one, &one, &(Ainv.desc[0]), work.data(), &lwork, &info);
        check_info(info, "ormqr");
        printf("Done pdormqr\n");
    } else {
        throw std::logic_error("qr_invert called with unsupported type!");
    }

    return Ainv;
}
template<class ValueType>
DistMatrix<ValueType> DistMatrix<ValueType>::cholesky(const char uplo) {
    DistMatrix<ValueType> LU(this->nrows, this->ncols, this->nrowsperblock, this->ncolsperblock);
    this->copy_to(LU);
    int info;
    int one = 1;
    if constexpr (std::is_same_v<ValueType, float>) {
        pspotrf_(&uplo, &(LU.nrows), LU.array.get(), &one, &one, &(LU.desc[0]), &info);
    } else if constexpr (std::is_same_v<ValueType, double>) {
        pdpotrf_(&uplo, &(LU.nrows), LU.array.get(), &one, &one, &(LU.desc[0]), &info);
    } else if constexpr (std::is_same_v<ValueType, std::complex<float>>) {
        pcpotrf_(&uplo, &(LU.nrows), LU.array.get(), &one, &one, &(LU.desc[0]), &info);
    } else if constexpr (std::is_same_v<ValueType, std::complex<double>>) {
        pzpotrf_(&uplo, &(LU.nrows), LU.array.get(), &one, &one, &(LU.desc[0]), &info);
    } else {
        throw std::logic_error("cholesky called with unsupported type!");
    }
    check_info(info, "potrf");
    LU = [&uplo](ValueType LUij, int i, int j) {
        if ((uplo == 'U' && j >= i) || (uplo == 'L' && i >= j)) {
            return LUij;
        }
        return ValueType(0);
    };
    return LU;
}
template<class ValueType>
DistMatrix<ValueType> DistMatrix<ValueType>::triangular_invert(const char uplo, const char unit_triangular) {
    DistMatrix<ValueType> LUinv(this->nrows, this->ncols, this->nrowsperblock, this->ncolsperblock);
    this->copy_to(LUinv);
    int info;
    int one = 1;
    if constexpr (std::is_same_v<ValueType, float>) {
        pstrtri_(&uplo, &unit_triangular, &(LUinv.nrows), LUinv.array.get(), &one, &one, &(LUinv.desc[0]), &info);
    } else if constexpr (std::is_same_v<ValueType, double>) {
        pdtrtri_(&uplo, &unit_triangular, &(LUinv.nrows), LUinv.array.get(), &one, &one, &(LUinv.desc[0]), &info);
    } else if constexpr (std::is_same_v<ValueType, std::complex<float>>) {
        pctrtri_(&uplo, &unit_triangular, &(LUinv.nrows), LUinv.array.get(), &one, &one, &(LUinv.desc[0]), &info);
    } else if constexpr (std::is_same_v<ValueType, std::complex<double>>) {
        pztrtri_(&uplo, &unit_triangular, &(LUinv.nrows), LUinv.array.get(), &one, &one, &(LUinv.desc[0]), &info);
    } else {
        throw std::logic_error("triangular_invert called with unsupported type!");
    }
    check_info(info, "trtri");

    return LUinv;
}

template<class ValueType>
void DistMatrix<ValueType>::scatter(ValueType *ptr, int i0, int j0, int p, int q) {
    int syscontext, allcontext, serialcontext, bigcontext;
    int nprows, npcols, myprow, mypcol;
    int nproc = blacs::nprows * blacs::npcols;

    int info, what = -1, one = 1, zero = 0, m = this->nrows, n = this->ncols;
    int serialdesc[9];
    int i = i0 + 1; // p?gemr2d_ starts from 1
    int j = j0 + 1;

    // get the system default context
    blacs_get_(&zero, &zero, &syscontext);

    bigcontext = syscontext;
    blacs_gridinit_(&bigcontext, &blacs::blacslayout, &blacs::nprows, &blacs::npcols);

    serialcontext = syscontext;
    blacs_gridinit_(&serialcontext, &blacs::blacslayout, &one, &one);

    if (blacs::mpirank == 0) {
        descinit_(&serialdesc[0], &p, &q, &p, &q, &zero, &zero, &serialcontext, &p, &info);
        check_info(info, "descinit scatter");
    }
    MPI_Bcast(&serialdesc, 9, MPI_INT, 0, MPI_COMM_WORLD);

    if constexpr (std::is_same_v<ValueType, float>) {
        psgemr2d_(&p, &q, ptr, &one, &one, &serialdesc[0],
                  this->array.get(), &i, &j, &desc[0], &bigcontext);
    } else if constexpr (std::is_same_v<ValueType, double>) {
        pdgemr2d_(&p, &q, ptr, &one, &one, &serialdesc[0],
                  this->array.get(), &i, &j, &desc[0], &bigcontext);
    } else if constexpr (std::is_same_v<ValueType, std::complex<float>>) {
        pcgemr2d_(&p, &q, ptr, &one, &one, &serialdesc[0],
                  this->array.get(), &i, &j, &desc[0], &bigcontext);
    } else if constexpr (std::is_same_v<ValueType, std::complex<double>>) {
        pzgemr2d_(&p, &q, ptr, &one, &one, &serialdesc[0],
                  this->array.get(), &i, &j, &desc[0], &bigcontext);
    } else if constexpr (std::is_same_v<ValueType, int>) {
        pigemr2d_(&p, &q, ptr, &one, &one, &serialdesc[0],
                  this->array.get(), &i, &j, &desc[0], &bigcontext);
    } else {
        throw std::logic_error("scatter called with unsupported type");
    }
    blacs::barrier();
    if (blacs::mpirank == 0) {
        blacs_gridexit_(&serialcontext);
    }
}


template<class ValueType>
void DistMatrix<ValueType>::collect(ValueType *ptr, int i0, int j0, int p, int q, int mb, int nb, int lld) {
    int syscontext, ptrcontext, bigcontext;
    int nprows, npcols, myprow, mypcol;
    int nproc = blacs::nprows * blacs::npcols;

    int info, what = -1, one = 1, zero = 0, m = this->nrows, n = this->ncols;
    int ptrdesc[9];
    int i = i0 + 1; // p?gemr2d_ starts from 1
    int j = j0 + 1;

    // get the system default context
    blacs_get_(&zero, &zero, &syscontext);

    bigcontext = syscontext;
    blacs_gridinit_(&bigcontext, &blacs::blacslayout, &blacs::nprows, &blacs::npcols);

    ptrcontext = syscontext;
    int ptr_nprow, ptr_npcol;
    if (p % mb == 0) {
        ptr_nprow = p / mb;
    } else {
        ptr_nprow = p / mb + 1;
    }
    if (q % nb == 0) {
        ptr_npcol = q / nb;
    } else {
        ptr_npcol = q / nb + 1;
    }
    blacs_gridinit_(&ptrcontext, &blacs::blacslayout, &ptr_nprow, &ptr_npcol);

    descinit_(&ptrdesc[0], &p, &q, &mb, &nb, &zero, &zero, &ptrcontext, &lld, &info);
    check_info(info, "descinit scatter");

    if constexpr (std::is_same_v<ValueType, float>) {
        psgemr2d_(&p, &q, ptr, &one, &one, &ptrdesc[0],
                  this->array.get(), &i, &j, &desc[0], &bigcontext);
    } else if constexpr (std::is_same_v<ValueType, double>) {
        pdgemr2d_(&p, &q, ptr, &one, &one, &ptrdesc[0],
                  this->array.get(), &i, &j, &desc[0], &bigcontext);
    } else if constexpr (std::is_same_v<ValueType, std::complex<float>>) {
        pcgemr2d_(&p, &q, ptr, &one, &one, &ptrdesc[0],
                  this->array.get(), &i, &j, &desc[0], &bigcontext);
    } else if constexpr (std::is_same_v<ValueType, std::complex<double>>) {
        pzgemr2d_(&p, &q, ptr, &one, &one, &ptrdesc[0],
                  this->array.get(), &i, &j, &desc[0], &bigcontext);
    } else if constexpr (std::is_same_v<ValueType, int>) {
        pigemr2d_(&p, &q, ptr, &one, &one, &ptrdesc[0],
                  this->array.get(), &i, &j, &desc[0], &bigcontext);
    } else {
        throw std::logic_error("collect called with unsupported type");
    }
    blacs::barrier();
    blacs_gridexit_(&ptrcontext);
}


template<class ValueType>
void DistMatrix<ValueType>::gather(ValueType *ptr) {

    int syscontext, allcontext, serialcontext, bigcontext;
    int nprows, npcols, myprow, mypcol;
    int nproc = blacs::nprows * blacs::npcols;

    int info, what = -1, one = 1, zero = 0, m = this->nrows, n = this->ncols;
    int serialdesc[9];

    // get the system default context
    blacs_get_(&zero, &zero, &syscontext);

    bigcontext = syscontext;
    blacs_gridinit_(&bigcontext, &blacs::blacslayout, &blacs::nprows, &blacs::npcols);

    serialcontext = syscontext;
    blacs_gridinit_(&serialcontext, &blacs::blacslayout, &one, &one);

    if (blacs::mpirank == 0) {
        descinit_(&serialdesc[0], &m, &n, &m, &n, &zero, &zero, &serialcontext, &m, &info);
        check_info(info, "descinit gather");
    }
    MPI_Bcast(&serialdesc, 9, MPI_INT, 0, MPI_COMM_WORLD);

    if constexpr (std::is_same_v<ValueType, float>) {
        psgemr2d_(&m, &n, this->array.get(), &one, &one, &desc[0],
                  ptr, &one, &one, &serialdesc[0], &bigcontext);
    } else if constexpr (std::is_same_v<ValueType, double>) {
        pdgemr2d_(&m, &n, this->array.get(), &one, &one, &desc[0],
                  ptr, &one, &one, &serialdesc[0], &bigcontext);
    } else if constexpr (std::is_same_v<ValueType, std::complex<float>>) {
        pcgemr2d_(&m, &n, this->array.get(), &one, &one, &desc[0],
                  ptr, &one, &one, &serialdesc[0], &bigcontext);
    } else if constexpr (std::is_same_v<ValueType, std::complex<double>>) {
        pzgemr2d_(&m, &n, this->array.get(), &one, &one, &desc[0],
                  ptr, &one, &one, &serialdesc[0], &bigcontext);
    } else if constexpr (std::is_same_v<ValueType, int>) {
        pigemr2d_(&m, &n, this->array.get(), &one, &one, &desc[0],
                  ptr, &one, &one, &serialdesc[0], &bigcontext);
    } else {
        throw std::logic_error("gather called with unsupported type");
    }
    blacs::barrier();
    if (blacs::mpirank == 0) {
        blacs_gridexit_(&serialcontext);
    }
}

template<class ValueType>
void DistMatrix<ValueType>::allgather(ValueType *ptr) {
    gather(ptr);
    MPI_Bcast(ptr, this->nrows * this->ncols, mpitype, 0, MPI_COMM_WORLD);
}

template<class ValueType>
void DistMatrix<ValueType>::triangular_solve(DistMatrix<ValueType> b, const char uplo, const char transA, const char unit_triangular) {
    int incx = 1;
    int zero = 0;
    int one = 1;
    int n = this->nrows;
    int info;

    if constexpr (std::is_same_v<ValueType, float>) {
        pstrsv_(&uplo, &transA, &unit_triangular, &n, this->array.get(), &one, &one, &desc[0], b.array.get(), &one, &one, &(b.desc[0]), &incx);
    } else if constexpr (std::is_same_v<ValueType, double>) {
        pdtrsv_(&uplo, &transA, &unit_triangular, &n, this->array.get(), &one, &one, &desc[0], b.array.get(), &one, &one, &(b.desc[0]), &incx);
    } else if constexpr (std::is_same_v<ValueType, std::complex<float>>) {
        pctrsv_(&uplo, &transA, &unit_triangular, &n, this->array.get(), &one, &one, &desc[0], b.array.get(), &one, &one, &(b.desc[0]), &incx);
    } else if constexpr (std::is_same_v<ValueType, std::complex<double>>) {
        pztrsv_(&uplo, &transA, &unit_triangular, &n, this->array.get(), &one, &one, &desc[0], b.array.get(), &one, &one, &(b.desc[0]), &incx);
    } else {
        throw std::logic_error("triangular_invert called with unsupported type!");
    }
}


template class DistMatrix<bool>;
template class DistMatrix<int>;
template class DistMatrix<float>;
template class DistMatrix<double>;
template class DistMatrix<std::complex<float>>;
template class DistMatrix<std::complex<double>>;
