
#include <distmatrix.h>
#include <extern_blacs.h>
#include <iostream>
#include <mpi.h>
#include <blacs.h>

void delete_window(MPI_Win *window) {
    MPI_Win_free(window);
    delete window;
}



template<class ValueType>
DistMatrix<ValueType>::DistMatrix(int ndistrows, int ndistcols, int nrowsperblock_, int ncolsperblock_) : nrowsperblock(nrowsperblock_), ncolsperblock(ncolsperblock_),
                                                                                                          Matrix<ValueType>(ndistrows, ndistcols), mpiwindow(new MPI_Win, delete_window) {
    if (blacs::nprows > ndistrows || blacs::npcols > ndistcols) {
        throw std::logic_error("process grid is larger than matrix - TODO");
    }
    if (nrowsperblock < 1 || ncolsperblock < 1) {
        nrowsperblock = std::max(1, ndistrows / blacs::nprows);
        ncolsperblock = std::max(1, ndistcols / blacs::npcols);
    }
    int zero = 0;
    int info;
    std::tie(this->nlocalrows, this->nlocalcols) = getlocalsizes(blacs::myprow, blacs::mypcol);

    this->nlocal = (this->nlocalrows) * (this->nlocalcols);
    this->array = std::shared_ptr<ValueType[]>(new ValueType[this->nlocal]);
    descinit_(&desc[0], &ndistrows, &ndistcols, &nrowsperblock, &ncolsperblock, &zero, &zero, &blacs::blacscontext, &this->nlocalrows, &info);
    if (info != 0) {
        std::cerr << "info = " << info << std::endl;
        throw std::runtime_error("DESCINIT error");
    }
    MPI_Win_create((this->array).get(), this->nlocal * sizeof(ValueType), sizeof(ValueType), MPI_INFO_NULL, MPI_COMM_WORLD, mpiwindow.get());

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
ValueType DistMatrix<ValueType>::operator()(int i, int j) {
    if (islocal(i, j)) {
        return Matrix<ValueType>::operator()(i, j);
    } else {
        int remoteidx = flatten(i, j);
        auto [ip, jp] = g2p(i, j);
        int remoterank = ip * blacs::npcols + jp;
        ValueType result;
        MPI_Win_lock(MPI_LOCK_SHARED, remoterank, 0, *mpiwindow);
        MPI_Get(&result, 1, mpitype, remoterank, remoteidx, 1, mpitype, *mpiwindow);
        MPI_Win_unlock(remoterank, *mpiwindow);
        return result;
    }
}

template<class ValueType>
void DistMatrix<ValueType>::set(int i, int j, const ValueType x) {
    if (islocal(i, j)) {
        Matrix<ValueType>::set(i, j, x);
    } else {
        int remoteidx = flatten(i, j);
        auto [ip, jp] = g2p(i, j);
        int remoterank = ip * blacs::npcols + jp;
        auto [rr, rc] = getlocalsizes(ip, jp);
//        std::cout << ip << ", " << jp << ", " << rr << ", " << rc << ", " << remoterank << ", " << remoteidx << "\n";
        MPI_Win_lock(MPI_LOCK_EXCLUSIVE, remoterank, 0, *mpiwindow);
        MPI_Put(&x, 1, mpitype, remoterank, remoteidx, 1, mpitype, *mpiwindow);
        MPI_Win_unlock(remoterank, *mpiwindow);
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
    return {nlocalrows, nlocalcols};
}
template<class ValueType>
void DistMatrix<ValueType>::fence() {
    MPI_Win_fence(0, *mpiwindow);
//    MPI_Win_flush_all(*mpiwindow);
}
template<class ValueType>
void DistMatrix<ValueType>::barrier() {
    MPI_Barrier(MPI_COMM_WORLD);
}


template class DistMatrix<bool>;
template class DistMatrix<int>;
template class DistMatrix<float>;
template class DistMatrix<double>;
template class DistMatrix<std::complex<float>>;
template class DistMatrix<std::complex<double>>;