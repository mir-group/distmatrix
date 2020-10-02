#ifndef DISTMATRIX_DISTMATRIX_H
#define DISTMATRIX_DISTMATRIX_H
#include <matrix.h>
#include <mpi.h>
#include <functional>

template<class ValueType>
class DistMatrix : public Matrix<ValueType>{
public:
    static void initialize(int nprows=-1, int npcols=-1);
    static void finalize(int finalize_mpi = 1);

    inline static const char blacslayout = 'R';
    inline static int nprows, npcols, mpisize, mpirank, blacscontext;
    // these are 1-based, everything else is 0-based
    inline static int myprow, mypcol;

    std::shared_ptr<MPI_Win> mpiwindow;

    int nbrows, nbcols;
    int nrowsperblock, ncolsperblock;
    int desc[9];

    DistMatrix(int ndistrows, int ndistcols, int nrowsperblock_=-1, int ncolsperblock_=-1);

    ValueType operator()(int i, int j) override;
    void set(int i, int j, ValueType x) override;

    MPI_Datatype mpitype;

    void fence() override;
    void barrier() override;

protected:
    std::pair<int,int> unflatten_local(int idx);
    std::pair<int,int> unflatten(int idx) override;
    std::pair<int,int> l2g(int ilocal, int jlocal);
    std::pair<int,int> g2l(int i, int j);
    std::pair<int,int> g2p(int i, int j);
    int flatten(int i, int j) override;
    bool islocal(int i, int j);
    std::pair<int,int> getlocalsizes(int ip, int jp);
};



#endif//DISTMATRIX_DISTMATRIX_H
