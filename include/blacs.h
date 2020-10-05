//
// Created by anders on 10/5/20.
//

#ifndef DISTMATRIX_BLACS_H
#define DISTMATRIX_BLACS_H


class blacs {
public:
    static void initialize(int nprows=-1, int npcols=-1);
    static void finalize(int finalize_mpi = 1);

    inline static const char blacslayout = 'R';
    inline static int nprows, npcols, mpisize, mpirank, blacscontext;
    inline static int myprow, mypcol;
};


#endif//DISTMATRIX_BLACS_H
