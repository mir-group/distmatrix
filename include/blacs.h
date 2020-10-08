//
// Created by anders on 10/5/20.
//

#ifndef DISTMATRIX_BLACS_H
#define DISTMATRIX_BLACS_H

/**
 * Helper class for BLACS and MPI functionality.
 */
class blacs {
public:
    /**
     * Initialize the BLACS grid. If `nprows` or `npcols` is less than 1,
     * `MPI_Dims_create` is used instead.
     *
     * @param nprows The number of rows in the process grid.
     * @param npcols The number of columns in the process grid.
     */
    static void initialize(int nprows = -1, int npcols = -1);

    /**
     * Finalize the BLACS grid and, optionally, MPI.
     * @param finalize_mpi
     */
    static void finalize(int finalize_mpi = 1);

    inline static const char blacslayout = 'R';///< The processor ordering.
    inline static int blacscontext;                   ///< The BLACS context obtained from initialization.

    inline static int nprows;///< The number of processor rows.
    inline static int npcols;///< The number of processor columns.
    inline static int myprow;///< The local processor row coordinate.
    inline static int mypcol;///< The local processor column coordinate.
    inline static int mpisize;///< The total number of processors.
    inline static int mpirank;///< The local MPI rank.

    /**
     * Call MPI_Barrier.
     */
    static void barrier();
};


#endif//DISTMATRIX_BLACS_H
