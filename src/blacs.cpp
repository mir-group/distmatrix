//
// Created by anders on 10/5/20.
//

#include <blacs.h>
#include <extern_blacs.h>
#include <mpi.h>
#include <vector>
#include <iostream>

void blacs::initialize(int nprows, int npcols) {
    int rank, size, context, myprow, mypcol;
    char layout = blacs::blacslayout;
    blacs_pinfo_(&rank, &size);
    blacs::mpisize = size;
    blacs::mpirank = rank;
    if (nprows < 1 || npcols < 1) {
        std::vector<int> dims(2);
        MPI_Dims_create(size, 2, &dims[0]);
        nprows = dims[0];
        npcols = dims[1];
    }
    int blah = -1, what = 0;
    blacs_get_(&blah, &what, &context);
    blacs_gridinit_(&context, &layout, &nprows, &npcols);
    blacs_gridinfo_(&context, &nprows, &npcols, &myprow, &mypcol);
    blacs::nprows = nprows;
    blacs::npcols = npcols;
    blacs::blacscontext = context;
    blacs::myprow = myprow;
    blacs::mypcol = mypcol;
    std::cout << "Hello from rank " << rank << " = (" << myprow << "," << mypcol << ")\n";
    MPI_Barrier(MPI_COMM_WORLD);
}

void blacs::finalize(int finalize_mpi) {
    int cont = !finalize_mpi;
    blacs_gridexit_(&blacs::blacscontext);
    blacs_exit_(&cont);
}

void blacs::barrier() {
    MPI_Barrier(MPI_COMM_WORLD);
}