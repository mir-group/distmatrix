#define DOCTEST_CONFIG_IMPLEMENT
#include <doctest.h>
#include <distmatrix.h>
#include <mpi.h>
#include <blacs.h>

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    blacs::initialize();
    doctest::Context context;

    context.applyCommandLine(argc, argv);

    int res = context.run();// run
    blacs::finalize(0);
    MPI_Finalize();

    if (context.shouldExit())// important - query flags (and --exit) rely on the user doing this
        return res;          // propagate the result of the tests

    return res;
}