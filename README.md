# distmatrix

distmatrix is a lightweight C++ distributed matrix library.
It wraps BLACS, ScaLAPACK and MPI-RMA in a user-friendly interface that lets you seamlessly distribute matrices across multiple machines.

## Example usage
See the `tests` directory for more examples.

```c++
#include <blacs.h>
#include <distmatrix.h>
...
// initialize BLACS
blacs::initialize();

// create matrix object with default blocking
DistMatrix<int> A(10,10);

// initialize all the elements
A = [](int i, int j){return 2*i + j;};

// synchronize
blacs::barrier();

// all mpi ranks can now read any element of A,
// local or remote
assert(A(3,4) == 10);

// set a matrix element, either local or remote
if(blacs::mpirank == 0) A.set(1,2,-1);
blacs::barrier();

// all ranks can now read that element
assert(A(1,2)==-1);

// finalize BLACS
blacs::finalize();
```

## Features

## Design choices
- 0-based indexing.
- Column-major memory layout.
- Reference semantics, i.e. `B=A` makes B point to the same underlying data as A.
- Leaving everything accessible to the user. While bad software design, it allows the user to call e.g. ScaLAPACK directly when this library does not wrap the required functionality.

## Caveats
- We use `MPI_Win_lock`/`MPI_Win_unlock` around the remote setting and fetching of elements, but that does not guarantee much. Use `blacs::barrier()` generously.
- The parenthesis operator *cannot* be used to set elements, so `A(1,2) = 4` does not work. This is because we cannot return a reference to a non-local element. Use `A.set(1,2,4)` instead.