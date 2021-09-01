#include <blacs.h>
#include <distmatrix.h>
#include <doctest.h>
#include <iostream>
#include <matrix.h>
#include <type_traits>

TEST_CASE_TEMPLATE("assignment and copy", MatType, Matrix<int>, DistMatrix<int>) {
    MatType A(3, 4);
    MatType B = A;
    int rank = blacs::mpirank;
    int should_set = rank == 0 || std::is_same_v<MatType, Matrix<int>>;

    if (should_set) A.set(2, 1, 3);
    A.fence();

    SUBCASE("assignment with reference semantics") {
        REQUIRE(A.nrows == 3);
        REQUIRE(A.ncols == 4);
        REQUIRE(A(2, 1) == 3);
        REQUIRE(B.ncols == 4);
        REQUIRE(B.nrows == 3);
        REQUIRE(B(2, 1) == 3);
        if constexpr (std::is_same_v<MatType, DistMatrix<int>>) {
            REQUIRE(A.nlocalrows == B.nlocalrows);
            REQUIRE(A.nlocalcols == B.nlocalcols);
            REQUIRE(A.nlocal == B.nlocal);
        }
    }
    blacs::barrier();
    SUBCASE("copy constructor with reference semantics") {
        MatType C(A);
        REQUIRE(C.ncols == 4);
        REQUIRE(C.nrows == 3);
        REQUIRE(C(2, 1) == 3);
        if (should_set) A.set(0, 0, 4);
        A.fence();
        REQUIRE(C(0, 0) == 4);
    }
    SUBCASE("explicit deep copy") {
        MatType C(3, 4);
        A.copy_to(C);
        C.fence();
        if (should_set) A.set(2, 1, 7);
        REQUIRE(C.ncols == 4);
        REQUIRE(C.nrows == 3);
        REQUIRE(C(2, 1) == 3);
    }
}

TEST_CASE_TEMPLATE("lambda init and transform", MatType, Matrix<int>, DistMatrix<int>) {
    MatType A(3, 4);
    A = [](int i, int j) {
        return i * i * j * j * j;
    };
    blacs::barrier();
    SUBCASE("init") {
        REQUIRE(A(1, 2) == 8);
        REQUIRE(A(2, 1) == 4);
        REQUIRE(A(2, 2) == 32);
    }
    SUBCASE("transform") {
        A = [](int x, int i, int j) {
            return x + i + j;
        };
        blacs::barrier();
        REQUIRE(A(1, 2) == 11);
        REQUIRE(A(2, 1) == 7);
        REQUIRE(A(2, 2) == 36);
    }
}

TEST_CASE_TEMPLATE("arithmetic and boolean", MatType, Matrix<int>, DistMatrix<int>) {
    MatType A(3, 2);
    A = 2;
    blacs::barrier();
    REQUIRE((A == 2));

    SUBCASE("rescale and add") {
        A *= 5;
        blacs::barrier();
        REQUIRE((A == 10));
        A += 3;
        blacs::barrier();
        REQUIRE((A == 13));
    }

    SUBCASE("equality between matrices, sum") {
        A = {1, 2, 3, 1, 2, 3};
        blacs::barrier();

        MatType expected(3, 2);
        MatType result(3, 2);
        result = [&A](int i, int j) { return A(i, j) == 2; };
        expected = {false, true, false, false, true, false};

        blacs::barrier();
        REQUIRE((result == expected));
        REQUIRE(A.sum() == 12);
    }

    SUBCASE("matrix addition") {
        MatType B(3, 2);
        A = {1, 2, 3, 1, 2, 3};
        B = {3, 2, 1, 3, 2, 1};
        A += B;
        REQUIRE((A == 4));
    }
}

TEST_CASE_TEMPLATE("gather and allgather", ValueType, int, float, double) {
    int m = 7, n = 11;
    DistMatrix<ValueType> A(m, n);
    A = [](int i, int j) {
        return 2 * i + j * j;
    };
    blacs::barrier();
    Matrix<ValueType> Aserial(m, n);

    DistMatrix<ValueType> B(m, n);
    B = [](int i, int j) {
        return 2 * i + j * j;
    };
    Matrix<ValueType> Bserial(m, n);
    blacs::barrier();
    SUBCASE("gather") {
        A.gather(Aserial.array.get());
        B.gather(Bserial.array.get());

        Matrix<int> check(m, n);
        check = [&Aserial](int i, int j) {
            return Aserial(i, j) == 2 * i + j * j;
        };
        if (blacs::mpirank == 0) {
            REQUIRE((check.sum() == m * n));
        }

    }
    SUBCASE("allgather") {
        A.allgather(Aserial.array.get());
        B.allgather(Bserial.array.get());

        Matrix<int> check(m, n);
        check = [&Aserial](int i, int j) {
            return Aserial(i, j) == 2 * i + j * j;
        };
        REQUIRE((check.sum() == m * n));
    }
}

TEST_CASE_TEMPLATE("collect", ValueType, int, float, double) {
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int M = 11, N = 16;
    int mb = 2, nb = 11;
    int m = mb, n = nb;
    if (world_size > 1 && blacs::mpirank == world_size - 1) m = mb - 1;
    int p;
    MPI_Allreduce(&m, &p, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    std::cout << "p=" << p << std::endl;

    DistMatrix<ValueType> A(M, N);
    A = [](int i, int j) {
        return -1;
    };

    int r = blacs::mpirank;
    Matrix<ValueType> Aserial(m, n);
    Aserial = [&r](int i, int j) {
        return j * j * r * r;
    };

    A.collect(Aserial.array.get(), 0, 0, p, n, mb, n, m);
    blacs::barrier();
    if (blacs::mpirank == 0) {
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                std::cout << A(i, j) << " ";
            }
            std::cout << std::endl;
        }
    }
    std::cout << std::endl;

    Matrix<ValueType> Acheck(M, N);
    Acheck = [](int i, int j) {
        return -1;
    };

    int row = 0;
    for (int k = 0; k < world_size; k++) {
      int m = 2;
      if (world_size > 1 && k == world_size - 1) m = 1;
      for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
          Acheck.set(row, j, j * j * k * k);
        }
        row += 1;
      }
    }

    if (blacs::mpirank == 0) {
        std::cout << "Acheck" << std::endl;
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                std::cout << Acheck(i, j) << " ";
            }
            std::cout << std::endl;
        }
    }
    std::cout << std::endl;


    Matrix<int> check(M, N);
    check = [&A, &Acheck](int i, int j) {
        return A(i, j) == Acheck(i, j);
    };
    std::cout << "check sum=" << check.sum() << std::endl; 
    REQUIRE((check.sum() == M * N));

}
