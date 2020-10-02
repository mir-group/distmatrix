#include <distmatrix.h>
#include <doctest.h>
#include <iostream>
#include <matrix.h>
#include <type_traits>

TEST_CASE_TEMPLATE("assignment and copy", MatType, Matrix<int>, DistMatrix<int>) {
    MatType A(3, 4);
    MatType B = A;
    int rank = DistMatrix<int>::mpirank;
    //    A.fence();
    int should_set = rank == 0 || std::is_same_v<MatType, Matrix<int>>;
    if (should_set) A.set(1, 2, 3);

    A.barrier();

    SUBCASE("assignment with reference semantics") {
        REQUIRE(A.nrows == 3);
        REQUIRE(A.ncols == 4);
        REQUIRE(A(1, 2) == 3);
        REQUIRE(B.ncols == 4);
        REQUIRE(B.nrows == 3);
        REQUIRE(B(1, 2) == 3);
    }
    SUBCASE("copy constructor with reference semantics") {
        MatType C(A);
        REQUIRE(C.ncols == 4);
        REQUIRE(C.nrows == 3);
        REQUIRE(C(1, 2) == 3);
        if (should_set) A.set(0, 0, 4);
        A.barrier();
        REQUIRE(C(0, 0) == 4);
    }
    SUBCASE("explicit deep copy") {
        MatType C(3, 4);
        A.copy_to(C);
        if (should_set) A.set(1, 2, 7);
        //        A.fence();
        REQUIRE(C.ncols == 4);
        REQUIRE(C.nrows == 3);
        REQUIRE(C(1, 2) == 3);
    }
}

TEST_CASE("lambda init and transform") {
    Matrix<int> A(3, 4);
    A = [](int i, int j) {
        return i * i * j * j * j;
    };
    SUBCASE("init") {
        REQUIRE(A(1, 2) == 8);
        REQUIRE(A(2, 1) == 4);
        REQUIRE(A(2, 2) == 32);
    }
    SUBCASE("transform") {
        A = [](int x, int i, int j) {
            return x + i + j;
        };
        REQUIRE(A(1, 2) == 11);
        REQUIRE(A(2, 1) == 7);
        REQUIRE(A(2, 2) == 36);
    }
}

TEST_CASE("arithmetic and boolean") {
    Matrix<int> A(2, 3);
    A = 2;
    REQUIRE(A.all_equal(2));
    REQUIRE((A == 2).all_equal(true));
    SUBCASE("rescale and add") {
        A *= 5;
        REQUIRE(A.all_equal(10));
        A += 3;
        REQUIRE(A.all_equal(13));
    }
    SUBCASE("equality between matrices, sum") {
        A = {1, 2, 3, 1, 2, 3};
        Matrix<bool> expected(2, 3);
        Matrix<bool> result(2, 3);
        result = A == 2;
        expected = {false, true, false, false, true, false};
        REQUIRE((result == expected).all_equal(true));
        REQUIRE(result.all_equal(expected));
        REQUIRE(A.sum() == 12);
    }
    SUBCASE("matrix addition") {
        Matrix<int> B(2, 3);
        A = {1, 2, 3, 1, 2, 3};
        B = {3, 2, 1, 3, 2, 1};
        A += B;
        REQUIRE(A.all_equal(4));
    }
}