#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest.h>
#include <iostream>
#include <matrix.h>

TEST_CASE("assignment and copy") {
    Matrix<int> A(3, 4);
    Matrix<int> B = A;
    A.set(1, 2, 3);
    SUBCASE("assignment with reference semantics") {
        REQUIRE(A.nrows == 3);
        REQUIRE(A.ncols == 4);
        REQUIRE(A.nlocal == 12);
        REQUIRE(A(1, 2) == 3);
        REQUIRE(B.ncols == 4);
        REQUIRE(B.nrows == 3);
        REQUIRE(B.nlocal == 12);
        REQUIRE(B(1, 2) == 3);
    }
    SUBCASE("explicit deep copy") {
        Matrix<int> C(3, 4);
        A.copy_to(C);
        A.set(1, 2, 7);
        REQUIRE(C.ncols == 4);
        REQUIRE(C.nrows == 3);
        REQUIRE(C.nlocal == 12);
        REQUIRE(C(1, 2) == 3);
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