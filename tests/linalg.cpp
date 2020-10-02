#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <cmath>
#include <doctest.h>
#include <iostream>
#include <matrix.h>
#include <numeric>

// TODO: Check a complex type too
TEST_CASE("matrix multiplication") {
    const int m = 7, k = 5, n = 6;
    Matrix<double> A(m, k);
    Matrix<double> B(k, n);
    Eigen::Matrix<double, m, k> Aeig = Eigen::Matrix<double, m, k>::Random();
    Eigen::Matrix<double, k, n> Beig = Eigen::Matrix<double, k, n>::Random();
    A = [&Aeig](int i, int j) {
        return Aeig(i, j);
    };
    B = [&Beig](int i, int j) {
        return Beig(i, j);
    };
    auto C = A.matmul(B, 3);
    auto Ceig = 3 * Aeig * Beig;
    Matrix<bool> difference_is_small(m, n);
    difference_is_small = [&C, &Ceig](int i, int j) {
        return std::abs(C(i, j) - Ceig(i, j)) < 1e-12;
    };
    REQUIRE(difference_is_small.all_equal(true));
}

TEST_CASE("general eigensolver") {
    const int n = 7;
    Matrix<double> A(n, n);
    Eigen::MatrixXd Aeig = Eigen::MatrixXd::Random(n, n);
    A = [&Aeig](int i, int j) {
        return Aeig(i, j);
    };
    /*std::cout << Aeig << "\n\n";
    A.print();*/
    auto [eigvals, eigvecs] = A.diagonalize();
    Eigen::EigenSolver<decltype(Aeig)> es(Aeig);
    auto eig_eigvals = es.eigenvalues();
    auto eig_eigvecs = es.eigenvectors();
    /*std::cout << eig_eigvals << std::endl;
    for (int i = 0; i < n; i++) {
        std::cout << eigvals[i] << ", ";
    }
    std::cout << std::endl;*/
    std::vector<double> diffs(n);
    SUBCASE("eigenvalues") {
        // compare with Eigen, with reordering
        double error_eigvals = 0.0;
        for (int i = 0; i < n; i++) {
            std::complex<double> l = eig_eigvals[i];
            std::transform(eigvals.begin(), eigvals.end(), diffs.begin(), [l](auto x) {
                return std::norm(x - l);
            });
            int i_mapped = std::distance(diffs.begin(), std::min_element(diffs.begin(), diffs.end()));
            /*std::cout << i_mapped << ",         ";
            std::for_each(diffs.begin(), diffs.end(), [](auto x) { std::cout << x << ", "; });
            std::cout << std::endl;*/
            error_eigvals += std::abs(diffs[i_mapped]);
        }
        REQUIRE(error_eigvals < 1e-15);
    }
    SUBCASE("eigenvectors") {
        double error_eigvecs = 0.0;
        /*for (int j = 0; j < n; j++) {
            for (int i = 0; i < n; i++) {
                eig_eigvecs(i, j) = eigvecs(i, j);
            }
            eig_eigvals[j] = eigvals[j];
        }
        auto AV = Aeig * eig_eigvecs;
        auto VL = eig_eigvecs * eig_eigvals.asDiagonal();
        std::cout << eig_eigvecs << "\n\n";
        eigvecs.print();
        error_eigvecs = (AV - VL).norm();*/
        for (int j = 0; j < n; j++) {
            std::complex<double> l = eigvals[j];
            for (int i = 0; i < n; i++) {
                std::complex<double> Avij(0.0);
                for (int k = 0; k < n; k++) {
                    Avij += A(i, k) * eigvecs(k, j);
                }
                error_eigvecs += std::norm(Avij - l * eigvecs(i, j));
            }
        }
        REQUIRE(error_eigvecs < 1e-12);
    }
}