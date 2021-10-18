#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <cmath>
#include <distmatrix.h>
#include <doctest.h>
#include <iostream>
#include <matrix.h>
#include <mpi.h>
#include <numeric>

// TODO: Check a complex type too
TEST_CASE_TEMPLATE("matrix multiplication", MatType, DistMatrix<double>) {
    //const int m = 7, k = 5, n = 6;
    const int m = 61, k = 59, n = 47;
    MatType A(m, k);
    MatType B(n, k);
    Eigen::Matrix<double, m, k> Aeig = Eigen::Matrix<double, m, k>::Random();
    Eigen::Matrix<double, n, k> Beig = Eigen::Matrix<double, n, k>::Random();
    // avoid random seeding issues
    MPI_Bcast(Aeig.data(), m * k, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(Beig.data(), k * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    A = [&Aeig](int i, int j) {
        return Aeig(i, j);
    };
    B = [&Beig](int i, int j) {
        return Beig(i, j);
    };
    MatType C = A.matmul(B, 3, 'N', 'T');
    auto Ceig = 3 * Aeig * Beig.transpose();
    MatType difference(m, n);
    difference = [&C, &Ceig](int i, int j) {
        return std::norm(C(i, j) - Ceig(i, j));
    };
    REQUIRE(difference.sum() < 1e-12);
}

TEST_CASE_TEMPLATE("QR matrix multiplication", MatType, DistMatrix<double>) {
    const int m = 289;
    const int n = 47;

    MatType A(m, n);
    Eigen::MatrixXd Aeig = Eigen::MatrixXd::Random(m, n);
    MPI_Bcast(Aeig.data(), m * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    A = [&Aeig](int i, int j) {
        return Aeig(i, j);
    };
    std::cout << "Created A" << std::endl; 

    MatType b(m, 1);
    std::cout << "created b" << std::endl; 
    Eigen::VectorXd beig = Eigen::VectorXd::Random(m);
    std::cout << "created beig" << std::endl; 
    MPI_Bcast(beig.data(), m, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    std::cout << "broadcast beig" << std::endl; 
    b = [&beig](int i, int j) {
        return beig(i, j);
    };
    std::cout << "assigned values to b" << std::endl; 

    MatType QR(m, n);
    std::vector<double> tau;
    std::tie(QR, tau) = A.qr();
    MatType R(n, n);
    R = [&QR](int i, int j) {return i > j ? 0 : QR(i, j, true);};
    MatType Rinv = R.triangular_invert('U');
    MatType Q_b = QR.Q_matmul(b, tau, 'L', 'T');
    std::cout << "Q_b size " << Q_b.nrows << " " << Q_b.ncols << std::endl;
    MatType alpha = Rinv.matmul(Q_b, 1.0, 'N', 'N');
    std::cout << "Done distmatrix alpha" << std::endl;

    Eigen::HouseholderQR<Eigen::MatrixXd> qr(Aeig);
    Eigen::VectorXd Q_beig = (qr.householderQ().transpose() * beig).segment(0, n);
    Eigen::MatrixXd eye_mat = Eigen::MatrixXd::Identity(n, n);
    Eigen::MatrixXd R_inv_eig = qr.matrixQR().block(0, 0, n, n)
                       .triangularView<Eigen::Upper>()
                       .solve(eye_mat);
    std::cout << "Done R_inv_eig" << std::endl;
    Eigen::VectorXd alpha_eig = R_inv_eig * Q_beig;
    std::cout << "Done alpha_eig" << std::endl;
    //MatType error(n, 1);
    std::cout << "beig size " << beig.size() << std::endl;
    std::cout << "Q.T size " << qr.householderQ().transpose().rows() << " " << qr.householderQ().transpose().cols() << std::endl;
    std::cout << "Q_beig size " << Q_beig.rows() << " " << Q_beig.cols() << std::endl;
    std::cout << "Qb compare" << std::endl;

    double error = 0.0;
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) {
        error += std::abs(Rinv(i, j, true) - R_inv_eig(i, j));
      }
    }
    std::cout << "Rinv error " << error << std::endl;
    REQUIRE(error < 1e-12);
 
    error = 0.0;
    for (int i = 0; i < n; i++) {
      error += std::abs(Q_b(i, 0, true) - Q_beig(i));
    }
    std::cout << "Q_b error " << error << std::endl;
    REQUIRE(error < 1e-12);


    std::cout << "alpha compare" << std::endl;
    error = 0.0;
    for (int i = 0; i < n; i++) {
      error += std::abs(alpha(i, 0, true) - alpha_eig(i));
    }
    std::cout << "alpha error " << error << std::endl;
    REQUIRE(error < 1e-10);
}

TEST_CASE_TEMPLATE("Cholesky decomposition and triangular inversion", MatType, DistMatrix<double>) {
    const int n = 61;
    MatType A(n, n);
    Eigen::MatrixXd Aeig = Eigen::MatrixXd::Random(n, n);
    MPI_Bcast(Aeig.data(), n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    // symmetric, positive definite matrix
    A = [&Aeig](int i, int j) {
        return Aeig(i, j) + Aeig(j, i) + (i == j ? n : 0);
    };
    MatType L = A.cholesky();
    SUBCASE("Cholesky decomposition") {
        MatType LLT = L.matmul(L, 1.0, 'N', 'T');
        MatType error(n, n);
        error = [&LLT, &A](int i, int j) {
            return std::norm(A(i, j, true) - LLT(i, j, true));
        };
        // std::cout << error.sum() << std::endl;
        REQUIRE(error.sum() < 1e-12);
    }
    SUBCASE("Triangular inversion") {
        MatType Linv = L.triangular_invert('L');
        auto I = L.matmul(Linv);
        MatType error(n, n);
        error = [&I](int i, int j) {
            return i == j ? std::abs(1 - I(i, j, true)) : std::norm(I(i, j, true));
        };
        // std::cout << error.sum() << std::endl;
        REQUIRE(error.sum() < 1e-12);
    }
    std::cout << "Cholesky matches" << std::endl; 

}

