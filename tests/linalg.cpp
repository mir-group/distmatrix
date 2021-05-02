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
    const int m = 7, k = 5, n = 6;
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

TEST_CASE_TEMPLATE("QR matrix inversion", MatType, DistMatrix<double>) {
    const int m = 7;
    const int n = 7;

    MatType A(m, n);
    Eigen::MatrixXd Aeig = Eigen::MatrixXd::Random(m, n);
    MPI_Bcast(Aeig.data(), m * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    A = [&Aeig](int i, int j) {
        return Aeig(i, j);
    };

    MatType Ainv = A.qr_invert();
    std::cout << "Done qr_invert in test" << std::endl;
    MatType I = A.matmul(Ainv, 1.0, 'T', 'T');
    std::cout << "Done A matmul A_inv" << std::endl;
    MatType error(n, n); // TODO: need to be nxn matrix
    error = [&I](int i, int j) {
        return i == j ? std::abs(1 - I(i, j)) : std::norm(I(i, j));
    };
    // std::cout << error.sum() << std::endl;
    REQUIRE(error.sum() < 1e-12);
    std::cout << "Passed test qr_invert" << std::endl;
}

//TEST_CASE_TEMPLATE("QR matrix multiplication", MatType, DistMatrix<double>) {
//    const int m = 17;
//    const int n = 7;
//
//    MatType A(m, n);
//    Eigen::MatrixXd Aeig = Eigen::MatrixXd::Random(m, n);
//    MPI_Bcast(Aeig.data(), m * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
//    A = [&Aeig](int i, int j) {
//        return Aeig(i, j);
//    };
//    std::cout << "Created A" << std::endl; 
//
//    MatType b(m, 1);
//    Eigen::VectorXd beig = Eigen::VectorXd::Random(m);
//    MPI_Bcast(beig.data(), m, MPI_DOUBLE, 0, MPI_COMM_WORLD);
//    b = [&beig](int i, int j) {
//        return beig(i);
//    };
//    std::cout << "Created b" << std::endl; 
//
//    MatType QR(m, n);
//    std::vector<double> tau;
//    std::tie(QR, tau) = A.qr();
//    MatType R(n, n);
//    R = [&QR](int i, int j) {return i > j ? 0 : QR(i, j);};
//    MatType Rinv = R.triangular_invert('U');
//    MatType Q_b = QR.QT_matmul(b, tau);
//    std::cout << "Q_b size " << Q_b.nrows << " " << Q_b.ncols << std::endl;
//    MatType alpha = Rinv.matmul(Q_b, 1.0, 'N', 'N');
//    std::cout << "Done distmatrix alpha" << std::endl;
//
//    Eigen::HouseholderQR<Eigen::MatrixXd> qr(Aeig);
//    Eigen::VectorXd Q_beig = (qr.householderQ().transpose() * beig).segment(0, n);
//    Eigen::MatrixXd eye_mat = Eigen::MatrixXd::Identity(n, n);
//    Eigen::MatrixXd R_inv_eig = qr.matrixQR().block(0, 0, n, n)
//                       .triangularView<Eigen::Upper>()
//                       .solve(eye_mat);
//    std::cout << "Done R_inv_eig" << std::endl;
//    Eigen::VectorXd alpha_eig = R_inv_eig * Q_beig;
//    std::cout << "Done alpha_eig" << std::endl;
//    //MatType error(n, 1);
//    std::cout << "beig size " << beig.size() << std::endl;
//    std::cout << "Q.T size " << qr.householderQ().transpose().rows() << " " << qr.householderQ().transpose().cols() << std::endl;
//    std::cout << "Q_beig size " << Q_beig.rows() << " " << Q_beig.cols() << std::endl;
//    std::cout << "Qb compare" << std::endl;
//
//    double error = 0.0;
//    for (int i = 0; i < n; i++) {
//      for (int j = 0; j < n; j++) {
//        error += std::abs(Rinv(i, j) - R_inv_eig(i, j));
//      }
//    }
//    std::cout << "Rinv error " << error << std::endl;
//    REQUIRE(error < 1e-12);
// 
//    error = 0.0;
//    for (int i = 0; i < n; i++) {
//      error += std::abs(Q_b(i, 0) - Q_beig(i));
//    }
//    std::cout << "Q_b error " << error << std::endl;
//    REQUIRE(error < 1e-12);
//
//
//    std::cout << "alpha compare" << std::endl;
//    error = 0.0;
//    for (int i = 0; i < n; i++) {
//      error += std::abs(alpha(i, 0) - alpha_eig(i));
//    }
//    std::cout << "alpha error " << error << std::endl;
//    REQUIRE(error < 1e-10);
//}

TEST_CASE_TEMPLATE("Cholesky decomposition and triangular inversion", MatType, DistMatrix<double>) {
    const int n = 17;
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
            return std::norm(A(i, j) - LLT(i, j));
        };
        // std::cout << error.sum() << std::endl;
        REQUIRE(error.sum() < 1e-12);
    }
    SUBCASE("Triangular inversion") {
        MatType Linv = L.triangular_invert('L');
        auto I = L.matmul(Linv);
        MatType error(n, n);
        error = [&I](int i, int j) {
            return i == j ? std::abs(1 - I(i, j)) : std::norm(I(i, j));
        };
        // std::cout << error.sum() << std::endl;
        REQUIRE(error.sum() < 1e-12);
    }
}


TEST_CASE("general eigensolver") {
    const int n = 7;
    Matrix<double> A(n, n);
    Eigen::MatrixXd Aeig = Eigen::MatrixXd::Random(n, n);
    A = [&Aeig](int i, int j) {
        return Aeig(i, j);
    };

    auto [eigvals, eigvecs] = A.diagonalize();
    Eigen::EigenSolver<decltype(Aeig)> es(Aeig);
    auto eig_eigvals = es.eigenvalues();
    auto eig_eigvecs = es.eigenvectors();

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
            error_eigvals += std::abs(diffs[i_mapped]);
        }
        REQUIRE(error_eigvals < 1e-15);
    }
    SUBCASE("eigenvectors") {
        double error_eigvecs = 0.0;
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
