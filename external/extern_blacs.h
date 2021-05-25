#ifndef DISTMATRIX_EXTERN_BLACS_H
#define DISTMATRIX_EXTERN_BLACS_H
#include <complex>

extern "C" {
void blacs_get_(int *, int *, int *);
void blacs_exit_(int *);
void blacs_pinfo_(int *, int *);
void blacs_gridinit_(int *, const char *, int *, int *);
void blacs_gridinfo_(int *, int *, int *, int *, int *);
void descinit_(int *, int *, int *, int *, int *, int *, int *, int *, int *, int *);
void blacs_gridexit_(const int *);
int numroc_(int *, int *, int *, int *, int *);

//void pdelset_(double *, int *, int *, int *, double *);
//void pdelget_(char *, char *, double *, double *, const int *, const int *,
//              const int *);
void infog2l_(const int *, const int *, const int *, const int *, const int *,
              const int *, const int *, int *, int *, int *, int *);

int indxg2p_(int *, int *, int *, int *, int *);

int indxg2l_(const int *, const int *, const int *, const int *, const int *);

int indxl2g_(int *, int *, int *, int *, int *);

void psgemm_(const char *, const char *, int *, int *, int *, const float *,
             float *, int *, int *, const int *, float *, int *, int *,
             const int *, float *, float *, int *, int *, int *);
void pdgemm_(const char *, const char *, int *, int *, int *, const double *,
             double *, int *, int *, const int *, double *, int *, int *,
             const int *, double *, double *, int *, int *, int *);
void pcgemm_(const char *, const char *, int *, int *, const int *,
             const std::complex<float> *, std::complex<float> *, int *, int *,
             const int *, std::complex<float> *, int *, int *, const int *,
             std::complex<float> *, std::complex<float> *, int *, int *,
             int *);
void pzgemm_(const char *, const char *, int *, int *, const int *,
             const std::complex<double> *, std::complex<double> *, int *, int *,
             const int *, std::complex<double> *, int *, int *, const int *,
             std::complex<double> *, std::complex<double> *, int *, int *,
             int *);

void psgeqrf_(int *m, int *n, float *a, int *ia, int *ja, int *desca, float *tau, float *work, int *lwork, int *info);
void pstrtri_(const char *uplo, const char *diag, int *n, float *a, int *ia, int *ja, int *desca, int *info);
void psormqr_(char *side, char *trans, int *m, int *n, int *k, float *a, int *ia, int *ja, int *desca, float *tau, float *c, int *ic, int *jc, int *descc, float *work, int *lwork, int *info);
void pdgeqrf_(int *m, int *n, double *a, int *ia, int *ja, int *desca, double *tau, double *work, int *lwork, int *info);
void pdtrtri_(const char *uplo, const char *diag, int *n, double *a, int *ia, int *ja, int *desca, int *info);
void pdormqr_(char *side, char *trans, int *m, int *n, int *k, double *a, int *ia, int *ja, int *desca, double *tau, double *c, int *ic, int *jc, int *descc, double *work, int *lwork, int *info);

void pctrtri_(const char *uplo, const char *diag, int *n, std::complex<float> *a, int *ia, int *ja, int *desca, int *info);
void pztrtri_(const char *uplo, const char *diag, int *n, std::complex<double> *a, int *ia, int *ja, int *desca, int *info);

void pspotrf_(const char *uplo, int *n, float *a, int *ia, int *ja, int *desca, int *info);
void pdpotrf_(const char *uplo, int *n, double *a, int *ia, int *ja, int *desca, int *info);
void pcpotrf_(const char *uplo, int *n, std::complex<float> *a, int *ia, int *ja, int *desca, int *info);
void pzpotrf_(const char *uplo, int *n, std::complex<double> *a, int *ia, int *ja, int *desca, int *info);

void psgemr2d_(int *m, int *n, float *a, int *ia, int *ja, int *desca, float *b, int *ib, int *jb, int *descb, int *ictxt);
void pdgemr2d_(int *m, int *n, double *a, int *ia, int *ja, int *desca, double *b, int *ib, int *jb, int *descb, int *ictxt);
void pcgemr2d_(int *m, int *n, std::complex<float> *a, int *ia, int *ja, int *desca, std::complex<float> *b, int *ib, int *jb, int *descb, int *ictxt);
void pzgemr2d_(int *m, int *n, std::complex<double> *a, int *ia, int *ja, int *desca, std::complex<double> *b, int *ib, int *jb, int *descb, int *ictxt);
void pigemr2d_(int *m, int *n, int *a, int *ia, int *ja, int *desca, int *b, int *ib, int *jb, int *descb, int *ictxt);

void pdsyev_(char *, char *, int *, double *, int *, int *, int *, double *,
             double *, int *, int *, int *, double *, int *, int *);
//void pzelset_(std::complex<double> *, int *, int *, int *,
//              std::complex<double> *);
//void pzelget_(char *, char *, std::complex<double> *, std::complex<double> *,
//              int *, int *, int *);

void pzheev_(char *, char *, int *, std::complex<double> *, int *, int *, int *,
             double *, std::complex<double> *, int *, int *, int *,
             std::complex<double> *, int *, std::complex<double> *, int *,
             int *);
};
#endif//DISTMATRIX_EXTERN_BLACS_H
