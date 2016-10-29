#ifndef _SVD_LAPACK_H_
#define _SVD_LAPACK_H_

#include <cstdlib>
#include <iostream>
#include <vector>
#include "GKlib.h"

/* DGESDD prototype */
extern "C" {
  void dgesdd_( char* jobz, int* m, int* n, double* a,
                int* lda, double* s, double* u, int* ldu, double* vt, int* ldvt,
                double* work, int* lwork, int* iwork, int* info );
}

void svdUsingLapack(gk_csr_t *mat, int rank, 
    std::vector<std::vector<double>>& uFac, 
    std::vector<std::vector<double>>& iFac);

void svdLapackRoutine(double *a, double *U, double *Vt, double *S, 
    int *iWork, int lda, int ldu, int ldvt, int m, int n);

#endif
