#include <iostream>

#include "svdLapack.h"

/* Parameters */
#define M 6
#define N 4
#define LDA M
#define LDU M
#define LDVT N


void displayMat(double *a, int m, int n) {
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      //a[i][j] - a[j*m + i] columnwise
      std::cout << a[j*m + i] << " ";
    }
    std::cout << std::endl;
  }
}


void svdRoutine(double *a, double *U, double *Vt, double *S, 
    int *iWork, int lda, int ldu, int ldvt, int m, int n) {
  
  int info, lwork;
  double *work;
  double wkopt;
  char jobz = 'S';
  
  /* Query and allocate the optimal workspace */
  lwork= -1; 
  dgesdd_(&jobz, &m, &n, a, &lda, S, U, &ldu, Vt, &ldvt, &wkopt, &lwork, iWork,
      &info);
  lwork = (int)wkopt;
  work = (double*) malloc(sizeof(double)*lwork);
  
  //compute SVD
  dgesdd_(&jobz, &m, &n, a, &lda, S, U, &ldu, Vt, &ldvt, work, &lwork, iWork,
      &info);

  /* Check for convergence */
  if( info > 0 ) {
    std::cout << "The algorithm computing SVD failed to converge.\n";
    exit(1);
  }

  free(work);
}


int main(int argc, char *argv[]) {
  
  int m = M, n = N, lda = LDA, ldu = LDU, ldvt = LDVT, info, lwork;
  int min_mn = n;
  double *U, *Vt, *S;
  int *iWork;
  double a[LDA*N] = {
            7.52, -0.76,  5.13, -4.75,  1.33, -2.40,
           -1.10,  0.62,  6.62,  8.52,  4.91, -6.77,
           -7.95,  9.34, -5.66,  5.75, -5.49,  2.34,
            1.08, -7.10,  0.87,  5.30, -3.52,  3.95
        };
  iWork = (int*) malloc(sizeof(int)*8*min_mn);  
  S = (double*) malloc(sizeof(double)*min_mn); 
  U = (double*) malloc(sizeof(double)*ldu*min_mn);
  Vt = (double*) malloc(sizeof(double)*ldvt*min_mn);
 
  svdLapackRoutine(a, U, Vt, S, iWork, lda, ldu, ldvt, m, n);
  
  std::cout << "\nU: " << std::endl;
  //displayMat(U, m, min_mn);
  //top-3 singular vectors
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < 3; j++) {
      std::cout << U[j*m + i] << " ";
    }
    std::cout << std::endl;
  }

  std::cout << "\nVt: " << std::endl;
  //displayMat(Vt, min_mn, n);
  //top-3 singular vectors
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < 3; j++) {
      std::cout << Vt[i*min_mn + j] << " ";
    }
    std::cout << std::endl;
  }


  std::cout << "\nSingular values: " << std::endl; 
  for (int i = 0; i < min_mn; i++) {
    std::cout << S[i] << " " ;
  }

  free(U);
  free(Vt);
  free(S);
  free(iWork);

  return 0;
}

