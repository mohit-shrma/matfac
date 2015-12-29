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
      std::cout << a[i*m + j] << " ";
    }
    std::cout << std::endl;
  }
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
  displayMat(U, m, min_mn);
  std::cout << "\nVt: " << std::endl;
  displayMat(Vt, min_mn, n);
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

