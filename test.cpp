#include <iostream>

#include "svdLapack.h"

/* Parameters */
#define M 6
#define N 4
#define LDA M
#define LDU M
#define LDVT N


/* Auxiliary routine: printing a matrix */
void print_matrix( char* desc, int m, int n, double* a, int lda ) {
        int i, j;
        printf( "\n %s\n", desc );
        for( i = 0; i < m; i++ ) {
                for( j = 0; j < n; j++ ) printf( " %6.2f", a[i+j*lda] );
                printf( "\n" );
        }
}


void displayMat(double *a, int m, int n) {
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      std::cout << a[i*m + j] << " ";
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
  dgesdd_(&jobz, &m, &n, a, &lda, S, U, &ldu, Vt, &ldvt, &wkopt, &lwork, iWork,
      &info);

  /* Check for convergence */
  if( info > 0 ) {
    std::cout << "The algorithm computing SVD failed to converge.\n";
    exit(1);
  }

  free(work);
}


/*
int main(int argc, char *argv[]) {
  
  int m = M, n = N, lda = LDA, ldu = LDU, ldvt = LDVT;
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


  
  int info, lwork;
  double *work;
  double wkopt;
  char jobz = 'S';
  
  // Query and allocate the optimal workspace 
  lwork= -1; 
  dgesdd_(&jobz, &m, &n, a, &lda, S, U, &ldu, Vt, &ldvt, &wkopt, &lwork, iWork,
      &info);
  lwork = (int)wkopt;
  work = (double*) malloc(sizeof(double)*lwork);
 
  std::cout << "\nQueried workspace" << std::endl;

  //compute SVD
  dgesdd_(&jobz, &m, &n, a, &lda, S, U, &ldu, Vt, &ldvt, &wkopt, &lwork, iWork,
      &info);

  // Check for convergence 
  if( info > 0 ) {
    std::cout << "The algorithm computing SVD failed to converge.\n" << std::endl;
    exit(1);
  }

  free(work);

  //svdRoutine(a, U, Vt, S, iWork, lda, ldu, ldvt, m, n);
  
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
*/


int main(int argc, char *argv[]) {
          /* Locals */
        int m = M, n = N, lda = LDA, ldu = LDU, ldvt = LDVT, info, lwork;
        double wkopt;
        double* work;
        /* Local arrays */
   /* iwork dimension should be at least 8*min(m,n) */
   int iwork[8*N];
        double s[N], u[LDU*M], vt[LDVT*N];
        double a[LDA*N] = {
            7.52, -0.76,  5.13, -4.75,  1.33, -2.40,
           -1.10,  0.62,  6.62,  8.52,  4.91, -6.77,
           -7.95,  9.34, -5.66,  5.75, -5.49,  2.34,
            1.08, -7.10,  0.87,  5.30, -3.52,  3.95
        };
        /* Executable statements */
        printf( " DGESDD Example Program Results\n" );
        /* Query and allocate the optimal workspace */
        lwork = -1;
        dgesdd_( "Singular vectors", &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, &wkopt,
         &lwork, iwork, &info );
        lwork = (int)wkopt;
        work = (double*)malloc( lwork*sizeof(double) );
        /* Compute SVD */
        dgesdd_( "Singular vectors", &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, work,
         &lwork, iwork, &info );
        /* Check for convergence */
        if( info > 0 ) {
                printf( "The algorithm computing SVD failed to converge.\n" );
                exit( 1 );
        }
        /* Print singular values */
        print_matrix( "Singular values", 1, n, s, 1 );
        /* Print left singular vectors */
        print_matrix( "Left singular vectors (stored columnwise)", m, n, u, ldu );
        /* Print right singular vectors */
        print_matrix( "Right singular vectors (stored rowwise)", n, n, vt, ldvt );
        /* Free workspace */
        free( (void*)work );
        exit( 0 );

}


