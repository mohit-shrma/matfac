#include "svdLapack.h"


void svdLapackRoutine(double *a, double *U, double *Vt, double *S, 
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


void svdUsingLapack(gk_csr_t *mat, int rank, 
    std::vector<std::vector<double>>& uFac, 
    std::vector<std::vector<double>>& iFac) {

  int nrows = mat->nrows;
  int ncols = mat->ncols;
  int u, k, ii, item, min_mn;
  double rating;

  //declare variables for svd comp
  double *U, *Vt, *S;
  int m, n, lda, ldu, ldvt;
  int *iWork;

  //create dense matrix
  double *dMat  = (double*) malloc(sizeof(double)*nrows*ncols);
  memset(dMat, 0, sizeof(double)*nrows*ncols);
  
  for (u = 0; u < nrows; u++) {
    for (ii = mat->rowptr[u]; ii < mat->rowptr[u+1]; ii++) {
      item = mat->rowind[ii];
      rating = mat->rowval[ii];
      dMat[u*ncols + item] = rating;    
    }
  }

  //define variables for svd comp  min_mn = ncols;
  min_mn = ncols;
  if (nrows < ncols) {
    min_mn = nrows;
  }
  m = nrows, n = ncols;
  lda = m, ldu =  m, ldvt = n;
  iWork = (int*) malloc(sizeof(int)*8*min_mn);  
  S = (double*) malloc(sizeof(double)*min_mn); 
  U = (double*) malloc(sizeof(double)*ldu*min_mn);
  Vt = (double*) malloc(sizeof(double)*ldvt*min_mn);
  
  svdLapackRoutine(dMat, U, Vt, S, iWork, lda, ldu, ldvt, m, n);
  
  //copy left-singular vectors to uFac
  for (u = 0; u < nrows; u++) {
    for (k = 0; k < rank; k++) {
      //columnwise storage [u][k] - [k*nrows + u]
      uFac[u][k] = U[k*nrows + u]; 
    }
  }

  //copy right-singular vectors to iFac
  for (item = 0; item < ncols; item++) {
    for (k = 0; k < rank; k++) {
      //columnwise storage and transpose
      //V[i,j] = Vt[j,i]
      iFac[item][k] = Vt[item*min_mn + k]; //Vt[k,item]
    }
  }
  

  free(U);
  free(Vt);
  free(S);
  free(dMat);
  free(iWork);
}


