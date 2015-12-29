#include "util.h"


double meanRating(gk_csr_t* mat) {
  int u, ii, nnz;
  double avg = 0;
  nnz = 0;
  for (u = 0; u < mat->nrows; u++) {
    for (ii = mat->rowptr[u]; ii < mat->rowptr[u+1]; ii++) {
      avg += mat->rowval[ii];
      nnz++;
    }
  }
  avg = avg/nnz;
  return avg;
}

//include start exclude end
int nnzSubMat(gk_csr_t *mat, int uStart, int uEnd, int iStart, int iEnd) {
  
  int u, ii, item;
  int nnz = 0;

  for (u = uStart; u < uEnd; u++) {
    for (int ii = mat->rowptr[u]; ii < mat->rowptr[u+1]; ii++) {
      item = mat->rowind[ii];
      if (item >= iStart && item < iEnd) {
        nnz++;
       }
     }
  } 

  return nnz;
}

