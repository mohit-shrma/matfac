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

