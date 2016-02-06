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
    for (ii = mat->rowptr[u]; ii < mat->rowptr[u+1]; ii++) {
      item = mat->rowind[ii];
      if (item >= iStart && item < iEnd) {
        nnz++;
       }
     }
  } 

  return nnz;
}


//check if (u, item) is present inside the passed block
//includes start but exclude end
bool isInsideBlock(int u, int item, int uStart, int uEnd, int iStart, 
    int iEnd) {
  if ((u >= uStart && u < uEnd) && (item >= iStart && item < iEnd)) {
    return true;
  } else {
    return false;
  }
}


double stddev(std::vector<double> v) {
  double sum = std::accumulate(v.begin(), v.end(), 0.0);
  double mean = sum / v.size();
  std::vector<double> diff(v.size());
  std::transform(v.begin(), v.end(), diff.begin(),
                     std::bind2nd(std::minus<double>(), mean));
  double sq_sum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
  double stdev = std::sqrt(sq_sum / v.size());
  return stdev;
}


