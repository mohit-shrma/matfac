#include <vector>
#include "GKlib.h"

#ifndef _UTIL_H_
#define _UTIL_H_

double meanRating(gk_csr_t* mat);


inline
double dotProd(const std::vector<double> &a, const std::vector<double> &b, int size) {
  double prod = 0.0;
  for (int i = 0; i < size; i++) {
    prod += a[i]*b[i];
  }
  return prod;
}

#endif

