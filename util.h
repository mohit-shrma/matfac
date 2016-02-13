#include <vector>
#include <functional>
#include <algorithm>
#include <unordered_set>
#include <iostream>
#include <cmath>
#include "GKlib.h"

#ifndef _UTIL_H_
#define _UTIL_H_

double meanRating(gk_csr_t* mat);
int nnzSubMat(gk_csr_t *mat, int uStart, int uEnd, int iStart, int iEnd);
bool isInsideBlock(int u, int item, int uStart, int uEnd, int iStart, 
    int iEnd);

inline
double dotProd(const std::vector<double> &a, const std::vector<double> &b, int size) {
  double prod = 0.0;
  for (int i = 0; i < size; i++) {
    prod += a[i]*b[i];
  }
  return prod;
}
double stddev(std::vector<double> v);

void genStats(gk_csr_t *mat, 
    std::vector<std::unordered_set<int>> uISetIgnore, std::string opPrefix);
int getNNZ(gk_csr_t *mat);
void getInvalidUsersItems(gk_csr_t *mat, 
    std::vector<std::unordered_set<int>>& uISetIgnore,
    std::unordered_set<int>& uSet,
    std::unordered_set<int>& itemSet);
#endif

