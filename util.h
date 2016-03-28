#ifndef _UTIL_H_
#define _UTIL_H_
#include <vector>
#include <functional>
#include <algorithm>
#include <unordered_set>
#include <iostream>
#include <fstream>
#include <cmath>
#include <tuple>
#include "GKlib.h"

std::unordered_set<int> getHeadItems(gk_csr_t *mat, float topPc);
double compRecall(std::vector<int> order1, std::vector<int> order2, int N);
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
std::pair<std::vector<double>, std::vector<double>> getRowColFreq(gk_csr_t *mat);
std::vector<std::pair<int, int>> getUIPairs(gk_csr_t *mat, 
    std::unordered_set<int>& invalidUsers, 
    std::unordered_set<int>& invalidItems);
std::vector<std::tuple<int, int, float>> getUIRatings(gk_csr_t* mat);
std::vector<std::tuple<int, int, float>> getUIRatings(gk_csr_t* mat, 
    std::unordered_set<int>& invalidUsers, 
    std::unordered_set<int>& invalidItems);
double normVec(std::vector<double>& vec);
bool descComp(std::pair<int, double>& a, std::pair<int, double>& b);
std::pair<double, double> getMeanVar(std::vector<std::vector<double>> uFac,
    std::vector<std::vector<double>> iFac, int facDim, int nUsers, int nItems);
void getUserStats(std::vector<int>& users, gk_csr_t* mat, 
    std::unordered_set<int>& filtItems, const char* opFName);
float sparseRowDotProd(gk_csr_t* mat1, int i, gk_csr_t* mat2, int j);
float sparseColDotProd(gk_csr_t* mat1, int i, gk_csr_t* mat2, int j);
int sparseBinColDotProd(gk_csr_t* mat1, int i, gk_csr_t* mat2, int j);
int sparseCoRatedUsers(gk_csr_t* mat, int i, int j);
int checkIfUISorted(gk_csr_t* mat);
int coRatedUsersFrmSortedMat(gk_csr_t* mat, int i, int j);
int coRatedUsersFrmSortedMatLinMerge(gk_csr_t* mat, int i, int j);
std::vector<int> getInvalidUsers(gk_csr_t *mat);
#endif

