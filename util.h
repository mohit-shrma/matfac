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
#include <random>
#include <numeric>
#include <Eigen/Dense>
#include "GKlib.h"

std::unordered_set<int> getHeadItems(gk_csr_t *mat, float pc);
std::unordered_set<int> getHeadUsers(gk_csr_t *mat, float pc);
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
std::pair<double, double> meanStdDev(std::vector<double> v);

void genStats(gk_csr_t *mat, 
    std::vector<std::unordered_set<int>> uISetIgnore, std::string opPrefix);
int getNNZ(gk_csr_t *mat);
void getInvalidUsersItems(gk_csr_t *mat, 
    std::vector<std::unordered_set<int>>& uISetIgnore,
    std::unordered_set<int>& uSet,
    std::unordered_set<int>& itemSet);
std::pair<std::vector<double>, std::vector<double>> getRowColFreq(gk_csr_t *mat);
std::pair<std::unordered_set<int>, std::unordered_set<int>> getNZUserItem(
    gk_csr_t *mat);
std::vector<std::pair<int, int>> getUIPairs(gk_csr_t *mat, 
    std::unordered_set<int>& invalidUsers, 
    std::unordered_set<int>& invalidItems);
std::vector<std::tuple<int, int, float>> getUIRatings(gk_csr_t* mat);
std::vector<std::tuple<int, int, float>> getUIRatings(gk_csr_t* mat, 
    std::unordered_set<int>& invalidUsers, 
    std::unordered_set<int>& invalidItems);
double normVec(std::vector<double>& vec);
bool descComp(const std::pair<int, double> a, const std::pair<int, double> b);
bool ascComp(const std::pair<int, double> a, const std::pair<int, double> b);
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
std::vector<std::vector<std::tuple<int,int,float>>> getRandUIRatings(
    gk_csr_t* mat, int nBlocks, int seed);
double avgPairs(std::vector<std::pair<int, double>> pairs);
bool compMat(gk_csr_t *mat1, gk_csr_t *mat2);
std::vector<double> meanItemRating(gk_csr_t *mat);
std::vector<double> getColFreq(gk_csr_t *mat, 
    std::unordered_set<int> sampUsers);
void getRatedItems(gk_csr_t* mat, int user, std::unordered_set<int>& ratedItems);
std::vector<std::pair<double, double>> trainUsersMeanVar(gk_csr_t* mat);
std::vector<std::pair<double, double>> trainItemsMeanVar(gk_csr_t* mat);
int setIntersect(std::unordered_set<int>& a, std::unordered_set<int>& b);
int setUnion(std::unordered_set<int>& a, std::unordered_set<int>& b);
float pearsonCorr(std::vector<float>& x, std::vector<float>& y, float xMean, 
    float yMean);
int getMaxItemInd(gk_csr_t* mat);
void parBlockShuffle(std::vector<size_t>& arr, std::mt19937& mt);
int binSearch(int *sortedArr, int key, int ub, int lb);
float adapDotProd(Eigen::MatrixXf& uFac, Eigen::MatrixXf& iFac, 
   int u, int item, int minRank);
void sgdUpdateBlockSeq(int dim, std::vector<std::pair<int, int>>& updateSeq, 
    std::mt19937& mt);

//return a min element of a vector
template<typename T>
T minVec(std::vector<T> v) {
  typename std::vector<T>::iterator result = std::min_element(std::begin(v), std::end(v));
  int minInd = std::distance(std::begin(v), result); 
  return v[minInd];
}

template<typename T>
T maxVec(std::vector<T> v) {
  typename std::vector<T>::iterator result = std::max_element(std::begin(v), std::end(v));
  int maxInd = std::distance(std::begin(v), result); 
  return v[maxInd];
}

std::unordered_set<int> getColInd(gk_csr_t* mat);
std::unordered_set<int> getRowInd(gk_csr_t* mat);
#endif

