#ifndef _IO_H_
#define _IO_H_

#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <cstdlib>
#include <cassert>
#include <random>
#include <omp.h>
#include "GKlib.h"
#include "util.h"
#include "const.h"

void dispVector(std::vector<double>& vec); 
void readMat(std::vector<std::vector<double>>& mat, int nrows, int ncols, 
    const char *fileName);
void readMat(Eigen::MatrixXf& mat, int nrows, int ncols, 
    const char *fileName);
void writeMat(std::vector<std::vector<double>>& mat, int nrows, int ncols,
              const char* opFileName);
void writeMat(Eigen::MatrixXf& mat, int nrows, int ncols,
              const char* opFileName);
void writeCSRWSparsityStructure(gk_csr_t *mat, const char *opFileName, 
    std::vector<std::vector<double>> uFac, 
    std::vector<std::vector<double>> iFac, int facDim);
void writeCSRWHalfSparsity(gk_csr_t *mat, const char *opFileName, int uStart,
    int uEnd, int iStart, int iEnd);
void writeVector(std::vector<double>& vec, const char *opFileName);
void writeVector(std::vector<double>& vec, std::ofstream& opFile);
std::vector<int> readVector(const char *ipFileName);
std::vector<double> readDVector(const char *ipFileName);
void writeTrainTestMat(gk_csr_t *mat,  const char* trainFileName, 
     const char* testFileName, float testPc, int seed);
void writeTrainTestValMat(gk_csr_t *mat,  const char* trainFileName, 
    const char* testFileName, const char *valFileName, float testPc, 
    float valPc, int seed);
void writeSubSampledMat(gk_csr_t *mat,  const char* sampFileName, 
    float sampPc, int seed);
bool isFileExist(const char *fileName);

template <typename Iter>
void writeContainer(Iter it, Iter end, const char *opFileName) {
  std::ofstream opFile(opFileName);
  if (opFile.is_open()) {
    for (; it != end; ++it) {
      opFile << *it << std::endl;
    }
    opFile.close();
  }
}


template <typename Iter>
void dispContainer(Iter it, Iter end) {
    for (; it != end; ++it) {
      std::cout << *it << std::endl;
    }
}

void writeBlkDiagJoinedCSR(const char* mat1Name, const char* mat2Name, 
    const char* opFileName);
void writeItemSimMat(gk_csr_t *mat, const char* fName);
void writeItemSimMatNonSymm(gk_csr_t *mat, const char* fName);
void writeItemJaccSimMat(gk_csr_t *mat, const char *fName);
void writeItemJaccSimMatPar(gk_csr_t *mat, const char *fName);
void writeCoRatings(gk_csr_t *mat, const char *fName);
void writeItemJaccSimFrmCorat(gk_csr_t *mat, gk_csr_t *coRatMat, 
    const char *fName);
void readItemScores(std::vector<std::pair<int, double>>& itemScores,
    const char* fileName);
void writeItemScores(std::vector<std::pair<int, double>>& itemScores,
    const char* fileName);
void writeTailTestMat(gk_csr_t *mat, const char* testFileName, 
    std::unordered_set<int>& headItems);
void writeRandMatCSR(const char* opFileName,
    std::vector<std::vector<double>>& uFac, 
    std::vector<std::vector<double>>& iFac, int facDim, int seed, int nnz);
void writeMatBin(std::vector<std::vector<double>>& mat, int nrows, int ncols, 
    const char *opFileName);
void readMatBin(std::vector<std::vector<double>>& mat, int nrows, int ncols, 
    const char *opFileName);
Eigen::VectorXf readEigVector(const char *ipFileName) ;


#endif
