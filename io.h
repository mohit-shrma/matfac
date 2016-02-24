#ifndef _IO_H_
#define _IO_H_

#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <cstdlib>
#include <cassert>
#include <random>
#include "GKlib.h"
#include "util.h"

void dispVector(std::vector<double>& vec); 
void readMat(std::vector<std::vector<double>>& mat, int nrows, int ncols, 
    const char *fileName);
void writeMat(std::vector<std::vector<double>>& mat, int nrows, int ncols,
              const char* opFileName);
void writeCSRWSparsityStructure(gk_csr_t *mat, const char *opFileName, 
    std::vector<std::vector<double>> uFac, 
    std::vector<std::vector<double>> iFac, int facDim);
void writeCSRWHalfSparsity(gk_csr_t *mat, const char *opFileName, int uStart,
    int uEnd, int iStart, int iEnd);
void writeVector(std::vector<double>& vec, const char *opFileName);

std::vector<int> readVector(const char *ipFileName);
void writeTrainTestMat(gk_csr_t *mat,  const char* trainFileName, 
     const char* testFileName, float testPc, int seed);

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

#endif
