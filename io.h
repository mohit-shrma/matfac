#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <cstdlib>
#include <cassert>
#include "GKlib.h"
#include "datastruct.h"
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

