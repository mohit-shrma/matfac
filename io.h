#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <cstdlib>
#include <cassert>
#include "GKlib.h"
#include "datastruct.h"
#include "util.h"

void readMat(std::vector<std::vector<double>>& mat, int nrows, int ncols, 
    char *fileName);
void writeMat(std::vector<std::vector<double>>& mat, int nrows, int ncols,
              const char* opFileName);
void writeCSRWSparsityStructure(gk_csr_t *mat, const char *opFileName, 
    std::vector<std::vector<double>> uFac, 
    std::vector<std::vector<double>> iFac, int facDim);


