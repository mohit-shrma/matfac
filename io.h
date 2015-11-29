#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <cstdlib>
#include <cassert>

void readMat(std::vector<std::vector<double>>& mat, int nrows, int ncols, 
    char *fileName);

void readMat(double **mat, int nrows, int ncols, 
    char *fileName);

void writeMat(std::vector<std::vector<double>>& mat, int nrows, int ncols,
              const char* opFileName);


