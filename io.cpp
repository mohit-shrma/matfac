#include "io.h"


void readMat(std::vector<std::vector<double>>& mat, int nrows, int ncols, 
    char *fileName) {
  
  std::string line, token;
  std::string delimiter = " ";
  std::ifstream inFile (fileName);
  int i, j; 
  size_t pos;

  if (inFile.is_open()) {
    i = 0;
    while (getline(inFile, line)) {
      j = 0;
      //split the line
      while((pos = line.find(delimiter)) != std::string::npos) {
        token = line.substr(0, pos);
        mat[i][j++] = std::stod(token);
        line.erase(0, pos + delimiter.length());
      }
      if (line.length() > 0) {
        mat[i][j++] = std::stod(line);
      }
      assert(j == ncols);
      i++;
    }
    inFile.close();
  } else {
    std::cout << "\nCan't open file: " << fileName;
    exit(0);
  }
  
}


void writeMat(std::vector<std::vector<double>>& mat, int nrows, int ncols, 
              const char* opFileName) {
  int i, j;
  std::ofstream opFile (opFileName);
  
  if (opFile.is_open()) {
    for (i = 0; i < nrows; i++) { 
      for (j = 0; j < ncols; j++) {
        opFile << mat[i][j] << " ";
      }
      opFile << std::endl;
    }
    opFile.close();
  }

}


void writeCSRWSparsityStructure(gk_csr_t *mat, const char *opFileName, 
    std::vector<std::vector<double>> uFac, 
    std::vector<std::vector<double>> iFac, int facDim) {
  
  int item, u, ii;
  float rating;

  std::ofstream opFile (opFileName);
  if (opFile.is_open()) {
    for (u = 0; u < mat->nrows; u++) {
      for (ii = mat->rowptr[u]; ii < mat->rowptr[u+1]; ii++) {
        item = mat->rowind[ii];
        rating = dotProd(uFac[u], iFac[item], facDim);         
        opFile << item << " " << rating << " ";
      }
      opFile << std::endl;
    }
    opFile.close();
  }

}

