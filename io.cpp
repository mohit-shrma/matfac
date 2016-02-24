#include "io.h"


void readMat(std::vector<std::vector<double>>& mat, int nrows, int ncols, 
    const char *fileName) {
  
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

std::vector<int> readVector(const char *ipFileName) {
  std::vector<int> vec;
  std::ifstream ipFile(ipFileName);
  std::string line; 
  if (ipFile.is_open()) {
    while(getline(ipFile, line)) {
      if (line.length() > 0) {
        vec.push_back(std::stoi(line));
      }
    }
    ipFile.close();
  } else {
    std::cerr <<  "\nCan't open file: " << ipFileName;
    exit(0);
  }
  return vec;
}


void writeVector(std::vector<double>& vec, const char *opFileName) {
  std::ofstream opFile(opFileName);
  if (opFile.is_open()) {
    for (int i = 0; i < vec.size(); i++) {
      opFile << vec[i] << std::endl;    
    }
    opFile.close();
  }
}


void dispVector(std::vector<double>& vec) {
  std::cout << "\n";
  for (int i = 0; i < vec.size(); i++) {
    std::cout << vec[i] << ",";
  }
  std::cout << std::endl;
}



void writeTrainTestMat(gk_csr_t *mat,  const char* trainFileName, 
    const char* testFileName, float testPc, int seed) {
  int k;
  int nnz = getNNZ(mat);
  int nTest = testPc * nnz;
  int* color = (int*) malloc(sizeof(int)*nnz);
  memset(color, 0, sizeof(int)*nnz);
 
  //initialize uniform random engine
  std::mt19937 mt(seed);
  //nnz dist
  std::uniform_int_distribution<int> nnzDist(0, nnz-1);

  for (int i = 0; i < nTest; i++) {
    k = nnzDist(mt);
    color[k] = 1;
  }

  //split the matrix based on color
  gk_csr_t** mats = gk_csr_Split(mat, color);
  
  //save first matrix as train
  gk_csr_Write(mats[0], (char*) trainFileName, GK_CSR_FMT_CSR, 1, 0);

  //save second matrix as test
  gk_csr_Write(mats[1], (char*) testFileName, GK_CSR_FMT_CSR, 1, 0);

  free(color);
  gk_csr_Free(&mats[0]);
  gk_csr_Free(&mats[1]);
  //TODO: free mats
  //gk_csr_Free(&mats);
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


void writeCSRWHalfSparsity(gk_csr_t *mat, const char *opFileName, int uStart,
    int uEnd, int iStart, int iEnd) {
  
  int u, ii, item;
  float rating;
  
  std::ofstream opFile (opFileName);
  if (opFile.is_open()) {
    for (u = 0; u < mat->nrows; u++) {
      for (ii = mat->rowptr[u]; ii < mat->rowptr[u+1]; ii++) {
        item = mat->rowind[ii];
        rating = mat->rowval[ii];
        if (isInsideBlock(u, item, uStart, uEnd, iStart, iEnd)) {
          if (std::rand()%2 == 0) {
            opFile << item << " " << rating << " ";
          }
        } else {
          opFile << item << " " << rating << " ";
        }
      }
      opFile << std::endl;
    }
    opFile.close();
  }

}

bool isFileExist(const char *fileName) {
  std::ifstream infile(fileName);
  return infile.good();
}


void writeBlkDiagJoinedCSR(const char* mat1Name, const char* mat2Name, 
    const char* opFileName) {
 
  gk_csr_t *mat1 = gk_csr_Read((char*)mat1Name, GK_CSR_FMT_CSR, 1, 0);
  gk_csr_t *mat2 = gk_csr_Read((char*)mat2Name, GK_CSR_FMT_CSR, 1, 0);

  std::cout << "\n mat1 nrows: " << mat1->nrows << " ncols: " << mat1->ncols;
  std::cout << "\n mat2 nrows: " << mat2->nrows << " ncols: " << mat2->ncols << std::endl;
  

  std::ofstream opFile(opFileName);

  if (opFile.is_open()) {

    int nItems1 = mat1->ncols;
  
    //write first block on the diagonal
    for (int u = 0; u < mat1->nrows; u++) {
      for (int ii = mat1->rowptr[u]; ii < mat1->rowptr[u+1]; ii++) {
        int item = mat1->rowind[ii];
        float rating = mat1->rowval[ii];
        opFile << item << " " << rating << " ";
      }
      opFile << std::endl;
    }

    //write second block on the diagonal, items offset by prev mat items
    for (int u = 0; u < mat2->nrows; u++) {
      for (int ii = mat2->rowptr[u]; ii < mat2->rowptr[u+1]; ii++) {
        int item = mat2->rowind[ii];
        float rating = mat2->rowval[ii];
        opFile << (item+nItems1) << " " << rating << " ";
      }
      opFile << std::endl;
    }
    
    opFile.close();
  } else {
    std::cerr << "\nCan't open file: " << opFileName  << std::endl;
  }
  

  if (NULL != mat1) {
    gk_csr_Free(&mat1);
  }
  
  if (NULL != mat2) {
    gk_csr_Free(&mat2);
  }
}


