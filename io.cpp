#include "io.h"


void readItemScores(std::vector<std::pair<int, double>>& itemScores,
    const char* fileName) {
  
  std::string line, token;
  std::string delimiter = " ";
  std::ifstream inFile (fileName);
  int item;
  double score;
  size_t pos;

  if (inFile.is_open()) {
    while (getline(inFile, line)) {
      //split the line
      if ((pos = line.find(delimiter)) != std::string::npos) {
        token = line.substr(0, pos);
        item = std::stoi(token);
        line.erase(0, pos + delimiter.length());
      }
      if (line.length() > 0) {
        score = std::stod(line);
        itemScores.push_back(std::make_pair(item, score));
      }
    }
    inFile.close();
  } else {
    std::cout << "\nCan't open file: " << fileName;
  }

}


void writeItemScores(std::vector<std::pair<int, double>>& itemScores,
    const char* fileName) {
  
  std::ofstream opFile(fileName);
  
  for (auto&& itemScore: itemScores) {
    opFile << itemScore.first << " " << itemScore.second << std::endl;
  }

  opFile.close();
}


void readMat(std::vector<std::vector<double>>& mat, int nrows, int ncols, 
    const char *fileName) {
 
  std::cout << "\nReading ... " << fileName << std::endl;

  std::string line, token;
  std::string delimiter = " ";
  std::ifstream inFile (fileName);
  int i, j; 
  size_t pos;

  if (inFile.is_open()) {
    i = 0;
    while (getline(inFile, line) && i < nrows) {
      j = 0;
      //split the line
      while((pos = line.find(delimiter)) != std::string::npos) {
        token = line.substr(0, pos);
        mat[i][j++] = std::stold(token);
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


void readMat(Eigen::MatrixXf& mat, int nrows, int ncols, 
    const char *fileName) {
 
  std::cout << "\nReading ... " << fileName << " nrows: " << nrows << 
    " ncols: " << ncols << std::endl;

  std::string line, token;
  std::string delimiter = " ";
  std::ifstream inFile (fileName);
  int i, j; 
  size_t pos;

  mat = Eigen::MatrixXf(nrows, ncols);

  if (inFile.is_open()) {
    i = 0;
    while (getline(inFile, line) && i < nrows) {
      j = 0;
      //split the line
      while((pos = line.find(delimiter)) != std::string::npos) {
        token = line.substr(0, pos);
        mat(i, j++) = std::stod(token);
        line.erase(0, pos + delimiter.length());
      }
      if (line.length() > 0) {
        mat(i, j++) = std::stod(line);
      }
      assert(j == ncols);
      i++;
     }
    inFile.close();
  } else {
    std::cout << "\nCan't open file: " << fileName << std::endl;
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


void writeMat(Eigen::MatrixXf& mat, int nrows, int ncols, 
              const char* opFileName) {
  int i, j;
  std::ofstream opFile (opFileName);
  
  if (opFile.is_open()) {
    for (i = 0; i < nrows; i++) { 
      for (j = 0; j < ncols; j++) {
        opFile << mat(i, j) << " ";
      }
      opFile << std::endl;
    }
    opFile.close();
  }

}


void writeMatBin(std::vector<std::vector<double>>& mat, int nrows, int ncols, 
    const char *opFileName) {
  std::ofstream opFile(opFileName, std::ios::binary);
  if (opFile.is_open()) {
    for (int i = 0; i < nrows; i++) {
      for (int j = 0; j < ncols; j++) {
        double val = mat[i][j];
        opFile.write((char*)& val, sizeof(double));
      }
    }
    opFile.close();
  }
}


void writeMatBin(Eigen::MatrixXf& mat, int nrows, int ncols, 
    const char *opFileName) {
  std::ofstream opFile(opFileName, std::ios::binary);
  if (opFile.is_open()) {
    for (int i = 0; i < nrows; i++) {
      for (int j = 0; j < ncols; j++) {
        double val = mat(i, j);
        opFile.write((char*)& val, sizeof(double));
      }
    }
    opFile.close();
  }
}


void writeBinarizedTrainValTest(gk_csr_t* mat, int ratThresh, 
    std::string opFilePref, int seed) {

  std::mt19937 mt(seed);
  
  std::string opFTrStr = opFilePref + ".rank.train.mat";
  std::string opFTeStr = opFilePref + ".rank.test.mat";
  std::string opFVaStr = opFilePref + ".rank.val.mat";

  std::ofstream opFileTrain(opFTrStr);
  std::ofstream opFileTest(opFTeStr);
  std::ofstream opFileVal(opFVaStr);

  if (opFileTrain.is_open()) {
    for (int u = 0; u < mat->nrows; u++) {
      
      std::vector<int> uItems;
      for (int ii = mat->rowptr[u]; ii < mat->rowptr[u+1]; ii++) {
        int item = mat->rowind[ii];
        float val = mat->rowval[ii];
        if (val > ratThresh) {
          uItems.push_back(item);
        }
      }
     
      int testI = -1, valI = -1;
      if (uItems.size() > 3) {
        std::uniform_int_distribution<> dis(0, uItems.size()-1);
        valI = uItems[dis(mt)];
        testI = uItems[dis(mt)];
        int nTry = 0;
        while (valI == testI && nTry++ < 500) {
          testI = uItems[dis(mt)];
        }
        if (testI != valI) {
          opFileTest << testI << " " << 1;
          opFileVal << valI << " " << 1;
        }
      }
 
      for (int ii = mat->rowptr[u]; ii < mat->rowptr[u+1]; ii++) {
        int item = mat->rowind[ii];
        float val = mat->rowval[ii];
        if (val > ratThresh) {
          if (item != valI && item != testI) {
            opFileTrain << item << " " << 1 << " ";
          }
        } else {
          opFileTrain << item << " " << 0 << " ";
        }
      }

      opFileTrain << std::endl;
      opFileTest << std::endl;
      opFileVal << std::endl;
    }
    
    opFileTrain.close();
    opFileTest.close();
    opFileVal.close();
  } else {
    std::cout << "Can't open file." << std::endl;
  }

}


void writeBinarizedMat(gk_csr_t* mat, int ratThresh, const char* opFName) {
  std::ofstream opFile(opFName);
  if (opFile.is_open()) {
    for (int u = 0; u < mat->nrows; u++) {
      for (int ii = mat->rowptr[u]; ii < mat->rowptr[u+1]; ii++) {
        int item = mat->rowind[ii];
        float rating = mat->rowval[ii];
        opFile << item << " ";
        if (rating > ratThresh) {
          opFile << 1 << " ";
        } else {
          opFile << 0 << " ";
        }
      }
      opFile << std::endl;
    }
    opFile.close();
  } else {
    std::cout << "Can't open file " << opFName << std::endl;
  }
}


void readMatBin(std::vector<std::vector<double>>& mat, int nrows, int ncols, 
    const char *opFileName) {
  std::ifstream ipFile(opFileName, std::ios::binary);
  if (ipFile.is_open()) {
    for (int i = 0; i < nrows; i++) {
      for (int j = 0; j < ncols; j++) {
        ipFile.read((char*)& mat[i][j], sizeof(double));
      }
    }
    ipFile.close();
  }
}


void readMatBin(Eigen::MatrixXf& mat, int nrows, int ncols, 
    const char *opFileName) {
  std::ifstream ipFile(opFileName, std::ios::binary);
  if (ipFile.is_open()) {
    for (int i = 0; i < nrows; i++) {
      for (int j = 0; j < ncols; j++) {
        ipFile.read((char*)& mat(i, j), sizeof(double));
      }
    }
    ipFile.close();
  }
}


std::vector<int> readVector(const char *ipFileName) {
  std::vector<int> vec;
  std::ifstream ipFile(ipFileName);
  std::string line; 
  if (ipFile.is_open()) {
    std::cout << "Reading... " << ipFileName << std::endl;
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


std::vector<double> readDVector(const char *ipFileName) {
  std::vector<double> vec;
  std::ifstream ipFile(ipFileName);
  std::string line; 
  if (ipFile.is_open ()) {
    std::cout << "Reading... " << ipFileName << std::endl;
    while(getline(ipFile, line)) {
      if (line.length( ) > 0) {
        vec.push_back(std::stod(line));
      }
    }
    ipFile.close();
  } else {
    std::cerr <<  "\nCan't open file: " << ipFileName;
    exit(0);
  }
  return vec;
}


Eigen::VectorXf readEigVector(const char *ipFileName) {
  std::vector<double> vec;
  std::ifstream ipFile(ipFileName);
  std::string line; 
  if (ipFile.is_open ()) {
    std::cout << "Reading... " << ipFileName << std::endl; 
    while(getline(ipFile, line)) {
      if (line.length( ) > 0) {
        vec.push_back(std::stod(line));
      }
    }
    ipFile.close();
  } else {
    std::cerr <<  "\nCan't open file: " << ipFileName;
    exit(0);
  }
  Eigen::VectorXf evec(vec.size());
  for (int i = 0; i < vec.size(); i++) {
    evec(i) = vec[i];
  }
  return evec;
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


void writeVector(Eigen::VectorXf& vec, const char *opFileName) {
  std::ofstream opFile(opFileName);
  if (opFile.is_open()) {
    for (int i = 0; i < vec.size(); i++) {
      opFile << vec[i] << std::endl;    
    }
    opFile.close();
  }
}


void writeVector(std::vector<double>& vec, std::ofstream& opFile) {
  if (opFile.is_open()) {
    for (int i = 0; i < vec.size(); i++) {
      opFile << vec[i] << ",";    
    }
  } else {
    std::cerr << "Can't write as opFile not open" << std::endl;
  }
}


void dispVector(std::vector<double>& vec) {
  for (int i = 0; i < vec.size(); i++) {
    std::cout << vec[i] << ",";
  }
  std::cout << std::endl;
}


void writeTrainTestValMat(gk_csr_t *mat,  const char* trainFileName, 
    const char* testFileName, const char *valFileName, float testPc, 
    float valPc, int seed) {
  int k, i;
  int nnz = getNNZ(mat);
  int nTest = testPc * nnz;
  int nVal = valPc * nnz;
  int* color = (int*) malloc(sizeof(int)*nnz);
  memset(color, 0, sizeof(int)*nnz);
 
  //initialize uniform random engine
  std::mt19937 mt(seed);
  //nnz dist
  std::uniform_int_distribution<int> nnzDist(0, nnz-1);
  
  std::cout << "nTest: " << nTest << " nVal: " << nVal << std::endl;
  for (i = 0; i < nTest; i++) {
    k = nnzDist(mt);
    color[k] = 1;
  }
   
  i = 0;
  while (i < nVal) {
    k = nnzDist(mt);
    if (!color[k]) {
      color[k] = 2;
      i++;
    }
  }

  std::cout  << "Partitionining matrix..." << std::endl;
  //split the matrix based on color
  gk_csr_t** mats = gk_csr_Split(mat, color);
  
  //save first matrix as train
  gk_csr_Write(mats[0], (char*) trainFileName, GK_CSR_FMT_CSR, GK_CSR_IS_VAL, 0);

  //save second matrix as test
  gk_csr_Write(mats[1], (char*) testFileName, GK_CSR_FMT_CSR, GK_CSR_IS_VAL, 0);

  //save third matrix as val
  gk_csr_Write(mats[2], (char*) valFileName, GK_CSR_FMT_CSR, GK_CSR_IS_VAL, 0);
  
  free(color);
  gk_csr_Free(&mats[0]);
  gk_csr_Free(&mats[1]);
  gk_csr_Free(&mats[2]);
  //TODO: free mats
  //gk_csr_Free(&mats);
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


void writeTailTestMat(gk_csr_t *mat, const char* testFileName, 
    std::unordered_set<int>& headItems) {
  
  std::ofstream opFile(testFileName); 
  int nTestRatings = 0; 
  for (int u = 0; u < mat->nrows; u++) {
    for (int ii = mat->rowptr[u]; ii < mat->rowptr[u+1] && nTestRatings < 5000; ii++) {
      int item = mat->rowind[ii];
      float rating = mat->rowval[ii];
      if (headItems.find(item) == headItems.end()) {
        //not found in head, tail item
        opFile << item << " " << rating << " ";
        nTestRatings++;
      }
    }
    opFile << std::endl;
  }

  opFile.close();
}


void writeSubSampledMat(gk_csr_t *mat,  const char* sampFileName, 
    float sampPc, int seed) {

  int k;
  int nnz = getNNZ(mat);
  int nSamp = sampPc * nnz;
  int* color = (int*) malloc(sizeof(int)*nnz);
  memset(color, 0, sizeof(int)*nnz);
 
  //initialize uniform random engine
  std::mt19937 mt(seed);
  //nnz dist
  std::uniform_int_distribution<int> nnzDist(0, nnz-1);

  int sumColor = 0;
  while (sumColor < nSamp) {
    k = nnzDist(mt);
    if (!color[k]) {
      color[k] = 1;
      sumColor++;
    }
  }

  //split the matrix based on color
  gk_csr_t** mats = gk_csr_Split(mat, color);

  int sampNNZ = getNNZ(mats[1]);
  std::cout << "\nparent NNZ: " << nnz << " sample NNZ: " << sampNNZ;
  std::cout << "\nPercent nnz in sample matrix: " 
    << (float)sampNNZ/(float)nnz << std::endl;
  
  std::cout << "Writing..." << sampFileName << std::endl;
  //save first matrix as sample mat
  gk_csr_Write(mats[1], (char*) sampFileName, GK_CSR_FMT_CSR, 1, 0);
  

  free(color);
  gk_csr_Free(&mats[0]);
  gk_csr_Free(&mats[1]);
  //TODO: free mats
  //gk_csr_Free(&mats);
}


void writeSampledSpMat(gk_csr_t* mat, const char *opFileName, int seed) {

  //initialize uniform random engine
  std::mt19937 mt(seed);
  
  std::vector<int> uSampledNRatings;
  
  std::cout << "sampling ratings for users..." << std::endl;
#pragma omp parallel for  
  for (int u = 0; u < mat->nrows; u++) {
    int nItems = mat->rowptr[u+1] - mat->rowptr[u];
    if (nItems <= 1) { 
      continue;
    } 
    std::uniform_int_distribution<> dis(1, nItems);
    int nSampRatings = dis(mt);
   
    std::unordered_set<int> filtUItems;
    int nTries = 0;
    while (filtUItems.size() < nSampRatings && nTries < nSampRatings + 500) {
      int sampOffset = dis(mt) - 1;
      int item = mat->rowind[mat->rowptr[u] + sampOffset];
      filtUItems.insert(item);
      nTries++;
    }

    for (int ii = mat->rowptr[u]; ii < mat->rowptr[u+1]; ii++) {
      int item = mat->rowind[ii]; 
      if (filtUItems.count(item) == 0) {
        mat->rowval[ii] = -1; //flag the rating for deletion
      }
    }

  }

  //create inverted index
  gk_csr_CreateIndex(mat, GK_CSR_COL);

  std::cout << "sampling ratings for items..." << std::endl;
  //randomly sampled ratings from columns 
#pragma omp parallel for
  for (int item = 0; item < mat->ncols; item++) {
    int nUsers = 0;
    
    for (int uu = mat->colptr[item]; uu < mat->colptr[item+1]; uu++) {
      if (mat->colval[uu] >= 0) {
        nUsers++;
      }
    }
    
    //std::cout << "item... " << item << " nUsers... " << nUsers << std::endl;
    if (nUsers <= 1) {
      continue;
    }

    std::uniform_int_distribution<> dis(1, nUsers);
    int nSampRatings = dis(mt);

    //std::cout << "nSampRatings... " << nSampRatings << std::endl;

    std::unordered_set<int> filtIUsers;
    int nTries = 0;
    while (filtIUsers.size() < nSampRatings && nTries < nSampRatings + 500) {
      int sampOffset = dis(mt) - 1;
      int user = mat->colind[mat->colptr[item] +  sampOffset];
      if (mat->colval[mat->colptr[item] + sampOffset] > 0) {
        filtIUsers.insert(user);
      }
      nTries++;
    }
     
    for (int uu = mat->colptr[item]; uu < mat->colptr[item+1]; uu++) {
      int user = mat->colind[uu];
      if (filtIUsers.count(user) == 0) {
        mat->colval[uu] = -1;
      }
    }
    
  }

  std::cout << "writing transpose..." << std::endl;
  //write transpose
  std::ofstream opFile("temp.transpose.mat");
  if (opFile.is_open()) {
    for (int item = 0; item < mat->ncols; item++) {
      for (int uu = mat->colptr[item]; uu < mat->colptr[item+1]; uu++) {
        if (mat->colval[uu] >= 0) {
          opFile << mat->colind[uu] << " " << mat->colval[uu] << " ";
        }
      }
      opFile << std::endl;
    } 
    opFile.close();
  }

  //read transposed matrix
  gk_csr_t *transMat = gk_csr_Read((char *) ("temp.transpose.mat"),
      GK_CSR_FMT_CSR, GK_CSR_IS_VAL, 0);
  gk_csr_t *writeMat = gk_csr_Transpose(transMat);
 
  //create inverted index
  gk_csr_CreateIndex(writeMat, GK_CSR_COL);
  
  //generate distribution of user ratings
  std::ofstream opFile2("userCount.txt");
  if (opFile2.is_open()) {
    for (int u = 0; u < writeMat->nrows; u++) {
      opFile2 << writeMat->rowptr[u+1] - writeMat->rowptr[u]  << std::endl;
    }
    opFile2.close();
  }

  //generate distribution of item ratings
  std::ofstream opFile3("itemCount.txt");
  if (opFile3.is_open()) {
    for (int item = 0; item < writeMat->ncols; item++) {
      opFile3 << writeMat->colptr[item+1] - writeMat->colptr[item] << std::endl;
    }
    opFile3.close();
  }


  //write the matrix
  gk_csr_Write(writeMat, (char *)opFileName, GK_CSR_FMT_CSR, GK_CSR_IS_VAL, 0);
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
        if (GK_CSR_IS_VAL) {
          opFile << item << " " << rating << " ";
        } else {
          opFile << item << " ";
        }
      }
      opFile << std::endl;
    }
    opFile.close();
  }

}


int nTuples(std::vector<std::unordered_set<int>> uItemSet) {
  int count = 0;
  for (auto&& uSet : uItemSet) {
    count += uSet.size();
  }
  return count;
}


void writeRandMatCSR(const char* opFileName,
    std::vector<std::vector<double>>& uFac, 
    std::vector<std::vector<double>>& iFac, int facDim, int seed, int nnz) {
  
  int nUsers = uFac.size();
  int nItems = iFac.size();
  
  std::vector<std::unordered_set<int>> uItemSet(nUsers);

  //initialize uniform random engine
  std::mt19937 mt(seed);
  //user dist
  std::uniform_int_distribution<int> uDist(0, nUsers-1);
  //item dist
  std::uniform_int_distribution<int> iDist(0, nItems-1);

  for (int u = 0; u < nUsers; u++) {
    //sample item
    int item = iDist(mt);
    uItemSet[u].insert(item);
  }
  
  for (int item = 0; item < nItems; item++) {
    //sample user
    int user = uDist(mt);
    uItemSet[user].insert(item);
  }
  
  int nPairs = nTuples(uItemSet);
  while(nPairs < nnz) {
  
    for (int i = 0; i < nnz-nPairs; i++) {
      //sample user
      int user = uDist(mt);
      //sample item
      int item = iDist(mt);
      uItemSet[user].insert(item);
    }
    
    nPairs = nTuples(uItemSet);
  }

  //write as CSR the sampled entries
  std::ofstream opFile(opFileName);
  if (opFile.is_open()) {
    for (int u = 0; u < nUsers; u++) {
      auto uSet = uItemSet[u];
      std::vector<int> items(uSet.begin(), uSet.end());
      std::sort(items.begin(), items.end());
      for (auto&& item: items) {
        if (GK_CSR_IS_VAL) {
          opFile << item << " " << dotProd(uFac[u], iFac[item], facDim) << " ";
        } else {
          opFile << item << " ";
        }
      }
      opFile << std::endl;
    }

    opFile.close();
  } 
}


void writeFiltRandMatCSR(const char* opFileName, 
    std::vector<std::vector<double>>& uFac,
    std::vector<std::vector<double>>& iFac, int facDim, int seed, 
    std::unordered_set<int>&  filtUsers,
    std::unordered_set<int>& filtItems, int nnz) {
  
  int nUsers = uFac.size();
  int nItems = iFac.size();
 
  int nFiltUsers = filtUsers.size();
  int nFiltItems = filtItems.size();

  std::vector<std::unordered_set<int>> uItemSet(nUsers);
  
  std::mt19937 mt(seed);
  std::vector<int> filtUVec(filtUsers.begin(), filtUsers.end());
  std::vector<int> filtIVec(filtItems.begin(), filtItems.end());
  
  std::uniform_int_distribution<int> uIndDist(0, nFiltUsers-1);
  std::uniform_int_distribution<int> itemIndDist(0, nFiltItems-1);

  int nPairs = 0;
  
  while (nPairs < nnz) {
    //sample user
    int user = filtUVec[uIndDist(mt)];
    //sample item
    int item = filtIVec[itemIndDist(mt)];
    if (uItemSet[user].count(item) == 0) {
      uItemSet[user].insert(item);
      nPairs++;
    }
  }
  
  //write as CSR for the sampled entries
  std::ofstream opFile(opFileName);
  
  if (opFile.is_open()) {
    
    for (int u = 0; u < nUsers; u++) {

      if (filtUsers.count(u) == 0) {
        opFile << std::endl;
        continue;
      }

      auto uSet = uItemSet[u];
      std::vector<int> items(uSet.begin(), uSet.end());

      std::sort(items.begin(), items.end());
      for (auto&& item: items) {
        if (GK_CSR_IS_VAL) {
          opFile << item << " " <<  dotProd(uFac[u], iFac[item], facDim) << " ";
        } else {
          opFile << item << " ";
        }
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


void writeItemSimMat(gk_csr_t *mat, const char* fName) {
  std::cout << "\nWriting... " << fName << std::endl;
  std::ofstream opFile(fName);
  for (int item1 = 0; item1 < mat->ncols; item1++) {
    for (int item2 = 0; item2 < mat->ncols; item2++) {
      if(item1 != item2 && sparseBinColDotProd(mat, item1, mat, item2)) {
        opFile << item2 << " "; 
      }
    }
    opFile << std::endl;
    if (item1 % 1000 == 0) {
      std::cout << "\nDone... Items " << item1 << std::endl;
    } 
  }
  opFile.close();
}


void writeItemSimMatNonSymm(gk_csr_t *mat, const char* fName) {
  std::cout << "\nWriting... " << fName << std::endl;
  std::ofstream opFile(fName);
  for (int item1 = 0; item1 < mat->ncols; item1++) {
    for (int item2 = item1+1; item2 < mat->ncols; item2++) {
      if(sparseBinColDotProd(mat, item1, mat, item2)) {
        opFile << item2 << " "; 
      }
    }
    opFile << std::endl;
    if (item1 % 1000 == 0) {
      std::cout << "\nDone... Items " << item1 << std::endl;
    } 
  }
  opFile.close();
}


void writeItemJaccSimMat(gk_csr_t *mat, const char *fName) {
  auto rowColFreq = getRowColFreq(mat);
  auto userFreq = rowColFreq.first;
  auto itemFreq = rowColFreq.second;
  int nItems = mat->ncols;

  int isSorted = checkIfUISorted(mat);
  if (!isSorted) {
    std::cerr << "\nwriteItemJaccSimMat: matrix not sorted" << std::endl;
    exit(0);
  }

  std::vector<std::vector<float>> jacSim(nItems, std::vector<float>(nItems, 0.0));
  std::cout << "\nComputing jaccard similarities..." << std::endl;

#pragma omp parallel for
  for (int item1 = 0; item1 < nItems; item1++) {
    for (int item2 = item1+1; item2 < nItems; item2++) {
      //find number of users who corated item1 and item2 
      float nCoRatedUsers = (float)coRatedUsersFrmSortedMatLinMerge(mat, item1, item2);
      float nUnionUsers = itemFreq[item1] + itemFreq[item2] - nCoRatedUsers; 
      float sim = 0;
      if (nUnionUsers > 0) {
        sim = nCoRatedUsers/nUnionUsers;
      }
      jacSim[item1][item2] = sim;
      jacSim[item2][item1] = jacSim[item1][item2];
    }
    if (item1 % 1000 == 0) {
      std::cout << "\nDone... items " << item1 << std::endl;
    }
  }
  
  std::ofstream opFile(fName);
  std::cout << "\nWriting Jaccard sim mat... " << fName << std::endl;
  for (int item1 = 0; item1 < nItems; item1++) {
    for (int item2 = 0; item2 < nItems; item2++) {
      if (item2 != item1 && jacSim[item1][item2] > EPS) {
        opFile << item2 << " " << jacSim[item1][item2] << " ";
      }
    }
    opFile << std::endl;
  }
  opFile.close();

}


void writeItemJaccSimMatPar(gk_csr_t *mat, const char *fName) {
  auto rowColFreq = getRowColFreq(mat);
  auto userFreq = rowColFreq.first;
  auto itemFreq = rowColFreq.second;
  int nItems = mat->ncols;

  int isSorted = checkIfUISorted(mat);
  if (!isSorted) {
    std::cerr << "\nwriteItemJaccSimMat: matrix not sorted" << std::endl;
    exit(0);
  }

  std::cout << "\nComputing jaccard similarities..." << std::endl;
  std::ofstream opFile(fName, std::ios::binary);
  int maxThreads = omp_get_max_threads();
  std::cout << "Max threads: " << maxThreads << std::endl;
  std::vector<std::vector<float>> jacSim(maxThreads, std::vector<float>(nItems, 0.0));

  for (int item1 = 0; item1 < nItems; item1+=maxThreads) {
    
    for (auto&& v: jacSim) {
      v.clear();
    }

#pragma omp parallel for collapse(2)
    for (int j = 0; j < maxThreads; j++) {
      for (int item2 = 0; item2 < nItems; item2++) {
        int itemA = item1+j; 
        if (itemA >= nItems) {
          continue;
        }
        //find number of users who corated item1 and item2 
        float nCoRatedUsers = (float)coRatedUsersFrmSortedMatLinMerge(mat, itemA, item2);
        float nUnionUsers = itemFreq[itemA] + itemFreq[item2] - nCoRatedUsers; 
        float sim = 0;
        if (nUnionUsers > 0) {
          sim = nCoRatedUsers/nUnionUsers;
        }
        jacSim[j][item2] = sim; 
      }
    }

    if (item1 % 1000 == 0) {
      std::cout << "\nDone... items " << item1 << std::endl;
    }
   
    float temp;
    for (int j = 0; j < maxThreads; j++) {
      int itemA = item1+j; 
      if (itemA >= nItems) {
        continue;
      }
      for (int item2 = 0; item2 < nItems; item2++) {
        int count = 0;
        if (item2 != itemA && jacSim[j][item2] > EPS) {
          count++;
        }
        opFile.write((char*) &count, sizeof(int));
        if (item2 != itemA && jacSim[j][item2] > EPS) {
          opFile.write((char*) &item2, sizeof(int));
          temp = jacSim[j][item2];
          opFile.write((char*) &temp, sizeof(float));
        }
      }
    }

  }

  opFile.close();
}


void writeItemJaccSimFrmCorat(gk_csr_t *mat, gk_csr_t *coRatMat, const char *fName) {
  auto rowColFreq = getRowColFreq(mat);
  auto userFreq = rowColFreq.first;
  auto itemFreq = rowColFreq.second;
  int nItems = mat->ncols;

  float nCoRatedUsers = 0, nUnionUsers = 0, sim = 0;
  std::vector<std::vector<float>> jacSim(nItems, std::vector<float>(nItems, 0.0));
  std::cout << "\nComputing jaccard similarities..." << std::endl;
  for (int item1 = 0; item1 < nItems; item1++) {
    for (int ii = coRatMat->rowptr[item1]; ii < coRatMat->rowptr[item1+1]; 
        ii++) {
      int item2 = coRatMat->rowind[ii];
      nCoRatedUsers = coRatMat->rowval[ii];
      nUnionUsers = itemFreq[item1] + itemFreq[item2] - nCoRatedUsers; 
      sim = 0;
      if (nUnionUsers > 0) {
        sim = nCoRatedUsers/nUnionUsers;
      }
      jacSim[item1][item2] = sim;
      jacSim[item2][item1] = jacSim[item1][item2];
    }
    if (item1 % 1000 == 0) {
      std::cout << "\nDone... items " << item1 << std::endl;
    }
  }
  
  std::ofstream opFile(fName);
  std::cout << "\nWriting Jaccard sim mat... " << fName << std::endl;
  for (int item1 = 0; item1 < nItems; item1++) {
    for (int item2 = 0; item2 < nItems; item2++) {
      if (item2 != item1 && jacSim[item1][item2] > EPS) {
        opFile << item2 << " " << jacSim[item1][item2] << " ";
      }
    }
    opFile << std::endl;
  }
  opFile.close();

}


void writeCoRatings(gk_csr_t *mat, const char *fName) {
  auto rowColFreq = getRowColFreq(mat);
  auto userFreq = rowColFreq.first;
  auto itemFreq = rowColFreq.second;
  int nItems = mat->ncols;
  
  int isSorted = checkIfUISorted(mat);
  if (!isSorted) {
    std::cerr << "\nwriteCoRatings: matrix not sorted" << std::endl;
    exit(0);
  }

  float nCoRatedUsers = 0;
  std::vector<std::vector<float>> jacSim(nItems, std::vector<float>(nItems, 0.0));
  std::cout << "\nComputing coratings..." << std::endl;
  for (int item1 = 0; item1 < nItems; item1++) {
    for (int item2 = item1+1; item2 < nItems; item2++) {
      //find number of users who corated item1 and item2 
      nCoRatedUsers = (float)coRatedUsersFrmSortedMatLinMerge(mat, item1, item2);
      jacSim[item1][item2] = nCoRatedUsers;
      jacSim[item2][item1] = jacSim[item1][item2];
    }
    if (item1 % 100 == 0) {
      std::cout << "\nDone... items " << item1 << std::endl;
    }
  }
  
  std::ofstream opFile(fName);
  std::cout << "\nWriting co rating mat... " << fName << std::endl;
  for (int item1 = 0; item1 < nItems; item1++) {
    for (int item2 = 0; item2 < nItems; item2++) {
      if (item2 != item1 && jacSim[item1][item2] > EPS) {
        opFile << item2 << " " << jacSim[item1][item2] << " ";
      }
    }
    opFile << std::endl;
  }
  opFile.close();

}


void writeTriplets(gk_csr_t *mat, const char *fName) {
  std::ofstream opFile(fName);
  for (int u = 0; u < mat->nrows; u++) {
    for (int ii = mat->rowptr[u]; ii < mat->rowptr[u+1]; ii++) {
      opFile << u << " " << mat->rowind[ii] << " " << mat->rowval[ii] << std::endl;
    }
  }
  opFile.close();
}


