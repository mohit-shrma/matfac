#ifndef _DATASTRUCT_H_
#define _DATASTRUCT_H_


#include <iostream>
#include <vector>
#include <numeric>
#include "GKlib.h"
#include "io.h"
#include "const.h"

class Params {
  
  public:
    int nUsers;
    int nItems;
    int facDim;
    int maxIter;
    int svdFacDim;
    int seed;
    float uReg;
    float iReg;
    float learnRate;
    float rhoRMS;
    float alpha;
    const char *trainMatFile;
    const char *testMatFile;
    const char *valMatFile;
    const char *graphMatFile;
    const char *origUFacFile;
    const char *origIFacFile;
    const char *initUFacFile;
    const char *initIFacFile;
    const char *prefix;

    Params(int facDim, int maxIter, int svdFacDim, int seed,
        float uReg, float iReg,  float learnRate, float rhoRMS, 
        float alpha, std::string trainMatFile, std::string testMatFile, std::string valMatFile,
        std::string graphMatFile, std::string origUFacFile, std::string origIFacFile, 
        std::string initUFacFile, std::string initIFacFile, std::string prefix)
      : nUsers(-1), nItems(-1), facDim(facDim), maxIter(maxIter), svdFacDim(svdFacDim), seed(seed),
      uReg(uReg), iReg(iReg), learnRate(learnRate), rhoRMS(rhoRMS), 
      alpha(alpha), trainMatFile(trainMatFile.c_str()), testMatFile(testMatFile.c_str()), 
      valMatFile(valMatFile.c_str()), 
      graphMatFile(graphMatFile.empty()?NULL:graphMatFile.c_str()),
      origUFacFile(origUFacFile.empty()?NULL:origUFacFile.c_str()), 
      origIFacFile(origIFacFile.empty()?NULL:origIFacFile.c_str()), 
      initUFacFile(initUFacFile.empty()?NULL:initUFacFile.c_str()), 
      initIFacFile(initIFacFile.empty()?NULL:initIFacFile.c_str()),
      prefix(prefix.c_str()){}

    void display() {
      std::cout << "*** PARAMETERS ***" << std::endl;
      std::cout << "nUsers: " << nUsers << " nItems: " << nItems << std::endl;
      std::cout << "facDim: " << facDim << " svdFacDim: " << svdFacDim << std::endl;
      std::cout << "maxIter: " << maxIter << std::endl;
      std::cout << "uReg: " << uReg << " iReg: " << iReg << std::endl;
      std::cout << "learnRate: " << learnRate << std::endl;
      std::cout << "trainMat: " << trainMatFile << std::endl;
      std::cout << "testMat: " << testMatFile << std::endl;
      std::cout << "valMat: " << valMatFile << std::endl;
      std::cout << "graphMat: " << (NULL != graphMatFile? graphMatFile : " ") << std::endl;
      std::cout << "origUFac: " << (NULL != origUFacFile? origUFacFile : " ") << std::endl;
      std::cout << "origIFac: " << (NULL != origIFacFile? origIFacFile : " ") << std::endl;
      std::cout << "initUFac: " << (NULL != initUFacFile? initUFacFile : " ") << std::endl;
      std::cout << "initIFac: " << (NULL != initIFacFile? initIFacFile : " ") << std::endl;
    }
}; 


 class Data {

  public:
    const char *prefix;

    gk_csr_t *trainMat;
    gk_csr_t *testMat;
    gk_csr_t *valMat;
    gk_csr_t *graphMat;

    std::vector<std::vector<double>> origUFac;
    std::vector<std::vector<double>> origIFac;
    
    int facDim;
    int trainNNZ;
    int nUsers;
    int nItems;

    double meanKnownSubMatRat(int uStart, int uEnd, int iStart, int iEnd) {

      int u, item;
      double rmse = 0;

      for (u = iStart; u <= uEnd; u++) {
        for (item = iStart; item <= iEnd; item++) {
          rmse += std::inner_product(origUFac[u].begin(), origUFac[u].end(),
                                     origIFac[item].begin(), 0.0);
        }
      }
      
      rmse = sqrt(rmse/((uEnd-uStart+1)*(iEnd-iStart+1)));
      return rmse;
    }

    
    Data(gk_csr_t *p_trainMat, gk_csr_t *p_testMat) {
      trainMat   = p_trainMat;
      testMat    = p_testMat;
      nUsers     = trainMat->nrows;
      nItems     = trainMat->ncols;
    }
    
    Data(const Params& params); 

    ~Data() {
      
      if (NULL != trainMat) {
        gk_csr_Free(&trainMat);
      }

      if (NULL != testMat) {
        gk_csr_Free(&testMat);
      }
      
      if (NULL != valMat) {
        gk_csr_Free(&valMat);
      }

      if (NULL != graphMat) {
        gk_csr_Free(&graphMat);
      }

    }

};


#endif
