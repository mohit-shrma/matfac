#ifndef _DATASTRUCT_H_
#define _DATASTRUCT_H_

#include <iostream>
#include <vector>
#include "GKlib.h"

class Data {

  public:
    gk_csr_t *trainMat;
    int trainNNZ;
    gk_csr_t *testMat;
    
    Data(char *trainMatFile, char *testMatFile) {
      std::cout << "\nReading partial matrix 0-indexed... ";
      trainMat = gk_csr_Read(trainMatFile, GK_CSR_FMT_CSR, 1, 0);
      gk_csr_CreateIndex(trainMat, GK_CSR_COL);
      
      //get nnz in train matrix
      trainNNZ = 0;
      for (int u = 0; u < trainMat->nrows; u++) {
        trainNNZ += trainMat->rowptr[u+1] - trainMat->rowptr[u];
      }

      std::cout << "\nReading full matrix 0-indexed... ";
      testMat = gk_csr_Read(testMatFile, GK_CSR_FMT_CSR, 1, 0);
      gk_csr_CreateIndex(testMat, GK_CSR_COL);
    }

    ~Data() {
      gk_csr_Free(&trainMat);
      gk_csr_Free(&testMat);
    }

};

class Params {
  
  public:
    int nUsers;
    int nItems;
    int facDim;
    int maxIter;
    float uReg;
    float iReg;
    float learnRate;
    float rhoRMS;
    float alpha;
    char *trainMatFile;
    char *testMatFile;
    
    /*
    Params(char *p_partialMatFile, char *p_testMatFile, float p_uReg, 
        float p_iReg, int p_facDim, float p_learnRate) {
      partialMatFile = p_partialMatFile;
      testMatFile = p_testMatFile;
      uReg = p_uReg;
      iReg = p_iReg;
      facDim = p_facDim;
      learnRate = p_learnRate;
    }
    */

    Params(int p_nUsers, int p_nItems, int p_facDim, int p_maxIter, 
        float p_uReg, float p_iReg,  float p_learnRate, float p_rhoRMS, float p_alpha,
        char *p_trainMatFile, char *p_testMatFile)
      : nUsers(p_nUsers), nItems(p_nItems), facDim(p_facDim), maxIter(p_maxIter),
      uReg(p_uReg), iReg(p_iReg), learnRate(p_learnRate), rhoRMS(p_rhoRMS), alpha(p_alpha),
      trainMatFile(p_trainMatFile), testMatFile(p_testMatFile){}


};

#endif
