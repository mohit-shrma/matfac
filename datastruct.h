#ifndef _DATASTRUCT_H_
#define _DATASTRUCT_H_

#include <iostream>
#include <vector>
#include "GKlib.h"
#include "io.h"

class Params {
  
  public:
    int nUsers;
    int nItems;
    int facDim;
    int maxIter;
    int origFacDim;
    float uReg;
    float iReg;
    float learnRate;
    float rhoRMS;
    float alpha;
    char *trainMatFile;
    char *testMatFile;
    char *origUFacFile;
    char *origIFacFile;

    Params(int p_nUsers, int p_nItems, int p_facDim, int p_maxIter, 
        int p_origFacDim,
        float p_uReg, float p_iReg,  float p_learnRate, float p_rhoRMS, float p_alpha,
        char *p_trainMatFile, char *p_testMatFile, char *p_origUFacFile, 
        char *p_origIFacFile)
      : nUsers(p_nUsers), nItems(p_nItems), facDim(p_facDim), maxIter(p_maxIter),
      origFacDim(p_origFacDim),
      uReg(p_uReg), iReg(p_iReg), learnRate(p_learnRate), rhoRMS(p_rhoRMS), alpha(p_alpha),
      trainMatFile(p_trainMatFile), testMatFile(p_testMatFile), 
      origUFacFile(p_origUFacFile), origIFacFile(p_origIFacFile){}


};

class Data {

  public:
    gk_csr_t *trainMat;
    gk_csr_t *testMat;
    
    std::vector<std::vector<double>> origUFac;
    std::vector<std::vector<double>> origIFac;
    
    int origFacDim;
    int trainNNZ;
    int nUsers;
    int nItems;

    Data(const Params& params) {
      origFacDim = params.origFacDim;
      nUsers = params.nUsers;
      nItems = params.nItems;

      if (origFacDim > 0) {
        if (NULL != params.origUFacFile) {
          origUFac.assign(nUsers, std::vector<double>(origFacDim, 0));
          readMat(origUFac, nUsers, origFacDim, params.origUFacFile);      
          writeMat(origUFac, nUsers, origFacDim, "readUFac.txt");
        }
        
        if (NULL != params.origIFacFile) {
          origIFac.assign(nItems, std::vector<double>(origFacDim, 0));
          readMat(origIFac, nItems, origFacDim, params.origIFacFile);
          writeMat(origIFac, nItems, origFacDim, "readIFac.txt");
        }

        //TODO: verify whether correct fators are read
        
      }

      std::cout << "\nReading partial train matrix 0-indexed... ";
      if (NULL != params.trainMatFile) {
        trainMat = gk_csr_Read(params.trainMatFile, GK_CSR_FMT_CSR, 1, 0);
        gk_csr_CreateIndex(trainMat, GK_CSR_COL);
      }
      
      //get nnz in train matrix
      trainNNZ = 0;
      for (int u = 0; u < trainMat->nrows; u++) {
        trainNNZ += trainMat->rowptr[u+1] - trainMat->rowptr[u];
      }
      std::cout<<"\ntrain nnz = " << trainNNZ;
      std::cout << "\nReading test matrix 0-indexed... ";
      if (NULL != params.testMatFile) {
        testMat = gk_csr_Read(params.testMatFile, GK_CSR_FMT_CSR, 1, 0);
        gk_csr_CreateIndex(testMat, GK_CSR_COL);
      }

    }

    ~Data() {
      gk_csr_Free(&trainMat);
      gk_csr_Free(&testMat);
    }

};


#endif
