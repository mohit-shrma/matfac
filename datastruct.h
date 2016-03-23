#ifndef _DATASTRUCT_H_
#define _DATASTRUCT_H_

#include <iostream>
#include <vector>
#include <numeric>
#include "GKlib.h"
#include "io.h"


class Params {
  
  public:
    int nUsers;
    int nItems;
    int facDim;
    int maxIter;
    int origFacDim;
    int gluSz;
    int seed;
    float uReg;
    float iReg;
    float learnRate;
    float rhoRMS;
    float alpha;
    char *trainMatFile;
    char *testMatFile;
    char *valMatFile;
    char *graphMatFile;
    char *origUFacFile;
    char *origIFacFile;
    char *initUFacFile;
    char *initIFacFile;
    char *prefix;

    Params(int p_nUsers, int p_nItems, int p_facDim, int p_maxIter, 
        int p_origFacDim, int p_seed,
        float p_uReg, float p_iReg,  float p_learnRate, float p_rhoRMS, 
        float p_alpha,
        char *p_trainMatFile, char *p_testMatFile, char *p_valMatFile,
        char *p_graphMatFile,
        char *p_origUFacFile, 
        char *p_origIFacFile, char *p_initUFacFile, char *p_initIFacFile,
        char *p_prefix)
      : nUsers(p_nUsers), nItems(p_nItems), facDim(p_facDim), maxIter(p_maxIter),
      origFacDim(p_origFacDim), seed(p_seed),
      uReg(p_uReg), iReg(p_iReg), learnRate(p_learnRate), rhoRMS(p_rhoRMS), 
      alpha(p_alpha),
      trainMatFile(p_trainMatFile), testMatFile(p_testMatFile), 
      valMatFile(p_valMatFile), graphMatFile(p_graphMatFile),
      origUFacFile(p_origUFacFile), origIFacFile(p_origIFacFile), 
      initUFacFile(p_initUFacFile), initIFacFile(p_initIFacFile),
      prefix(p_prefix){}


}; 


 class Data {

  public:
    char *prefix;

    gk_csr_t *trainMat;
    gk_csr_t *testMat;
    gk_csr_t *valMat;
    gk_csr_t *graphMat;

    std::vector<std::vector<double>> origUFac;
    std::vector<std::vector<double>> origIFac;
    
    int origFacDim;
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
      origFacDim = 0;
    }
    

    Data(const Params& params) {
      origFacDim = params.origFacDim;
      nUsers = params.nUsers;
      nItems = params.nItems;
      prefix = params.prefix;

      if (origFacDim > 0) {
        if (NULL != params.origUFacFile) {
          origUFac.assign(nUsers, std::vector<double>(origFacDim, 0));
          readMat(origUFac, nUsers, origFacDim, params.origUFacFile);      
          //writeMat(origUFac, nUsers, origFacDim, "readUFac.txt");
        }
        
        if (NULL != params.origIFacFile) {
          origIFac.assign(nItems, std::vector<double>(origFacDim, 0));
          readMat(origIFac, nItems, origFacDim, params.origIFacFile);
          //writeMat(origIFac, nItems, origFacDim, "readIFac.txt");
        }
        
      }

      trainMat = NULL;
      if (NULL != params.trainMatFile) {
        std::cout << "\nReading partial train matrix 0-indexed... " 
          << params.trainMatFile;
        trainMat = gk_csr_Read(params.trainMatFile, GK_CSR_FMT_CSR, 1, 0);
        gk_csr_CreateIndex(trainMat, GK_CSR_COL);
        //get nnz in train matrix
        trainNNZ = 0;
        int minItemInd = trainMat->ncols;
        int maxItemInd = -1;
        for (int u = 0; u < trainMat->nrows; u++) {
          trainNNZ += trainMat->rowptr[u+1] - trainMat->rowptr[u];
          for (int ii = trainMat->rowptr[u]; ii < trainMat->rowptr[u+1];
              ii++) {
            int item = trainMat->rowind[ii];
            if (item < minItemInd) {
              minItemInd = item;
            }
            if (item > maxItemInd) {
              maxItemInd = item;
            }
          }
        }
        std::cout << "\nminItem: " << minItemInd << " maxItem: " <<  maxItemInd;
      }
      
      std::cout <<"\ntrain nnz = " << trainNNZ;
      std::cout <<"\ntrain nrows: " << trainMat->nrows << " ncols: " << trainMat->ncols;
      
      if (trainMat->nrows != nUsers || trainMat->ncols != nItems) {
        std::cout << "\n!!passed parameter of nUsers and nItems dont match!!"
            << std:: endl;
        nUsers = trainMat->nrows;
        nItems = trainMat->ncols;
        exit(0);
      }

      testMat = NULL;
      if (NULL != params.testMatFile) {
        std::cout << "\nReading test matrix 0-indexed... " 
          << params.testMatFile;
        testMat = gk_csr_Read(params.testMatFile, GK_CSR_FMT_CSR, 1, 0);
        gk_csr_CreateIndex(testMat, GK_CSR_COL);
      }
      
      valMat = NULL;
      if (NULL != params.valMatFile) {
        std::cout << "\nReading val matrix 0-indexed... " 
          << params.valMatFile;
        valMat = gk_csr_Read(params.valMatFile, GK_CSR_FMT_CSR, 1, 0);
        gk_csr_CreateIndex(valMat, GK_CSR_COL);
      }

      graphMat = NULL;
      if (NULL != params.graphMatFile) {
        if (isFileExist(params.graphMatFile)) { 
        std::cout << "\nReading graph mat file... 0-indexed w val " 
          << params.graphMatFile;
        graphMat = gk_csr_Read(params.graphMatFile, GK_CSR_FMT_CSR, 1, 0);
        std::cout << "\ngraph nrows: " << graphMat->nrows << " ncols: " 
          << graphMat->ncols;
        } else {
          std::cerr << "\nFile dont exists: " << params.graphMatFile << std::endl;
        }
      }
    }


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
