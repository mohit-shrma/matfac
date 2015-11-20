#include <iostream>


class Data {

  public:
    gk_csr_t *trainMat;
    gk_csr_t *fullMat;
    
    Data(char *trainMatFile, char *fullMatFile) {
      std::cout << "\nReading partial matrix 0-indexed... ";
      trainMat = gk_csr_Read(trainMatFile, GK_CSR_FMT_CSR, 1, 0);
      gk_csr_CreateIndex(trainMat, GK_CSR_COL);
      std::cout << "\nReading full matrix 0-indexed... ";
      fullMat = gk_csr_Read(fullMatFile, GK_CSR_FMT_CSR, 1, 0);
      gk_csr_CreateIndex(fullMat, GK_CSR_COL);
    }

    ~Data() {
      gk_csr_Free(&trainMat);
      gk_csr_Free(&fullMat);
    }

}


class Params {
  
  public:
    int nUsers;
    int nItems;
    char *trainMatFile;
    char *fullMatFile;
    float uReg;
    float iReg;
    int facDim;
    float learnRate;
    float rhoRMS;
    float maxIter;
    
    /*
    Params(char *p_partialMatFile, char *p_fullMatFile, float p_uReg, 
        float p_iReg, int p_facDim, float p_learnRate) {
      partialMatFile = p_partialMatFile;
      fullMatFile = p_fullMatFile;
      uReg = p_uReg;
      iReg = p_iReg;
      facDim = p_facDim;
      learnRate = p_learnRate;
    }
    */

    Params(int p_nUsers, int p_nItems, char *p_trainMatFile, 
        char *p_fullMatFile, float p_rhoRMS, int p_maxIter, float p_uReg, 
        float p_iReg, int p_facDim, float p_learnRate)
      : nUsers(p_nUsers), nItems(p_nItems), trainMatFile(p_trainMatFile), 
      fullMatFile(p_fullMatFile), uReg(p_uReg), iReg(p_iReg), facDim(p_facDim), 
      learnRate(p_learnRate), rhoRMS(p_rhoRMS), learnRate(p_learnRate) {}

    ~Params() {}

}





