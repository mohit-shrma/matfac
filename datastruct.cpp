#include "datastruct.h"

Data::Data(const Params& params) {
      facDim = params.facDim;
      nUsers = -1;
      nItems = -1;
      prefix = params.prefix;

      
      int minItemInd = -1, maxItemInd = -1;

      trainMat = NULL;
      if (NULL != params.trainMatFile) {
        std::cout << "\nReading partial train matrix 0-indexed... " 
          << params.trainMatFile << std::endl;
        trainMat = gk_csr_Read((char*) params.trainMatFile, GK_CSR_FMT_CSR, GK_CSR_IS_VAL, 0);
        //trainMat = gk_csr_Read(params.trainMatFile, GK_CSR_FMT_CSR, 1, 0);
        gk_csr_CreateIndex(trainMat, GK_CSR_COL);
        //get nnz in train matrix
        trainNNZ = 0;
        minItemInd = 0;
        maxItemInd = trainMat->ncols-1;
        nUsers = trainMat->nrows;
#pragma omp parallel for reduction(max: maxItemInd), reduction(min: minItemInd)
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
      }
      
      std::cout <<"\ntrain nnz = " << trainNNZ << std::endl;
      std::cout <<"train nrows: " << trainMat->nrows << " ncols: " 
        << trainMat->ncols  << std::endl;
      std::cout << "minItemInd: " << minItemInd << " maxItemInd: " 
        << maxItemInd << std::endl; 

      testMat = NULL;
      if (NULL != params.testMatFile) {
        std::cout << "Reading test matrix 0-indexed... " << params.testMatFile << std::endl;
        testMat = gk_csr_Read((char*) params.testMatFile, GK_CSR_FMT_CSR, GK_CSR_IS_VAL, 0);
        //testMat = gk_csr_Read(params.testMatFile, GK_CSR_FMT_CSR, 1, 0);
        gk_csr_CreateIndex(testMat, GK_CSR_COL);
#pragma omp parallel for reduction(max: maxItemInd), reduction(min: minItemInd)
        for (int u = 0; u < testMat->nrows; u++) {
          for (int ii = testMat->rowptr[u]; ii < testMat->rowptr[u+1]; ii++) {
            int item = testMat->rowind[ii];
            if (item < minItemInd) {
              minItemInd = item;
            }
            if (item > maxItemInd) {
              maxItemInd = item;
            }
          } 
        }
      }
      
      std::cout << "minItemInd: " << minItemInd << " maxItemInd: " 
        << maxItemInd << std::endl; 
      
      valMat = NULL;
      if (NULL != params.valMatFile) {
        std::cout << "Reading val matrix 0-indexed... " << params.valMatFile << std::endl;
        valMat = gk_csr_Read((char*) params.valMatFile, GK_CSR_FMT_CSR, GK_CSR_IS_VAL, 0);
        //valMat = gk_csr_Read(params.valMatFile, GK_CSR_FMT_CSR, 1, 0);
        gk_csr_CreateIndex(valMat, GK_CSR_COL);
#pragma omp parallel for reduction(max: maxItemInd), reduction(min: minItemInd)
        for (int u = 0; u < valMat->nrows; u++) {
          for (int ii = valMat->rowptr[u]; ii < valMat->rowptr[u+1]; ii++) {
            int item = valMat->rowind[ii];
            if (item < minItemInd) {
              minItemInd = item;
            }
            if (item > maxItemInd) {
              maxItemInd = item;
            }
          } 
        }
      }
      
      std::cout << "minItemInd: " << minItemInd << " maxItemInd: " 
        << maxItemInd << std::endl; 
      nItems = maxItemInd + 1;

      graphMat = NULL;
      if (NULL != params.graphMatFile) {
        if (isFileExist(params.graphMatFile)) { 
        std::cout << "\nReading graph mat file... 0-indexed w val " 
          << params.graphMatFile;
        graphMat = gk_csr_Read((char*) params.graphMatFile, GK_CSR_FMT_CSR, 1, 0);
        std::cout << "\ngraph nrows: " << graphMat->nrows << " ncols: " 
          << graphMat->ncols;
        } else {
          std::cerr << "\nFile dont exists: " << params.graphMatFile << std::endl;
        }
      }

      negMat = NULL;
      if (NULL != params.negMatFile) {
        if (isFileExist(params.negMatFile)) { 
        std::cout << "\nReading neg mat file... 0-indexed w val " 
          << params.negMatFile;
        negMat = gk_csr_Read((char*) params.negMatFile, GK_CSR_FMT_CSR, 1, 0);
        std::cout << "\nnegatives nrows: " << negMat->nrows << " ncols: " 
          << negMat->ncols;
        } else {
          std::cerr << "\nFile dont exists: " << params.negMatFile << std::endl;
        }
      }

      if (facDim > 0) {
        if (NULL != params.origUFacFile) {
          origUFac.assign(nUsers, std::vector<double>(facDim, 0));
          readMat(origUFac, nUsers, facDim, params.origUFacFile);      
        }
        
        if (NULL != params.origIFacFile) {
          origIFac.assign(nItems, std::vector<double>(facDim, 0));
          readMat(origIFac, nItems, facDim, params.origIFacFile);
        }
        
      }


}

