#include "datastruct.h"

Data::Data(const Params& params) {
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
          << params.trainMatFile << std::endl;
        trainMat = gk_csr_Read(params.trainMatFile, GK_CSR_FMT_CSR, GK_CSR_IS_VAL, 0);
        //trainMat = gk_csr_Read(params.trainMatFile, GK_CSR_FMT_CSR, 1, 0);
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
        //nUsers = trainMat->nrows;
        //nItems = trainMat->ncols;
        //exit(0);
      }

      testMat = NULL;
      if (NULL != params.testMatFile) {
        std::cout << "\nReading test matrix 0-indexed... " 
          << params.testMatFile;
        testMat = gk_csr_Read(params.testMatFile, GK_CSR_FMT_CSR, GK_CSR_IS_VAL, 0);
        //testMat = gk_csr_Read(params.testMatFile, GK_CSR_FMT_CSR, 1, 0);
        gk_csr_CreateIndex(testMat, GK_CSR_COL);
      }
      
      valMat = NULL;
      if (NULL != params.valMatFile) {
        std::cout << "\nReading val matrix 0-indexed... " 
          << params.valMatFile;
        valMat = gk_csr_Read(params.valMatFile, GK_CSR_FMT_CSR, GK_CSR_IS_VAL, 0);
        //valMat = gk_csr_Read(params.valMatFile, GK_CSR_FMT_CSR, 1, 0);
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

