#include "modelMF.h"


void sgdUpdateBlockSeq(MatrixXb& bMask, 
    int dim, std::vector<std::pair<int, int>>& updateSeq, std::mt19937& mt) {
  
  bMask.fill(false);
  updateSeq.clear();
  std::uniform_int_distribution<int> dis(0, dim-1);
  std::vector<bool> rowMask(dim, false);
  std::vector<int> rowInds(dim);
  std::iota(rowInds.begin(), rowInds.end(), 0);
  std::shuffle(rowInds.begin(), rowInds.end(), mt);

  for (int ind = 0; ind < dim; ind++) {
    int currRow = rowInds[ind];
    int currCol = dis(mt);
    while (true) {
      if (bMask.col(currCol).any()) {
        currCol = dis(mt);
      } else{
        bMask(currRow, currCol) = true;
        updateSeq.push_back(std::make_pair(currRow, currCol));
        break;
      }
    }
  }
  
}


void ModelMF::train(const Data &data, Model &bestModel, 
    std::unordered_set<int>& invalidUsers,
    std::unordered_set<int>& invalidItems) {

  //copy passed known factors
  //uFac = data.origUFac;
  //iFac = data.origIFac;
  
  std::cout << "\nModelMF::train trainSeed: " << trainSeed;
  
  int nnz = data.trainNNZ;
  
  std::cout << "\nObj b4 svd: " << objective(data) 
    << " Train RMSE: " << RMSE(data.trainMat) 
    << " Train nnz: " << nnz << std::endl;
  
  std::chrono::time_point<std::chrono::system_clock> startSVD, endSVD;
  startSVD = std::chrono::system_clock::now();
  //initialization with svd of the passed matrix
  //svdFrmSvdlibCSR(data.trainMat, facDim, uFac, iFac, false); 
  
  endSVD = std::chrono::system_clock::now();
  std::chrono::duration<double> durationSVD =  (endSVD - startSVD) ;
  std::cout << "\nsvd duration: " << durationSVD.count();

  int u, item, iter, bestIter = -1, i; 
  float itemRat;
  double diff, r_ui_est;
  double bestObj, prevObj;
  double bestValRMSE, prevValRMSE;

  gk_csr_t *trainMat = data.trainMat;

  std::vector<std::unordered_set<int>> uISet(nUsers);
  genStats(trainMat, uISet, std::to_string(trainSeed));
  getInvalidUsersItems(trainMat, uISet, invalidUsers, invalidItems);
  for (int u = trainMat->nrows; u < data.nUsers; u++) {
    invalidUsers.insert(u);
  }
  for (int item = trainMat->ncols; item < data.nItems; item++) {
    invalidItems.insert(item);
  }

  //std::cout << "\nNNZ = " << nnz;
  prevObj = objective(data, invalidUsers, invalidItems);
  bestObj = prevObj;
  bestValRMSE = prevValRMSE = RMSE(data.valMat, invalidUsers, invalidItems);
  std::cout << "\nObj aftr svd: " << prevObj << " Train RMSE: " 
    << RMSE(data.trainMat, invalidUsers, invalidItems) << " Val RMSE: " 
    << bestValRMSE;

  std::chrono::time_point<std::chrono::system_clock> start, end;
  std::chrono::duration<double> duration;
  
 std::cout << "\nModelMF::train trainSeed: " << trainSeed 
    << " invalidUsers: " << invalidUsers.size()
    << " invalidItems: " << invalidItems.size() << std::endl;
  
  //random engine
  std::mt19937 mt(trainSeed);
  //get user-item ratings from training data
  const auto uiRatings = getUIRatings(trainMat, invalidUsers, invalidItems);
  //index to above uiRatings pair
  std::vector<size_t> uiRatingInds(uiRatings.size());
  std::iota(uiRatingInds.begin(), uiRatingInds.end(), 0);


  std::cout << "\nTrain NNZ after removing invalid users and items: " 
    << uiRatings.size() << std::endl;
  double subIterDuration = 0;
  for (iter = 0; iter < maxIter; iter++) {  
    
    start = std::chrono::system_clock::now();
    if (iter % 10 == 0) {
      //shuffle the user item rating indexes
      std::shuffle(uiRatingInds.begin(), uiRatingInds.end(), mt);
    } else {
      parBlockShuffle(uiRatingInds, mt);
    }

    for (const auto& ind: uiRatingInds) {
      //get user, item and rating
      u       = std::get<0>(uiRatings[ind]);
      item    = std::get<1>(uiRatings[ind]);
      itemRat = std::get<2>(uiRatings[ind]);
      
      //std::cout << "\nGradCheck u: " << u << " item: " << item;
      //gradCheck(u, item, itemRat);
      r_ui_est = uFac.row(u).dot(iFac.row(item));
      diff = itemRat - r_ui_est;

      //update user
      for (int i = 0; i < facDim; i++) {
        uFac(u, i) -= learnRate * (-2.0*diff*iFac(item, i) + 2.0*uReg*uFac(u, i));
      }

    
      //update item
      for (int i = 0; i < facDim; i++) {
        iFac(item, i) -= learnRate * (-2.0*diff*uFac(u, i) + 2.0*iReg*iFac(item, i));
      }

    }
    end = std::chrono::system_clock::now();  
   
    duration =  end - start;
    subIterDuration = duration.count();

    //check objective
    if (iter % OBJ_ITER == 0 || iter == maxIter-1) {
      
      if (GK_CSR_IS_VAL) {
        if (isTerminateModel(bestModel, data, iter, bestIter, bestObj, prevObj,
              bestValRMSE, prevValRMSE, invalidUsers, invalidItems)) {
          break; 
        }
      } else {
        if (isTerminateModel(bestModel, data, iter, bestIter, bestObj, prevObj,
              invalidUsers, invalidItems)) {
          break; 
        }
      }
       
      if (iter % DISP_ITER == 0) {
        std::cout << "ModelMF::train trainSeed: " << trainSeed
                  << " Iter: " << iter << " Objective: " << std::scientific << prevObj 
                  << " Train RMSE: " << RMSE(data.trainMat, invalidUsers, invalidItems)
                  << " Val RMSE: " << prevValRMSE
                  << " subIterDuration: " << subIterDuration
                  << std::endl;
      }

      if (iter % SAVE_ITER == 0 || iter == maxIter - 1) {
        std::string modelFName = std::string(data.prefix);
        bestModel.saveFacs(modelFName);
      }

    }
     
  }
      
  //save best model found till now
  std::string modelFName = std::string(data.prefix);
  bestModel.saveFacs(modelFName);

  std::cout << "\nBest model validation RMSE: " << bestModel.RMSE(data.valMat, 
      invalidUsers, invalidItems);

}


void ModelMF::trainSGDPar(const Data &data, Model &bestModel, 
    std::unordered_set<int>& invalidUsers,
    std::unordered_set<int>& invalidItems) {

  //copy passed known factors
  //uFac = data.origUFac;
  //iFac = data.origIFac;
  
  std::cout << "\nModelMF::trainSGDPar trainSeed: " << trainSeed;
  
  int nnz = data.trainNNZ;
  
  std::cout << "\nObj b4 svd: " << objective(data) 
    << " Train RMSE: " << RMSE(data.trainMat) 
    << " Train nnz: " << nnz << std::endl;
  
  std::chrono::time_point<std::chrono::system_clock> startSVD, endSVD;
  startSVD = std::chrono::system_clock::now();
  //initialization with svd of the passed matrix
  //svdFrmSvdlibCSR(data.trainMat, facDim, uFac, iFac, false); 
  
  endSVD = std::chrono::system_clock::now();
  std::chrono::duration<double> durationSVD =  (endSVD - startSVD) ;
  std::cout << "\nsvd duration: " << durationSVD.count();

  int iter, bestIter = -1, i; 
  double bestObj, prevObj;
  double bestValRMSE, prevValRMSE;

  gk_csr_t *trainMat = data.trainMat;

  std::vector<std::unordered_set<int>> uISet(nUsers);
  genStats(trainMat, uISet, std::to_string(trainSeed));
  getInvalidUsersItems(trainMat, uISet, invalidUsers, invalidItems);
  for (int u = trainMat->nrows; u < data.nUsers; u++) {
    invalidUsers.insert(u);
  }
  for (int item = trainMat->ncols; item < data.nItems; item++) {
    invalidItems.insert(item);
  }

  std::vector<int> trainUsers, trainItems;
  
  for (int u = 0; u < trainMat->nrows; u++) {
    if (invalidUsers.count(u) == 0) {
      trainUsers.push_back(u);
    }
  }
  
  for (int item = 0; item < trainMat->ncols; item++) {
    if (invalidItems.count(item) == 0) {
      trainItems.push_back(item);
    }
  }

  //std::cout << "\nNNZ = " << nnz;
  prevObj = objective(data, invalidUsers, invalidItems);
  bestObj = prevObj;
  bestValRMSE = prevValRMSE = RMSE(data.valMat, invalidUsers, invalidItems);
  std::cout << "\nObj aftr svd: " << prevObj << " Train RMSE: " 
    << RMSE(data.trainMat, invalidUsers, invalidItems) << " Val RMSE: " 
    << bestValRMSE;

  std::chrono::time_point<std::chrono::system_clock> start, end;
  std::chrono::duration<double> duration;
  
 std::cout << "\nModelMF::train trainSeed: " << trainSeed 
    << " invalidUsers: " << invalidUsers.size()
    << " invalidItems: " << invalidItems.size() << std::endl;
  
  //random engine
  std::mt19937 mt(trainSeed);
  //get user-item ratings from training data
  const auto uiRatings = getUIRatings(trainMat, invalidUsers, invalidItems);

  std::cout << "\nTrain NNZ after removing invalid users and items: " 
    << uiRatings.size() << std::endl;
  double subIterDuration = 0;
  
  std::shuffle(trainUsers.begin(), trainUsers.end(), mt);
  std::shuffle(trainItems.begin(), trainItems.end(), mt);
  int maxThreads = omp_get_max_threads(); 
  
  //create maxThreads partitions of users
  int usersPerPart = trainUsers.size()/maxThreads;
  std::vector<std::unordered_set<int>> usersPart(maxThreads); 
  int currPart = 0;
  for (int i = 0; i < trainUsers.size(); i++) {
    usersPart[currPart].insert(trainUsers[i]);
    if (i % usersPerPart == 0) {
      if (currPart != maxThreads - 1) {
        currPart++;
      }
    }
  }

  //create maxThreads partitions of items
  int itemsPerPart = trainItems.size()/maxThreads;
  std::vector<std::unordered_set<int>> itemsPart(maxThreads); 
  currPart = 0;
  for (int i = 0; i < trainItems.size(); i++) {
    itemsPart[currPart].insert(trainItems[i]);
    if (i % itemsPerPart == 0) {
      if (currPart != maxThreads - 1) {
        currPart++;
      }
    }
  }

  std::vector<std::pair<int, int>> updateSeq;
  MatrixXb bMask(maxThreads, maxThreads);

  for (iter = 0; iter < maxIter; iter++) {  
    
    start = std::chrono::system_clock::now();
    sgdUpdateBlockSeq(bMask, maxThreads, updateSeq, mt);

#pragma omp parallel for
    for (int t = 0; t < maxThreads; t++) {
      const auto& users = usersPart[t];
      const auto& items = itemsPart[t];
      for (const auto& u: users) {
        for (int ii = trainMat->rowptr[u]; ii < trainMat->rowptr[u+1]; ii++) {
          
          int item = trainMat->rowind[ii];
          if (items.count(item) == 0) {
            continue;
          }

          float itemRat = trainMat->rowval[ii];
          float r_ui_est = uFac.row(u).dot(iFac.row(item));
          float diff = itemRat - r_ui_est;

          //update user
          for (int i = 0; i < facDim; i++) {
            uFac(u, i) -= learnRate * (-2.0*diff*iFac(item, i) + 2.0*uReg*uFac(u, i));
          }
        
          //update item
          for (int i = 0; i < facDim; i++) {
            iFac(item, i) -= learnRate * (-2.0*diff*uFac(u, i) + 2.0*iReg*iFac(item, i));
          }

        }
      }
    }

    end = std::chrono::system_clock::now();  
   
    duration =  end - start;
    subIterDuration = duration.count();

    //check objective
    if (iter % OBJ_ITER == 0 || iter == maxIter-1) {
      
      if (GK_CSR_IS_VAL) {
        if (isTerminateModel(bestModel, data, iter, bestIter, bestObj, prevObj,
              bestValRMSE, prevValRMSE, invalidUsers, invalidItems)) {
          break; 
        }
      } else {
        if (isTerminateModel(bestModel, data, iter, bestIter, bestObj, prevObj,
              invalidUsers, invalidItems)) {
          break; 
        }
      }
       
      if (iter % DISP_ITER == 0) {
        std::cout << "ModelMF::train trainSeed: " << trainSeed
                  << " Iter: " << iter << " Objective: " << std::scientific << prevObj 
                  << " Train RMSE: " << RMSE(data.trainMat, invalidUsers, invalidItems)
                  << " Val RMSE: " << prevValRMSE
                  << " subIterDuration: " << subIterDuration
                  << std::endl;
      }

      if (iter % SAVE_ITER == 0 || iter == maxIter - 1) {
        std::string modelFName = std::string(data.prefix);
        bestModel.saveFacs(modelFName);
      }

    }
     
  }
      
  //save best model found till now
  std::string modelFName = std::string(data.prefix);
  bestModel.saveFacs(modelFName);

  std::cout << "\nBest model validation RMSE: " << bestModel.RMSE(data.valMat, 
      invalidUsers, invalidItems);

}


void ModelMF::trainUShuffle(const Data &data, Model &bestModel, 
    std::unordered_set<int>& invalidUsers,
    std::unordered_set<int>& invalidItems) {

  //copy passed known factors
  //uFac = data.origUFac;
  //iFac = data.origIFac;
  
  std::cout << "\nModelMF::train trainSeed: " << trainSeed;
  
  int nnz = data.trainNNZ;
  
  std::cout << "\nObj b4 svd: " << objective(data) 
    << " Train RMSE: " << RMSE(data.trainMat) 
    << " Train nnz: " << nnz << std::endl;
  
  std::chrono::time_point<std::chrono::system_clock> startSVD, endSVD;
  startSVD = std::chrono::system_clock::now();
  //initialization with svd of the passed matrix
  //svdFrmSvdlibCSR(data.trainMat, facDim, uFac, iFac, false); 
  
  endSVD = std::chrono::system_clock::now();
  std::chrono::duration<double> durationSVD =  (endSVD - startSVD) ;
  std::cout << "\nsvd duration: " << durationSVD.count();

  int u, item, iter, bestIter = -1, i; 
  float itemRat;
  double diff, r_ui_est;
  double bestObj, prevObj;
  double bestValRMSE, prevValRMSE;

  gk_csr_t *trainMat = data.trainMat;

  std::vector<std::unordered_set<int>> uISet(nUsers);
  genStats(trainMat, uISet, std::to_string(trainSeed));
  getInvalidUsersItems(trainMat, uISet, invalidUsers, invalidItems);
  for (int u = trainMat->nrows; u < data.nUsers; u++) {
    invalidUsers.insert(u);
  }
  for (int item = trainMat->ncols; item < data.nItems; item++) {
    invalidItems.insert(item);
  }

  //std::cout << "\nNNZ = " << nnz;
  prevObj = objective(data, invalidUsers, invalidItems);
  bestObj = prevObj;
  bestValRMSE = prevValRMSE = RMSE(data.valMat, invalidUsers, invalidItems);
  std::cout << "\nObj aftr svd: " << prevObj << " Train RMSE: " 
    << RMSE(data.trainMat, invalidUsers, invalidItems) << " Val RMSE: " 
    << bestValRMSE;

  std::chrono::time_point<std::chrono::system_clock> start, end;
  std::chrono::duration<double> duration;
  
 std::cout << "\nModelMF::trainUShuffle trainSeed: " << trainSeed 
    << " invalidUsers: " << invalidUsers.size()
    << " invalidItems: " << invalidItems.size() << std::endl;
  
  //random engine
  std::mt19937 mt(trainSeed);
  std::vector<size_t> validUsers;
  for (int u = 0; u < nUsers; u++) {
    if (!(invalidUsers.count(u) > 0)) {
      validUsers.push_back(u);
    } 
  }
  std::cout << "No. of valid users: " << validUsers.size() << std::endl;
  
  double subIterDuration = 0;
  
  for (iter = 0; iter < maxIter; iter++) {  
    
    start = std::chrono::system_clock::now();
    
    //shuffle the users
    std::shuffle(validUsers.begin(), validUsers.end(), mt);

    for (const auto& u: validUsers) {

      for (int ii = trainMat->rowptr[u]; ii < trainMat->rowptr[u+1]; ii++) {
        
        item = trainMat->rowind[ii];
        
        itemRat = trainMat->rowval[ii];
        r_ui_est = uFac.row(u).dot(iFac.row(item));
        diff = itemRat - r_ui_est;

        //update user
        for (int i = 0; i < facDim; i++) {
          uFac(u, i) -= learnRate * (-2.0*diff*iFac(item, i) + 2.0*uReg*uFac(u, i));
        }
      
        //update item
        for (int i = 0; i < facDim; i++) {
          iFac(item, i) -= learnRate * (-2.0*diff*uFac(u, i) + 2.0*iReg*iFac(item, i));
        }

      }
     
    }

    end = std::chrono::system_clock::now();  
   
    duration =  end - start;
    subIterDuration = duration.count();

    //check objective
    if (iter % OBJ_ITER == 0 || iter == maxIter-1) {
      
      if (GK_CSR_IS_VAL) {
        if (isTerminateModel(bestModel, data, iter, bestIter, bestObj, prevObj,
              bestValRMSE, prevValRMSE, invalidUsers, invalidItems)) {
          break; 
        }
      } else {
        if (isTerminateModel(bestModel, data, iter, bestIter, bestObj, prevObj,
              invalidUsers, invalidItems)) {
          break; 
        }
      }
       
      if (iter % DISP_ITER == 0) {
        std::cout << "ModelMF::trainUShuffle trainSeed: " << trainSeed
                  << " Iter: " << iter << " Objective: " << std::scientific << prevObj 
                  << " Train RMSE: " << RMSE(data.trainMat, invalidUsers, invalidItems)
                  << " Val RMSE: " << prevValRMSE
                  << " subIterDuration: " << subIterDuration
                  << std::endl;
      }

      if (iter % SAVE_ITER == 0 || iter == maxIter - 1) {
        std::string modelFName = std::string(data.prefix);
        bestModel.saveFacs(modelFName);
      }

    }
     
  }
      
  //save best model found till now
  std::string modelFName = std::string(data.prefix);
  bestModel.saveFacs(modelFName);

  std::cout << "\nBest model validation RMSE: " << bestModel.RMSE(data.valMat, 
      invalidUsers, invalidItems);

}


void ModelMF::trainALS(const Data &data, Model &bestModel, 
    std::unordered_set<int>& invalidUsers,
    std::unordered_set<int>& invalidItems) {

  //copy passed known factors
  //uFac = data.origUFac;
  //iFac = data.origIFac;
  
  std::cout << "\nModelMF::trainALS trainSeed: " << trainSeed;
  
  int nnz = data.trainNNZ;
  
  std::cout << "\nObj b4 svd: " << objective(data) 
    << " Train RMSE: " << RMSE(data.trainMat) 
    << " Train nnz: " << nnz << std::endl;
  
  std::chrono::time_point<std::chrono::system_clock> startSVD, endSVD;
  startSVD = std::chrono::system_clock::now();
  //initialization with svd of the passed matrix
  //svdFrmSvdlibCSR(data.trainMat, facDim, uFac, iFac, false); 
  
  endSVD = std::chrono::system_clock::now();
  std::chrono::duration<double> durationSVD =  (endSVD - startSVD) ;
  std::cout << "\nsvd duration: " << durationSVD.count();

  int u, item, iter, bestIter = -1; 
  float itemRat;
  double diff, r_ui_est;
  double bestObj, prevObj;
  double bestValRMSE, prevValRMSE;

  gk_csr_t *trainMat = data.trainMat;

  //array to hold user and item gradients
  std::vector<double> uGrad (facDim, 0);
  std::vector<double> iGrad (facDim, 0);
 
  //vector to hold user gradient accumulation
  std::vector<std::vector<double>> uGradsAcc (nUsers, 
      std::vector<double>(facDim,0)); 

  //vector to hold item gradient accumulation
  std::vector<std::vector<double>> iGradsAcc (nItems, 
      std::vector<double>(facDim,0)); 

  std::vector<std::unordered_set<int>> uISet(nUsers);
  genStats(trainMat, uISet, std::to_string(trainSeed));
  getInvalidUsersItems(trainMat, uISet, invalidUsers, invalidItems);
 
  //std::cout << "\nNNZ = " << nnz;
  prevObj = objective(data, invalidUsers, invalidItems);
  bestObj = prevObj;
  bestValRMSE = prevValRMSE = RMSE(data.valMat, invalidUsers, invalidItems);
  std::cout << "\nObj aftr svd: " << prevObj << " Train RMSE: " 
    << RMSE(data.trainMat, invalidUsers, invalidItems);

  std::chrono::time_point<std::chrono::system_clock> start, end;
  std::chrono::duration<double> duration;
  
 std::cout << "\nModelMF::trainALS trainSeed: " << trainSeed 
    << " invalidUsers: " << invalidUsers.size()
    << " invalidItems: " << invalidItems.size() << std::endl;
  
  //random engine
  std::mt19937 mt(trainSeed);
  //get user-item ratings from training data
  auto uiRatings = getUIRatings(trainMat, invalidUsers, invalidItems);
  //index to above uiRatings pair
  std::vector<size_t> uiRatingInds(uiRatings.size());
  std::iota(uiRatingInds.begin(), uiRatingInds.end(), 0);


  std::cout << "\nTrain NNZ after removing invalid users and items: " 
    << uiRatings.size() << std::endl;
  double subIterDuration = 0;
  for (iter = 0; iter < maxIter; iter++) { 

    start = std::chrono::system_clock::now();


#pragma omp parallel
{

    Eigen::MatrixXf YTY = Eigen::MatrixXf::Zero(facDim, facDim);
    Eigen::VectorXf b = Eigen::VectorXf::Zero(facDim);

    int u, user, item, uu, ii, j, k;
    float rating;
    
    //update users
#pragma omp for
    for (u = 0; u < nUsers; u++) {
      
      //skip if invalid
      if (invalidUsers.count(u) > 0) {
        continue;
      } 
      YTY.fill(0);
      b.fill(0);

      
      for (ii = trainMat->rowptr[u]; ii < trainMat->rowptr[u+1]; ii++) {
        item = trainMat->rowind[ii];
        rating = trainMat->rowval[ii];
        if (rating > 0) {
          //update YTY
          for (int j = 0; j < facDim; j++) {
            for (int k = 0; k < facDim; k++) {
              YTY(j, k) += iFac(item, j)*iFac(item, k);
            }
            b(j) += rating*iFac(item, j);
          }
        }
      }
      
      //add u reg
      for (j = 0; j < facDim; j++) {
        YTY(j,j) += uReg;
      }
      
      //solve YTY * u = b
      Eigen::VectorXf ufac =  YTY.ldlt().solve(b); 
      //Eigen::VectorXf ufac =  YTY.lu().solve(b); 
      for (int j = 0; j < facDim; j++) {
        uFac(u, j) = ufac[j];
      } 
    } 

    //update items
#pragma omp for
    for (item = 0; item < nItems; item++) {
      //skip if invalid
      if (invalidItems.count(item) > 0) {
        continue;
      }
     
      YTY.fill(0);
      b.fill(0);
      
      for (uu = trainMat->colptr[item]; uu < trainMat->colptr[item+1]; uu++) {
        user = trainMat->colind[uu];
        rating = trainMat->colval[uu];
        if (rating > 0) {
          //update YTY
          for (int j = 0; j < facDim; j++) {
            for (int k = 0; k < facDim; k++) {
              YTY(j, k) += uFac(user, j)*uFac(user, k);
            }
            b(j) += rating*uFac(user, j);
          }
        }
      }
      
      //add item reg
      for (j = 0; j < facDim; j++) {
        YTY(j, j) += iReg;
      }
      
      //solve YTY * v = b
      Eigen::VectorXf ifac =  YTY.ldlt().solve(b); 
      //Eigen::VectorXf ifac =  YTY.lu().solve(b); 
      for (int j = 0; j < facDim; j++) {
        iFac(item, j) = ifac[j];
      } 

    } 

}
    end = std::chrono::system_clock::now();  
   
    duration =  end - start;
    subIterDuration = duration.count();

    //check objective
    if (iter % OBJ_ITER == 0 || iter == maxIter-1) {
      
      if (GK_CSR_IS_VAL) {
        if (isTerminateModel(bestModel, data, iter, bestIter, bestObj, prevObj,
              bestValRMSE, prevValRMSE, invalidUsers, invalidItems)) {
          break; 
        }
      } else {
        if (isTerminateModel(bestModel, data, iter, bestIter, bestObj, prevObj,
              invalidUsers, invalidItems)) {
          break; 
        }
      }
      
      if (iter % DISP_ITER == 0) {
        std::cout << "ModelMF::trainALS trainSeed: " << trainSeed
                  << " Iter: " << iter << " Objective: " << std::scientific << prevObj 
                  << " Train RMSE: " << RMSE(data.trainMat, invalidUsers, invalidItems)
                  << " Val RMSE: " << prevValRMSE
                  << " subIterDuration: " << subIterDuration
                  << std::endl;
      }

      if (iter % SAVE_ITER == 0 || iter == maxIter - 1) {
        std::string modelFName = std::string(data.prefix);
        bestModel.saveFacs(modelFName);
      }

    }
     
  }
      
  //save best model found till now
  std::string modelFName = std::string(data.prefix);
  bestModel.saveFacs(modelFName);

  std::cout << "\nBest model validation RMSE: " << bestModel.RMSE(data.valMat, 
      invalidUsers, invalidItems);

}


void ModelMF::trainCCDPP(const Data &data, Model &bestModel, 
    std::unordered_set<int>& invalidUsers,
    std::unordered_set<int>& invalidItems) {

  //copy passed known factors
  //uFac = data.origUFac;
  //iFac = data.origIFac;
  
  std::cout << "\nModelMF::trainCCDPP trainSeed: " << trainSeed;
  
  int nnz = data.trainNNZ;
  
  std::cout << "\nObj b4 svd: " << objective(data) 
    << " Train RMSE: " << RMSE(data.trainMat) 
    << " Train nnz: " << nnz << std::endl;
  
  std::chrono::time_point<std::chrono::system_clock> startSVD, endSVD;
  startSVD = std::chrono::system_clock::now();
  //initialization with svd of the passed matrix
  //svdFrmSvdlibCSR(data.trainMat, facDim, uFac, iFac, false); 
  
  endSVD = std::chrono::system_clock::now();
  std::chrono::duration<double> durationSVD =  (endSVD - startSVD) ;
  std::cout << "\nsvd duration: " << durationSVD.count();

  int iter, bestIter = -1; 
  double diff, r_ui_est;
  double bestObj, prevObj;
  double bestValRMSE, prevValRMSE;

  gk_csr_t *trainMat = data.trainMat;

  //array to hold user and item gradients
  std::vector<double> uGrad (facDim, 0);
  std::vector<double> iGrad (facDim, 0);
 
  //vector to hold user gradient accumulation
  std::vector<std::vector<double>> uGradsAcc (nUsers, 
      std::vector<double>(facDim,0)); 

  //vector to hold item gradient accumulation
  std::vector<std::vector<double>> iGradsAcc (nItems, 
      std::vector<double>(facDim,0)); 

  std::vector<std::unordered_set<int>> uISet(nUsers);
  genStats(trainMat, uISet, std::to_string(trainSeed));
  getInvalidUsersItems(trainMat, uISet, invalidUsers, invalidItems);
 
  //std::cout << "\nNNZ = " << nnz;
  prevObj = objective(data, invalidUsers, invalidItems);
  bestObj = prevObj;
  bestValRMSE = prevValRMSE = RMSE(data.valMat, invalidUsers, invalidItems);
  std::cout << "\nObj aftr svd: " << prevObj << " Train RMSE: " 
    << RMSE(data.trainMat, invalidUsers, invalidItems);

  std::chrono::time_point<std::chrono::system_clock> start, end;
  std::chrono::duration<double> duration;
  
 std::cout << "\nModelMF::trainCCDPP trainSeed: " << trainSeed 
    << " invalidUsers: " << invalidUsers.size()
    << " invalidItems: " << invalidItems.size() << std::endl;
  
  //random engine
  std::mt19937 mt(trainSeed);
  //get user-item ratings from training data
  auto uiRatings = getUIRatings(trainMat, invalidUsers, invalidItems);
  //index to above uiRatings pair
  std::vector<size_t> uiRatingInds(uiRatings.size());
  std::iota(uiRatingInds.begin(), uiRatingInds.end(), 0);
  
  std::vector<int> dims(facDim);
  std::iota(dims.begin(), dims.end(), 0);
  
  std::uniform_int_distribution<> binDis(0, 1);

  //residual mat
  gk_csr_t *res = gk_csr_Dup(trainMat);

  std::cout << "\nTrain NNZ after removing invalid users and items: " 
    << uiRatings.size() << std::endl;
  double subIterDuration = 0;
  
  //fill uFac as 0
  uFac.fill(0);
  Eigen::VectorXf u_k(nUsers), v_k(nItems);

  for (iter = 0; iter < maxIter; iter++) { 

    start = std::chrono::system_clock::now();
    std::shuffle(dims.begin(), dims.end(), mt);
    for (const auto& k : dims) {
      u_k = uFac.col(k);
      v_k = iFac.col(k);

      //update residual
      if (iter > 0) {
          //update residual res
#pragma omp parallel for
          for (int u = 0; u < nUsers; u++) {
            if (invalidUsers.count(u) > 0) {
              continue;
            }
            for (int ii = res->rowptr[u]; ii < res->rowptr[u+1]; ii++) {
              int item = res->rowind[ii];
              res->rowval[ii] += uFac(u, k)*iFac(item, k); 
            }
          }
          
          //update residual res^T
#pragma omp parallel for
          for (int item = 0; item < nItems; item++) {
            if (invalidItems.count(item) > 0 || item >= res->ncols) {
              continue;
            }
            for (int uu = res->colptr[item]; uu < res->colptr[item+1]; uu++) {
              int u = res->colind[uu];
              res->colval[uu] += uFac(u, k)*iFac(item, k); 
            }
          } 
      }     
      
      for (int subIter = 0; subIter < 5; subIter++) {
        
        //update u
#pragma omp parallel for 
        for (int u = 0; u < nUsers; u++) {
          if (invalidUsers.count(u) > 0) {
            continue;
          }
          double num = 0, denom = uReg, newV;
          for (int ii = res->rowptr[u]; ii < res->rowptr[u+1]; ii++) {
            int item = res->rowind[ii];
            num += res->rowval[ii] * v_k(item);
            denom += v_k(item)*v_k(item);
          }
          newV = num/denom;
          u_k(u) = newV;
        }

        //update v
#pragma omp parallel for 
        for (int item = 0; item < nItems; item++) {
          if (invalidItems.count(item) > 0 || item >= res->ncols) {
            continue;
          }
          double num = 0, denom = iReg, newV;
          for (int uu = res->colptr[item]; uu < res->colptr[item+1]; uu++) {
            int u = res->colind[uu];
            num += res->colval[uu] * u_k(u);
            denom += u_k(u)*u_k(u);
          }
          newV = num/denom;
          v_k(item) = newV;
        }

      }

        //update residual res
#pragma omp parallel for
        for (int u = 0; u < nUsers; u++) {
          if (invalidUsers.count(u) > 0) {
            continue;
          }
          for (int ii = res->rowptr[u]; ii < res->rowptr[u+1]; ii++) {
            int item = res->rowind[ii];
            res->rowval[ii] -= u_k(u)*v_k(item); 
          }
        }
        
        //update residual res^T
#pragma omp parallel for
        for (int item = 0; item < nItems; item++) {
          if (invalidItems.count(item) > 0 || item >= res->ncols) {
            continue;
          }
          for (int uu = res->colptr[item]; uu < res->colptr[item+1]; uu++) {
            int u = res->colind[uu];
            res->colval[uu] -= u_k(u)*v_k(item); 
          }
        } 
      
       //update factors
       uFac.col(k) = u_k;
       iFac.col(k) = v_k;
    }

    end = std::chrono::system_clock::now();  
   
    duration =  end - start;
    subIterDuration = duration.count();

    //check objective
    if (iter % OBJ_ITER == 0 || iter == maxIter-1) {
      
      if (GK_CSR_IS_VAL) {
        if (isTerminateModel(bestModel, data, iter, bestIter, bestObj, prevObj,
              bestValRMSE, prevValRMSE, invalidUsers, invalidItems)) {
          break; 
        }
      } else {
        if (isTerminateModel(bestModel, data, iter, bestIter, bestObj, prevObj,
              invalidUsers, invalidItems)) {
          break; 
        }
      }
     
      if (iter % DISP_ITER == 0) {
        std::cout << "ModelMF::trainCCDPP trainSeed: " << trainSeed
                  << " Iter: " << iter << " Objective: " << std::scientific << prevObj 
                  << " Train RMSE: " << RMSE(data.trainMat, invalidUsers, invalidItems)
                  << " Val RMSE: " << prevValRMSE
                  << " subIterDuration: " << subIterDuration
                  << std::endl;
      }

      if (iter % SAVE_ITER == 0 || iter == maxIter - 1) {
        std::string modelFName = std::string(data.prefix);
        bestModel.saveFacs(modelFName);
      }

    }
     
  }
      
  //save best model found till now
  std::string modelFName = std::string(data.prefix);
  bestModel.saveFacs(modelFName);

  std::cout << "\nBest model validation RMSE: " << bestModel.RMSE(data.valMat, 
      invalidUsers, invalidItems);
  
  gk_csr_Free(&res);
}


void ModelMF::trainCCD(const Data &data, Model &bestModel, 
    std::unordered_set<int>& invalidUsers,
    std::unordered_set<int>& invalidItems) {

  //copy passed known factors
  //uFac = data.origUFac;
  //iFac = data.origIFac;
  
  std::cout << "\nModelMF::trainCCD trainSeed: " << trainSeed;
  
  int nnz = data.trainNNZ;
  
  std::cout << "\nObj b4 svd: " << objective(data) 
    << " Train RMSE: " << RMSE(data.trainMat) 
    << " Train nnz: " << nnz << std::endl;
  
  std::chrono::time_point<std::chrono::system_clock> startSVD, endSVD;
  startSVD = std::chrono::system_clock::now();
  //initialization with svd of the passed matrix
  //svdFrmSvdlibCSR(data.trainMat, facDim, uFac, iFac, false); 
  
  endSVD = std::chrono::system_clock::now();
  std::chrono::duration<double> durationSVD =  (endSVD - startSVD) ;
  std::cout << "\nsvd duration: " << durationSVD.count();

  int u, item, iter, bestIter = -1; 
  float itemRat;
  double diff, r_ui_est;
  double bestObj, prevObj;
  double bestValRMSE, prevValRMSE;

  gk_csr_t *trainMat = data.trainMat;

  //array to hold user and item gradients
  std::vector<double> uGrad (facDim, 0);
  std::vector<double> iGrad (facDim, 0);
 
  //vector to hold user gradient accumulation
  std::vector<std::vector<double>> uGradsAcc (nUsers, 
      std::vector<double>(facDim,0)); 

  //vector to hold item gradient accumulation
  std::vector<std::vector<double>> iGradsAcc (nItems, 
      std::vector<double>(facDim,0)); 

  std::vector<std::unordered_set<int>> uISet(nUsers);
  genStats(trainMat, uISet, std::to_string(trainSeed));
  getInvalidUsersItems(trainMat, uISet, invalidUsers, invalidItems);
 
  //std::cout << "\nNNZ = " << nnz;
  prevObj = objective(data, invalidUsers, invalidItems);
  bestObj = prevObj;
  bestValRMSE = prevValRMSE = RMSE(data.valMat, invalidUsers, invalidItems);
  std::cout << "\nObj aftr svd: " << prevObj << " Train RMSE: " 
    << RMSE(data.trainMat, invalidUsers, invalidItems);

  std::chrono::time_point<std::chrono::system_clock> start, end;
  std::chrono::duration<double> duration;
  
 std::cout << "\nModelMF::trainCCD trainSeed: " << trainSeed 
    << " invalidUsers: " << invalidUsers.size()
    << " invalidItems: " << invalidItems.size() << std::endl;
  
  //random engine
  std::mt19937 mt(trainSeed);
  //get user-item ratings from training data
  auto uiRatings = getUIRatings(trainMat, invalidUsers, invalidItems);
  //index to above uiRatings pair
  std::vector<size_t> uiRatingInds(uiRatings.size());
  std::iota(uiRatingInds.begin(), uiRatingInds.end(), 0);
 
  std::vector<int> dims(facDim);
  std::iota(dims.begin(), dims.end(), 0);

  std::uniform_int_distribution<> dis(0, facDim-1);

  //residual mat
  gk_csr_t *res = gk_csr_Dup(trainMat);

  std::cout << "\nTrain NNZ after removing invalid users and items: " 
    << uiRatings.size() << std::endl;
  double subIterDuration = 0;
  
  //fill uFac as 0
#pragma omp parallel for
  for (int u = 0; u < nUsers; u++) {
    for (int k = 0; k < facDim; k++) {
      uFac(u, k) = 0;
     }
  } 


  for (iter = 0; iter < maxIter; iter++) { 

    start = std::chrono::system_clock::now();

#pragma omp parallel for
      for (int u = 0; u < nUsers; u++) {
        
        if (invalidUsers.count(u) > 0) {
          continue;
        }
      
        std::vector<int> udims(dims);
        std::shuffle(udims.begin(), udims.end(), mt);

         for (const auto& k: udims) { 
          double num = 0, denom = uReg, newV;
          
          //compute update
          for (int ii = res->rowptr[u]; ii < res->rowptr[u+1]; ii++) {
            int item = res->rowind[ii];
            num += (res->rowval[ii] + uFac(u, k)*iFac(item, k))*iFac(item, k);
            denom += iFac(item, k)*iFac(item, k);
          }
          newV = num/denom;
          
           //update residual
           for (int ii = res->rowptr[u]; ii < res->rowptr[u+1]; ii++) {
            int item = res->rowind[ii];
            double upd = (newV - uFac(u, k))*iFac(item, k); 
            res->rowval[ii] -= upd;
            //update residual in item view for the user
            int lb = res->colptr[item];
            int ub = res->colptr[item+1]-1;
            int binInd = binSearch(res->colind, u, ub, lb);
            if (binInd != -1) {
              res->colval[binInd] -= upd;          
            }
          }
          
          uFac(u, k) = newV;
        }
      }

#pragma omp parallel for
      for (int item = 0; item < nItems; item++) {
        
        if (invalidItems.count(item) > 0 || item >= res->ncols) {
          continue;
        }
       
        std::vector<int> udims(dims);
        std::shuffle(udims.begin(), udims.end(), mt);

        for (const auto& k: udims) {
          double num = 0, denom = iReg, newV;
          
          //compute update
          for (int uu = res->colptr[item]; uu < res->colptr[item+1]; uu++) {
            int u = res->colind[uu];
            num += (res->colval[uu] + uFac(u, k)*iFac(item, k))*uFac(u, k);
            denom += uFac(u, k)*uFac(u, k);
          }
          newV = num/denom;
          
          //update residual
          for (int uu = res->colptr[item]; uu < res->colptr[item+1]; uu++) {
            int u = res->colind[uu];
            double upd = (newV - iFac(item, k))*uFac(u, k); 
            res->colval[uu] -= upd;
            //update residual in user view
            int lb = res->rowptr[u];
            int ub = res->rowptr[u+1]-1;
            int binInd = binSearch(res->rowind, item, ub, lb);
            if (binInd !=  -1) {
              res->rowval[binInd] -= upd;
            }
          }

          //update factor
          iFac(item, k) = newV;
        }
      }

    end = std::chrono::system_clock::now();  
   
    duration =  end - start;
    subIterDuration = duration.count();

    //check objective
    if (iter % OBJ_ITER == 0 || iter == maxIter-1) {
      
      if (GK_CSR_IS_VAL) {
        if (isTerminateModel(bestModel, data, iter, bestIter, bestObj, prevObj,
              bestValRMSE, prevValRMSE, invalidUsers, invalidItems)) {
          break; 
        }
      } else {
        if (isTerminateModel(bestModel, data, iter, bestIter, bestObj, prevObj,
              invalidUsers, invalidItems)) {
          break; 
        }
      }
     
      if (iter % DISP_ITER == 0) {
        std::cout << "ModelMF::trainCCD trainSeed: " << trainSeed
                  << " Iter: " << iter << " Objective: " << std::scientific << prevObj 
                  << " Train RMSE: " << RMSE(data.trainMat, invalidUsers, invalidItems)
                  << " Val RMSE: " << prevValRMSE
                  << " subIterDuration: " << subIterDuration
                  << std::endl;
      }

      if (iter % SAVE_ITER == 0 || iter == maxIter - 1) {
        std::string modelFName = std::string(data.prefix);
        bestModel.saveFacs(modelFName);
      }

    }
     
  }
      
  //save best model found till now
  std::string modelFName = std::string(data.prefix);
  bestModel.saveFacs(modelFName);

  std::cout << "\nBest model validation RMSE: " << bestModel.RMSE(data.valMat, 
      invalidUsers, invalidItems);
  
  gk_csr_Free(&res);
}


void ModelMF::hogTrain(const Data &data, Model &bestModel, 
    std::unordered_set<int>& invalidUsers,
    std::unordered_set<int>& invalidItems) {

  //copy passed known factors
  //uFac = data.origUFac;
  //iFac = data.origIFac;
  
  std::cout << "\nModelMF::hogTrain trainSeed: " << trainSeed;
  
  int nnz = data.trainNNZ;
  
  std::cout << "\nObj b4 svd: " << objective(data) 
    << " Train RMSE: " << RMSE(data.trainMat) 
    << " Train nnz: " << nnz << std::endl;
  
  std::chrono::time_point<std::chrono::system_clock> startSVD, endSVD;
  startSVD = std::chrono::system_clock::now();
  //initialization with svd of the passed matrix
  //svdFrmSvdlibCSR(data.trainMat, facDim, uFac, iFac, false); 
  
  endSVD = std::chrono::system_clock::now();
  std::chrono::duration<double> durationSVD =  (endSVD - startSVD) ;
  std::cout << "\nsvd duration: " << durationSVD.count();

  int iter, bestIter = -1; 
  double bestObj, prevObj;
  double bestValRMSE, prevValRMSE;

  gk_csr_t *trainMat = data.trainMat;

 
  //vector to hold user gradient accumulation
  std::vector<std::vector<double>> uGradsAcc (nUsers, 
      std::vector<double>(facDim,0)); 

  //vector to hold item gradient accumulation
  std::vector<std::vector<double>> iGradsAcc (nItems, 
      std::vector<double>(facDim,0)); 

  std::vector<std::unordered_set<int>> uISet(nUsers);
  genStats(trainMat, uISet, std::to_string(trainSeed));
  getInvalidUsersItems(trainMat, uISet, invalidUsers, invalidItems);
  
  //std::cout << "\nNNZ = " << nnz;
  prevObj = objective(data, invalidUsers, invalidItems);
  bestObj = prevObj;
  bestValRMSE = prevValRMSE = RMSE(data.valMat, invalidUsers, invalidItems);
  std::cout << "\nObj aftr svd: " << prevObj << " Train RMSE: " 
    << RMSE(data.trainMat, invalidUsers, invalidItems) << " Val RMSE: " 
    << bestValRMSE;

  std::chrono::time_point<std::chrono::system_clock> start, end;
  std::chrono::duration<double> duration;
  
  
  std::cout << "\nModelMF::hogTrain trainSeed: " << trainSeed 
    << " invalidUsers: " << invalidUsers.size()
    << " invalidItems: " << invalidItems.size() << std::endl;
  
  //random engine
  std::mt19937 mt(trainSeed);
  //get user-item ratings from training data
  auto uiRatings = getUIRatings(trainMat, invalidUsers, invalidItems);
  //index to above uiRatings pair
  std::vector<size_t> uiRatingInds(uiRatings.size());
  std::iota(uiRatingInds.begin(), uiRatingInds.end(), 0);


  std::cout << "\nTrain NNZ after removing invalid users and items: " 
    << uiRatings.size() << std::endl;
  
  double subIterDuration = 0;
  const int indsSz = uiRatingInds.size();
  
  for (iter = 0; iter < maxIter; iter++) {  
    
    if (iter % 10 == 0) {
      //shuffle the user item rating indexes
      std::shuffle(uiRatingInds.begin(), uiRatingInds.end(), mt);
    } else {
      parBlockShuffle(uiRatingInds, mt);
    }

    start = std::chrono::system_clock::now();
#pragma omp parallel for
    for (int k = 0; k < indsSz; k++) {
      //auto ind = uiRatingInds[k];
      //get user, item and rating
      const int u       = std::get<0>(uiRatings[uiRatingInds[k]]);
      const int item    = std::get<1>(uiRatings[uiRatingInds[k]]);
      const float itemRat = std::get<2>(uiRatings[uiRatingInds[k]]);
      
      double r_ui_est = uFac.row(u).dot(iFac.row(item));
      const double diff = itemRat - r_ui_est;

      //update user
      uFac.row(u) -= learnRate*(-2.0*diff*iFac.row(item) + 2.0*uReg*uFac.row(u));
    
      //update item
      iFac.row(item) -= learnRate*(-2.0*diff*uFac.row(u) + 2.0*iReg*iFac.row(item));
    }
    end = std::chrono::system_clock::now();  
   
    duration =  end - start;
    subIterDuration = duration.count();

    //check objective
    if (iter % OBJ_ITER == 0 || iter == maxIter-1) {
     
      if (GK_CSR_IS_VAL) {
        if (isTerminateModel(bestModel, data, iter, bestIter, bestObj, prevObj,
              bestValRMSE, prevValRMSE, invalidUsers, invalidItems)) {
          break; 
        }
      } else {
        if (isTerminateModel(bestModel, data, iter, bestIter, bestObj, prevObj,
              invalidUsers, invalidItems)) {
          break; 
        }
      }

      if (iter % DISP_ITER == 0) {
        std::cout << "ModelMF::hogTrain trainSeed: " << trainSeed
                  << " Iter: " << iter << " Objective: " << std::scientific << prevObj 
                  << " Train RMSE: " << RMSE(data.trainMat, invalidUsers, invalidItems)
                  << " Val RMSE: " << prevValRMSE
                  << " sub duration: " << subIterDuration
                  << std::endl;
      }

      if (iter % SAVE_ITER == 0 || iter == maxIter - 1) {
        std::string modelFName = std::string(data.prefix);
        bestModel.saveFacs(modelFName);
      }

    }
     
  }
      
  //save best model found till now
  std::string modelFName = std::string(data.prefix);
  bestModel.saveFacs(modelFName);

  std::cout << "\nBest model validation RMSE: " << bestModel.RMSE(data.valMat, 
      invalidUsers, invalidItems);
}


