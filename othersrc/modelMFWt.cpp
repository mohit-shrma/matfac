#include "modelMFWt.h"


double ModelMFWt::objective(const Data& data, std::unordered_set<int>& invalidUsers,
    std::unordered_set<int>& invalidItems) {
  int u, ii, item;
  float itemRat;
  double rmse = 0, uRegErr = 0, iRegErr = 0, obj = 0, diff = 0;
  gk_csr_t *trainMat = data.trainMat;
  std::unordered_set<int> headItems = getHeadItems(trainMat, 0.5);
  std::unordered_set<int> headUsers = getHeadUsers(trainMat, 0.5);
  double lambda0 = 0.8;
  double lambda1 = 1.0 - lambda0;

  for (u = 0; u < nUsers; u++) {
    //skip if invalid user
    auto search = invalidUsers.find(u);
    if (search != invalidUsers.end()) {
      //found and skip
      continue;
    }
    for (ii = trainMat->rowptr[u]; ii < trainMat->rowptr[u+1]; ii++) {
      item = trainMat->rowind[ii];
      //skip if invalid item
      search = invalidItems.find(item);
      if (search != invalidItems.end()) {
        //found and skip
        continue;
      }

      itemRat = trainMat->rowval[ii];
      diff = itemRat - estRating(u, item);
      if (headItems.find(item) != headItems.end() 
          && headUsers.find(u) != headUsers.end()) {
        rmse += diff*diff*lambda0;
      } else {
        rmse += diff*diff*(lambda0+lambda1);
      }

    }
    uRegErr += dotProd(uFac[u], uFac[u], facDim);
  }
  uRegErr = uRegErr*uReg;
  
  for (item = 0; item < nItems; item++) {
    //skip if invalid item
    auto search = invalidItems.find(item);
    if (search != invalidItems.end()) {
      //found and skip
      continue;
    }
    iRegErr += dotProd(iFac[item], iFac[item], facDim);
  }
  iRegErr = iRegErr*iReg;

  obj = rmse + uRegErr + iRegErr;
    
  //std::cout <<"\nrmse: " << std::scientific << rmse << " uReg: " << uRegErr << " iReg: " << iRegErr ; 

  return obj;
}


void ModelMFWt::hogTrain(const Data &data, Model &bestModel, 
    std::unordered_set<int>& invalidUsers,
    std::unordered_set<int>& invalidItems) {

  //copy passed known factors
  //uFac = data.origUFac;
  //iFac = data.origIFac;
  
  std::cout << "\nModelMFWt::hogTrain trainSeed: " << trainSeed;
  
  int nnz = data.trainNNZ;
  
  std::cout << "\nObj b4 svd: " << objective(data, invalidUsers, invalidItems) 
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

  //std::cout << "\nNNZ = " << nnz;
  prevObj = objective(data, invalidUsers, invalidItems);
  bestObj = prevObj;
  std::cout << "\nObj aftr svd: " << prevObj << " Train RMSE: " << RMSE(data.trainMat);


  std::chrono::time_point<std::chrono::system_clock> start, end;
  std::chrono::duration<double> duration;
  
  std::vector<std::unordered_set<int>> uISet(nUsers);
  genStats(trainMat, uISet, std::to_string(trainSeed));
  getInvalidUsersItems(trainMat, uISet, invalidUsers, invalidItems);
  
  std::unordered_set<int> headItems = getHeadItems(trainMat, 0.5);
  std::unordered_set<int> headUsers = getHeadItems(trainMat, 0.5);
  double lambda0 = 0.8;
  double lambda1 = 1.0 - lambda0;

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
    
    //shuffle the user item rating indexes
    std::shuffle(uiRatingInds.begin(), uiRatingInds.end(), mt);

    start = std::chrono::system_clock::now();
    const int indsSz = uiRatingInds.size();
#pragma omp parallel for
    for (int k = 0; k < indsSz; k++) {
      auto ind = uiRatingInds[k];
      //get user, item and rating
      int u       = std::get<0>(uiRatings[ind]);
      int item    = std::get<1>(uiRatings[ind]);
      float itemRat = std::get<2>(uiRatings[ind]);
      
      double r_ui_est = dotProd(uFac[u], iFac[item], facDim);
      double diff = itemRat - r_ui_est;

      if (headItems.find(item) != headItems.end()) {
        diff = diff*lambda0;
      } else {
        diff = diff*(lambda0 + lambda1);
      }

      //update user
      for (int i = 0; i < facDim; i++) {
        uFac[u][i] -= learnRate*(-2.0*diff*iFac[item][i] + 2.0*uReg*uFac[u][i]);
      }


      r_ui_est = dotProd(uFac[u], iFac[item], facDim);
      diff = itemRat - r_ui_est;
    
      if (headItems.find(item) != headItems.end()) {
        diff = diff*lambda0;
      } else {
        diff = diff*(lambda0 + lambda1);
      }

      //update item
      for (int i = 0; i < facDim; i++) {
        iFac[item][i] -= learnRate*(-2.0*diff*uFac[u][i] + 2.0*iReg*iFac[item][i]);
      }
    }
    end = std::chrono::system_clock::now();  
   
    duration =  end - start;
    subIterDuration = duration.count();

    //check objective
    if (iter % OBJ_ITER == 0 || iter == maxIter-1) {
      if (isTerminateModel(bestModel, data, iter, bestIter, bestObj, prevObj,
            invalidUsers, invalidItems)) {
        break; 
      }

      if (iter % 50 == 0) {
        std::cout << "ModelMFWt::train trainSeed: " << trainSeed
                  << " Iter: " << iter << " Objective: " << std::scientific << prevObj 
                  << " Train RMSE: " << RMSE(data.trainMat, invalidUsers, invalidItems)
                  << " Val RMSE: " << prevValRMSE
                  << " sub duration: " << subIterDuration
                  << std::endl;
      }

      if (iter % 500 == 0 || iter == maxIter - 1) {
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


void ModelMFWt::hogUITrain(const Data &data, Model &bestModel, 
    std::unordered_set<int>& invalidUsers,
    std::unordered_set<int>& invalidItems) {

  //copy passed known factors
  //uFac = data.origUFac;
  //iFac = data.origIFac;
  
  std::cout << "\nModelMFWt::hogTrain trainSeed: " << trainSeed;
  
  int nnz = data.trainNNZ;
  
  std::cout << "\nObj b4 svd: " << objective(data, invalidUsers, invalidItems) 
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

  //std::cout << "\nNNZ = " << nnz;
  prevObj = objective(data, invalidUsers, invalidItems);
  bestObj = prevObj;
  std::cout << "\nObj aftr svd: " << prevObj << " Train RMSE: " << RMSE(data.trainMat);


  std::chrono::time_point<std::chrono::system_clock> start, end;
  std::chrono::duration<double> duration;
  
  std::vector<std::unordered_set<int>> uISet(nUsers);
  genStats(trainMat, uISet, std::to_string(trainSeed));
  getInvalidUsersItems(trainMat, uISet, invalidUsers, invalidItems);
  
  std::unordered_set<int> headItems = getHeadItems(trainMat, 0.5);
  std::unordered_set<int> headUsers = getHeadItems(trainMat, 0.5);
  double lambda0 = 0.8;
  double lambda1 = 1.0 - lambda0;

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
    
    //shuffle the user item rating indexes
    std::shuffle(uiRatingInds.begin(), uiRatingInds.end(), mt);

    start = std::chrono::system_clock::now();
    const int indsSz = uiRatingInds.size();
#pragma omp parallel for
    for (int k = 0; k < indsSz; k++) {
      auto ind = uiRatingInds[k];
      //get user, item and rating
      int u       = std::get<0>(uiRatings[ind]);
      int item    = std::get<1>(uiRatings[ind]);
      float itemRat = std::get<2>(uiRatings[ind]);
      
      double r_ui_est = dotProd(uFac[u], iFac[item], facDim);
      double diff = itemRat - r_ui_est;
      
      if (headItems.find(item) != headItems.end() 
          && headUsers.find(u) != headUsers.end()) {
        diff = diff*lambda0;
      } else {
        diff = diff*(lambda0 + lambda1);
      }

      //update user
      for (int i = 0; i < facDim; i++) {
        uFac[u][i] -= learnRate*(-2.0*diff*iFac[item][i] + 2.0*uReg*uFac[u][i]);
      }


      r_ui_est = dotProd(uFac[u], iFac[item], facDim);
      diff = itemRat - r_ui_est;
    
      if (headItems.find(item) != headItems.end() 
          && headUsers.find(u) != headUsers.end()) {
        diff = diff*lambda0;
      } else {
        diff = diff*(lambda0 + lambda1);
      }

      //update item
      for (int i = 0; i < facDim; i++) {
        iFac[item][i] -= learnRate*(-2.0*diff*uFac[u][i] + 2.0*iReg*iFac[item][i]);
      }
    }
    end = std::chrono::system_clock::now();  
   
    duration =  end - start;
    subIterDuration = duration.count();

    //check objective
    if (iter % OBJ_ITER == 0 || iter == maxIter-1) {
      if (isTerminateModel(bestModel, data, iter, bestIter, bestObj, prevObj,
            invalidUsers, invalidItems)) {
        break; 
      }

      if (iter % 50 == 0) {
        std::cout << "ModelMFWt::train trainSeed: " << trainSeed
                  << " Iter: " << iter << " Objective: " << std::scientific << prevObj 
                  << " Train RMSE: " << RMSE(data.trainMat, invalidUsers, invalidItems)
                  << " Val RMSE: " << prevValRMSE
                  << " sub duration: " << subIterDuration
                  << std::endl;
      }

      if (iter % 500 == 0 || iter == maxIter - 1) {
        std::string modelFName = std::string(data.prefix);
        bestModel.saveFacs(modelFName);
      }

    }
     
  }
      
  //save best model found till now
  std::string modelFName = std::string(data.prefix);
  bestModel.saveFacs(modelFName);

}

