#include "modelMFBias.h"

double ModelMFBias::objective(const Data& data) {

  int u, ii, item;
  float itemRat;
  double rmse = 0, uRegErr = 0, iRegErr = 0, obj = 0, diff = 0;
  double uBiasReg = 0, iBiasReg = 0;
  gk_csr_t *trainMat = data.trainMat;

  for (u = 0; u < nUsers; u++) {
    for (ii = trainMat->rowptr[u]; ii < trainMat->rowptr[u+1]; ii++) {
      item = trainMat->rowind[ii];
      itemRat = trainMat->rowval[ii];
      diff = itemRat - estRating(u, item);
      rmse += diff*diff;
    }
    uRegErr += dotProd(uFac[u], uFac[u], facDim);
    uBiasReg += uBias[u]*uBias[u];
  }
  uRegErr = uRegErr*uReg;
  uBiasReg = uBiasReg*uReg;

  for (item = 0; item < nItems; item++) {
    iRegErr += dotProd(iFac[item], iFac[item], facDim);
    iBiasReg += iBias[item]*iBias[item];
  }
  iRegErr = iRegErr*iReg;
  iBiasReg = iBiasReg*iReg;
  
  obj = rmse + uRegErr + iRegErr + uBiasReg + iBiasReg;
    
  //std::cout <<"\nrmse: " << std::scientific << rmse << " uReg: " << uRegErr << " iReg: " << iRegErr ; 

  return obj;
}


double ModelMFBias::objective(const Data& data, std::unordered_set<int>& invalidUsers, 
    std::unordered_set<int>& invalidItems) {
  int u, ii, item;
  float itemRat;
  double rmse = 0, uRegErr = 0, iRegErr = 0, obj = 0, diff = 0;
  double uBiasReg = 0, iBiasReg = 0;
  gk_csr_t *trainMat = data.trainMat;

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
      rmse += diff*diff;
    }
    uRegErr += dotProd(uFac[u], uFac[u], facDim);
    uBiasReg += uBias[u]*uBias[u];
  }
  uRegErr = uRegErr*uReg;
  uBiasReg = uBiasReg*uReg;

  for (item = 0; item < nItems; item++) {
    //skip if invalid item
    auto search = invalidItems.find(item);
    if (search != invalidItems.end()) {
      //found and skip
      continue;
    }
    iRegErr += dotProd(iFac[item], iFac[item], facDim);
    iBiasReg += iBias[item]*iBias[item];
  }
  iRegErr = iRegErr*iReg;
  iBiasReg = iBiasReg*iReg;
  
  obj = rmse + uRegErr + iRegErr + uBiasReg + iBiasReg;
    
  return obj;
}


double ModelMFBias::estRating(int user, int item) {
  double rating = mu + uBias[user] + iBias[item] + 
    dotProd(uFac[user], iFac[item], facDim);
  return rating;
}


void ModelMFBias::computeUGrad(int user, int item, float r_ui, 
    double r_ui_est, std::vector<double> &uGrad) {

  double diff = r_ui - r_ui_est;
  for (int i = 0; i < facDim; i++) {
    uGrad[i] = -2.0*diff*iFac[item][i] + 2.0*uReg*uFac[user][i];
  }
}


void ModelMFBias::computeIGrad(int user, int item, float r_ui, 
    double r_ui_est, std::vector<double> &iGrad) {

  double diff = r_ui - r_ui_est;
  for (int i = 0; i < facDim; i++) {
    iGrad[i] = -2.0*diff*uFac[user][i] + 2.0*iReg*iFac[item][i];
  }
}


void ModelMFBias::train(const Data& data, Model& bestModel, 
    std::unordered_set<int>& invalidUsers, 
    std::unordered_set<int>& invalidItems) {

  std::cout << "\nModelMFBias::train trainSeed: " << trainSeed;
  

  //global bias
  mu = meanRating(data.trainMat);
  std::cout << "\nGlobal bias: " << mu;

  int nnz = data.trainNNZ;
   
  //modify these methods
  std::cout << "\nObj b4 svd: " << objective(data) 
    << " Train RMSE: " << RMSE(data.trainMat) 
    << " Train nnz: " << nnz << std::endl;
  
  std::chrono::time_point<std::chrono::system_clock> startSVD, endSVD;
  startSVD = std::chrono::system_clock::now();
  //initialization with svd of the passed matrix
  svdFrmSvdlibCSR(data.trainMat, facDim, uFac, iFac); 
  
  endSVD = std::chrono::system_clock::now();
  std::chrono::duration<double> durationSVD =  (endSVD - startSVD) ;
  std::cout << "\nsvd duration: " << durationSVD.count();

  int u, item, iter, bestIter;
  float itemRat;
  double bestObj, prevObj, r_ui_est, diff;

  gk_csr_t *trainMat = data.trainMat;

  //array to hold user and item gradients
  std::vector<double> uGrad (facDim, 0);
  std::vector<double> iGrad (facDim, 0);
  
  prevObj = objective(data);
  std::cout << "\nObj aftr svd: " << prevObj << " Train RMSE: " << RMSE(data.trainMat);

  std::chrono::time_point<std::chrono::system_clock> start, end;
  
  std::vector<std::unordered_set<int>> uISet(nUsers);
  genStats(trainMat, uISet, std::to_string(trainSeed));
  getInvalidUsersItems(trainMat, uISet, invalidUsers, invalidItems);
  
  std::cout << "\nModelMFBias::train trainSeed: " << trainSeed 
    << " invalidUsers: " << invalidUsers.size()
    << " invalidItems: " << invalidItems.size() << std::endl;

  //random engine
  std::mt19937 mt(trainSeed);
  //get user-item ratings from training data
  auto uiRatings = getUIRatings(trainMat, invalidUsers, invalidItems);
  std::cout << "\nNo. of training ratings: " << uiRatings.size(); 
  double prevTestRMSE = 100, currTestRMSE = 100;
  for (iter = 0; iter < maxIter; iter++) {  
    start = std::chrono::system_clock::now();

    //shuffle the user item ratings
    std::shuffle(uiRatings.begin(), uiRatings.end(), mt);

    end = std::chrono::system_clock::now();  
    
    for (auto&& uiRating: uiRatings) {
      //get user, item and rating
      u       = std::get<0>(uiRating);
      item    = std::get<1>(uiRating);
      itemRat = std::get<2>(uiRating);
      
      //get estimated rating
      r_ui_est = estRating(u, item);
      
      //get difference with actual rating
      diff = itemRat - r_ui_est;

      //compute user gradient
      computeUGrad(u, item, itemRat, r_ui_est, uGrad);

      //update user
      //updateAdaptiveFac(uFac[u], uGrad, uGradsAcc[u]); 
      updateFac(uFac[u], uGrad); 

      //update user bias
      uBias[u] -= learnRate*(-2.0*diff + 2.0*uReg*uBias[u]);

      //compute item gradient
      computeIGrad(u, item, itemRat, r_ui_est, iGrad);

      //update item
      //updateAdaptiveFac(iFac[item], iGrad, iGradsAcc[item]);
      updateFac(iFac[item], iGrad);

      //update item bias
      iBias[item] -= learnRate*(-2.0*diff + 2.0*iReg*iBias[item]);
    }

    //check objective
    if (iter % OBJ_ITER == 0 || iter == maxIter-1) {
      if (isTerminateModel(bestModel, data, iter, bestIter, bestObj, prevObj,
            invalidUsers, invalidItems)) {
        break; 
      }
      currTestRMSE = RMSE(data.testMat, invalidUsers, invalidItems);
      std::cout << "\nModelMFBias::train trainSeed: " << trainSeed
                << " Iter: " << iter << " Objective: " << std::scientific << prevObj 
                << " Train RMSE: " << RMSE(data.trainMat, invalidUsers, invalidItems) 
                << " Test RMSE: " <<  currTestRMSE
                << std::endl;
      if (currTestRMSE  > prevTestRMSE) {
        break;
      }
      prevTestRMSE = currTestRMSE;
      std::chrono::duration<double> duration =  (end - start) ;
      std::cout << "\nsub duration: " << duration.count() << std::endl;
      //save best model found till now
      std::string modelFName = "ModelFull_" + std::to_string(trainSeed);
      bestModel.save(modelFName);
    }
  }

}




