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
    uRegErr += uFac.row(u).dot(uFac.row(u));
    uBiasReg += uBias[u]*uBias[u];
  }
  uRegErr = uRegErr*uReg;
  uBiasReg = uBiasReg*uReg;

  for (item = 0; item < nItems; item++) {
    iRegErr += iFac.row(item).dot(iFac.row(item));
    iBiasReg += iBias[item]*iBias[item];
  }
  iRegErr = iRegErr*iReg;
  iBiasReg = iBiasReg*iReg;
  
  //obj = rmse + uRegErr + iRegErr + uBiasReg + iBiasReg;
  obj = rmse + uBiasReg + iBiasReg;
    
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
    uRegErr += uFac.row(u).dot(uFac.row(u));
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
    iRegErr += iFac.row(item).dot(iFac.row(item));
    iBiasReg += iBias[item]*iBias[item];
  }
  iRegErr = iRegErr*iReg;
  iBiasReg = iBiasReg*iReg;
  
  //obj = rmse + uRegErr + iRegErr + uBiasReg + iBiasReg;
  obj = rmse + uBiasReg + iBiasReg;
    
  return obj;
}


double ModelMFBias::estRating(int user, int item) {
  double rating = uBias[user] + iBias[item];
  //double rating = mu + uBias[user] + iBias[item] + 
  //  dotProd(uFac[user], iFac[item], facDim);
  return rating;
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
  std::chrono::duration<double> durationSVD ;

  int u, item, iter, bestIter;
  float itemRat;
  double bestObj, prevObj, r_ui_est, diff;
  double bestValRMSE, prevValRMSE;

  gk_csr_t *trainMat = data.trainMat;

  //array to hold user and item gradients
  std::vector<double> uGrad (facDim, 0);
  std::vector<double> iGrad (facDim, 0);
   std::chrono::time_point<std::chrono::system_clock> start, end;
  
  std::vector<std::unordered_set<int>> uISet(nUsers);
  genStats(trainMat, uISet, std::to_string(trainSeed));
  getInvalidUsersItems(trainMat, uISet, invalidUsers, invalidItems);
  for (int u = trainMat->nrows; u < data.nUsers; u++) {
    invalidUsers.insert(u);
  }
  for (int item = trainMat->ncols; item < data.nItems; item++) {
    invalidItems.insert(item);
  }
   
  prevObj = objective(data, invalidUsers, invalidItems);
  bestObj = prevObj;
  bestValRMSE = prevValRMSE = RMSE(data.valMat, invalidUsers, invalidItems);
  std::cout << "\nObj aftr svd: " << prevObj << " Train RMSE: " << RMSE(data.trainMat);


  std::cout << "\nModelMFBias::train trainSeed: " << trainSeed 
    << " invalidUsers: " << invalidUsers.size()
    << " invalidItems: " << invalidItems.size() << std::endl;

  std::cout << "ubias norm: " << uBias.norm() << " iBias norm: " << iBias.norm() << std::endl;


  //random engine
  std::mt19937 mt(trainSeed);
  //get user-item ratings from training data
  auto uiRatings = getUIRatings(trainMat, invalidUsers, invalidItems);
  std::cout << "\nNo. of training ratings: " << uiRatings.size(); 
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

      //update user
      //updateAdaptiveFac(uFac[u], uGrad, uGradsAcc[u]); 
      //updateFac(uFac[u], uGrad); 

      //update user bias
      uBias[u] -= learnRate*(-2.0*diff + 2.0*uReg*uBias[u]);

      //compute item gradient

      //update item
      //updateAdaptiveFac(iFac[item], iGrad, iGradsAcc[item]);
      //updateFac(iFac[item], iGrad);

      //update item bias
      iBias[item] -= learnRate*(-2.0*diff + 2.0*iReg*iBias[item]);
    }

    //check objective
    if (iter % OBJ_ITER == 0 || iter == maxIter-1) {
      if (isTerminateModel(bestModel, data, iter, bestIter, bestObj, prevObj,
            bestValRMSE, prevValRMSE,
            invalidUsers, invalidItems)) {
        break; 
      }
      if (iter % DISP_ITER == 0) {
        std::chrono::duration<double> duration =  (end - start) ;
        std::cout << "ModelMFBias::train trainSeed: " << trainSeed
                << " Iter: " << iter << " Objective: " << std::scientific << prevObj 
                << " Train RMSE: " << RMSE(data.trainMat, invalidUsers, invalidItems) 
                << " Val RMSE: " <<  prevValRMSE << " dur: " << duration.count() 
                << " uBias: " << uBias.norm() << " iBias: " << iBias.norm()
                << " best uBias: " << bestModel.uBias.norm() 
                << " best iBias: " << bestModel.iBias.norm()
                << std::endl;
      }
      
      if (iter % SAVE_ITER == 0 || iter == maxIter - 1) {
        //save best model found till now
        std::string modelFName = std::string(data.prefix);
        bestModel.save(modelFName);
      }
    }
  }

}




