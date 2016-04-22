#include "modelMFLoc.h"


void ModelMFLoc::zeroedTailItemFacs(std::unordered_set<int>& headItems) {
  
  for (int item = 0; item < nItems; item++) {
    if (headItems.find(item) == headItems.end()) {
      //tail item found
      //for (int i = 3*(facDim/4); i < facDim; i++) {
      for (int i = (facDim/2); i < facDim; i++) {
        iFac[item][i] = 0;
      }
    }
  }

}


void ModelMFLoc::zeroedTailUserFacs(std::unordered_set<int>& headUsers) {
  
  for (int user = 0; user < nUsers; user++) {
    if (headUsers.find(user) == headUsers.end()) {
      //tail item found
      //for (int i = 3*(facDim/4); i < facDim; i++) {
      for (int i = (facDim/2); i < facDim; i++) {
        uFac[user][i] = 0;
      }
    }
  }

}


void ModelMFLoc::train(const Data &data, Model &bestModel, 
    std::unordered_set<int>& invalidUsers,
    std::unordered_set<int>& invalidItems) {

  int u, item, iter, bestIter; 
  float itemRat;
  double diff, r_ui_est;
  double bestObj, prevObj;
  double bestValRMSE, prevValRMSE;
  int nnz = data.trainNNZ;
  
  std::cout << "\nModelMFLoc::train trainSeed: " << trainSeed;
  
  std::cout << "\nObj b4: " << objective(data) 
    << " Train RMSE: " << RMSE(data.trainMat) 
    << " Train nnz: " << nnz << std::endl;
  std::cout << "\nNo. of head items: " << headItems.size();

  //zeroed out head users and head items
  zeroedTailItemFacs(headItems);
  zeroedTailUserFacs(headUsers);

  std::cout << "\nObj after: " << objective(data) 
    << " Train RMSE: " << RMSE(data.trainMat) 
    << " Train nnz: " << nnz << std::endl;

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

  std::chrono::time_point<std::chrono::system_clock> start, end;
  std::chrono::duration<double> duration;
  
  std::vector<std::unordered_set<int>> uISet(nUsers);
  genStats(trainMat, uISet, std::to_string(trainSeed));
  getInvalidUsersItems(trainMat, uISet, invalidUsers, invalidItems);
  
  std::cout << "\nModelMFLoc::train trainSeed: " << trainSeed 
    << " invalidUsers: " << invalidUsers.size()
    << " invalidItems: " << invalidItems.size() << std::endl;
  
  //random engine
  std::mt19937 mt(trainSeed);
  //get user-item ratings from training data
  auto uiRatings = getUIRatings(trainMat, invalidUsers, invalidItems);
  //index to above uiRatings pair
  std::vector<size_t> uiRatingInds(uiRatings.size());
  std::iota(uiRatingInds.begin(), uiRatingInds.end(), 0);

  std::vector<bool> bTailUsers(nUsers, true);
  for (auto&& user: headUsers) {
    bTailUsers[user] = false;
  }
  std::vector<bool> bTailItems(nItems, true);
  for (auto&& item: headItems) {
    bTailItems[item] = false;
  }

  std::cout << "\nTrain NNZ after removing invalid users and items: " 
    << uiRatings.size() << std::endl;
  double subIterDuration = 0;
  int effFacDim = facDim;

  for (iter = 0; iter < maxIter; iter++) {  
    
    //shuffle the user item rating indexes
    std::shuffle(uiRatingInds.begin(), uiRatingInds.end(), mt);

    start = std::chrono::system_clock::now();
    for (auto&& ind: uiRatingInds) {
      //get user, item and rating
      u       = std::get<0>(uiRatings[ind]);
      item    = std::get<1>(uiRatings[ind]);
      itemRat = std::get<2>(uiRatings[ind]);
      
      //std::cout << "\nGradCheck u: " << u << " item: " << item;
      //gradCheck(u, item, itemRat);
      r_ui_est = dotProd(uFac[u], iFac[item], facDim);
      diff = itemRat - r_ui_est;

      //compute user gradient
      for (int i = 0; i < facDim; i++) {
        uGrad[i] = -2.0*diff*iFac[item][i] + 2.0*uReg*uFac[u][i];
      }

      //update user
      effFacDim = facDim;
      if (bTailUsers[u]) {
        //tail user
        effFacDim = (facDim/2);
      }

      for (int i = 0; i < effFacDim; i++) {
        uFac[u][i] -= learnRate * uGrad[i];
      }

      r_ui_est = dotProd(uFac[u], iFac[item], facDim);
      diff = itemRat - r_ui_est;
    
      //compute item gradient
      for (int i = 0; i < facDim; i++) {
        iGrad[i] = -2.0*diff*uFac[u][i] + 2.0*iReg*iFac[item][i];
      }

      //update item
      effFacDim = facDim;
      if (bTailItems[item]) {
        //tail item
        //effFacDim = 3*(facDim/4);
        effFacDim = (facDim/2);
      }

      for (int i = 0; i < effFacDim; i++) {
        iFac[item][i] -= learnRate * iGrad[i];
      }

    }
    end = std::chrono::system_clock::now();  
   
    duration =  end - start;
    subIterDuration += duration.count();

    //check objective
    if (iter % OBJ_ITER == 0 || iter == maxIter-1) {
      if (isTerminateModel(bestModel, data, iter, bestIter, bestObj, prevObj,
            bestValRMSE, prevValRMSE,
            invalidUsers, invalidItems)) {
        break; 
      }
      std::cout << "\nModelMFLoc::train trainSeed: " << trainSeed
                << " Iter: " << iter << " Objective: " << std::scientific << prevObj 
                << " Train RMSE: " << RMSE(data.trainMat, invalidUsers, invalidItems)
                << " Val RMSE: " << prevValRMSE
                << std::endl;
      std::cout << "\navg sub duration: " << subIterDuration/(iter+1) << std::endl;
      //save best model found till now
      std::string modelFName = std::string(data.prefix);
      bestModel.saveFacs(modelFName);
    }
  
  }

  std::cout << "\nBest model validation RMSE: " << bestModel.RMSE(data.valMat, 
      invalidUsers, invalidItems);

}


