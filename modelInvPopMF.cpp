#include "modelInvPopMF.h"

double ModelInvPopMF::objective(const Data& data, std::unordered_set<int>& invalidUsers,
    std::unordered_set<int>& invalidItems) {

  double rmse = 0, uRegErr = 0, iRegErr = 0, obj = 0;
  gk_csr_t *trainMat = data.trainMat;

#pragma omp parallel for reduction(+:rmse, uRegErr)
  for (int u = 0; u < nUsers; u++) {
    //skip if invalid user
    if (invalidUsers.count(u) > 0) {
      //found and skip
      continue;
    }
    for (int ii = trainMat->rowptr[u]; ii < trainMat->rowptr[u+1]; ii++) {
      int item = trainMat->rowind[ii];
      //skip if invalid item
      if (invalidItems.count(item) > 0) {
        continue;
      }
      
      float wt = invPopI[item];
      if (itemFreq[item] > userFreq[u]) {
        wt = invPopU[u];
      }
      //wt = (1.0 + rhoRMS*wt);
      wt = (1.0/(1.0 + rhoRMS*wt));

      float itemRat = trainMat->rowval[ii];
      double diff = itemRat - estRating(u, item);
      rmse += wt*diff*diff;
    }
    uRegErr += uFac.row(u).dot(uFac.row(u));
  }
  uRegErr = uRegErr*uReg;
  
#pragma omp parallel for reduction(+: iRegErr)
  for (int item = 0; item < nItems; item++) {
    //skip if invalid item
    if (invalidItems.count(item) > 0) {
      //found and skip
      continue;
    }
    iRegErr += iFac.row(item).dot(iFac.row(item));
  }
  iRegErr = iRegErr*iReg;

  obj = rmse + uRegErr + iRegErr;
    
  //std::cout <<"\nrmse: " << std::scientific << rmse << " uReg: " << uRegErr 
  //  << " iReg: " << iRegErr << std::endl; 

  return obj;
}


void ModelInvPopMF::train(const Data &data, Model &bestModel, 
    std::unordered_set<int>& invalidUsers,
    std::unordered_set<int>& invalidItems) {

  std::cout << "\nModelInvPopMF::train trainSeed: " << trainSeed;
  
  int nnz = data.trainNNZ;
  int u, item, iter, bestIter = -1, i; 
  float itemRat;
  double diff, r_ui_est;
  double bestObj, prevObj;
  double bestValRMSE, prevValRMSE;
  std::vector<int> trainUsers, trainItems;

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
  
  for (int u = 0; u < trainMat->nrows; u++) {
    if (invalidUsers.count(u) == 0) {
      trainUsers.push_back(u);
    }
  }
  nTrainUsers = trainUsers.size();

  for (int item = 0; item < trainMat->ncols; item++) {
    if (invalidItems.count(item) == 0) {
      trainItems.push_back(item);
    }
  }
  nTrainItems = trainItems.size();

  double sumPopScore = 0;
  for (auto& u: trainUsers) {
    invPopU[u] = userFreq[u]/((double)nTrainItems);//((double)nTrainItems)/userFreq[u];
    sumPopScore += invPopU[u];
  }
  for (auto& u: trainUsers) {
    invPopU[u] = invPopU[u]/sumPopScore;
  }

  sumPopScore = 0;
  for (auto& item: trainItems) {
    invPopI[item] = itemFreq[item]/((double)nTrainUsers);//((double)nTrainUsers)/itemFreq[item];
    sumPopScore += invPopI[item];
  }
  for (auto& item: trainItems) {
    invPopI[item] = invPopI[item]/sumPopScore;
  }

  //std::cout << "\nNNZ = " << nnz;
  prevObj = objective(data, invalidUsers, invalidItems);
  bestObj = prevObj;
  bestValRMSE = prevValRMSE = RMSE(data.valMat, invalidUsers, invalidItems);
  std::cout << "\nObj: " << prevObj << " Train RMSE: " 
    << RMSE(data.trainMat, invalidUsers, invalidItems) << " Val RMSE: " 
    << bestValRMSE;

  std::chrono::time_point<std::chrono::system_clock> start, end;
  std::chrono::duration<double> duration;
  
 std::cout << "\nModelInvPopMF::train trainSeed: " << trainSeed 
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

      float wt = invPopI[item]; 
      if (itemFreq[item] > userFreq[u]) {
        wt = invPopU[u];
      }
      //wt = (1.0 + rhoRMS*wt);
      wt = (1.0/(1.0 + rhoRMS*wt));
      
      //update user
      for (int i = 0; i < facDim; i++) {
        uFac(u, i) -= learnRate * (-2.0*wt*diff*iFac(item, i) + 2.0*uReg*uFac(u, i));
      }

      //update item
      for (int i = 0; i < facDim; i++) {
        iFac(item, i) -= learnRate * (-2.0*wt*diff*uFac(u, i) + 2.0*iReg*iFac(item, i));
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
        std::cout << "ModelInvPopMF::train trainSeed: " << trainSeed
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


void ModelInvPopMF::trainSGDPar(const Data &data, Model &bestModel, 
    std::unordered_set<int>& invalidUsers,
    std::unordered_set<int>& invalidItems) {

  std::cout << "\nModelInvPopMF::trainSGDPar trainSeed: " << trainSeed;
  
  int nnz = data.trainNNZ;
  int iter, bestIter = -1, i; 
  double bestObj, prevObj;
  double bestValRMSE, prevValRMSE;
  std::vector<int> trainUsers, trainItems;

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

  
  for (int u = 0; u < trainMat->nrows; u++) {
    if (invalidUsers.count(u) == 0) {
      trainUsers.push_back(u);
    }
  }
  nTrainUsers = trainUsers.size();

  for (int item = 0; item < trainMat->ncols; item++) {
    if (invalidItems.count(item) == 0) {
      trainItems.push_back(item);
    }
  }
  nTrainItems = trainItems.size();

  double sumPopScore = 0;
  for (auto& u: trainUsers) {
    invPopU[u] = userFreq[u]/((double)nTrainItems);//((double)nTrainItems)/userFreq[u];
    sumPopScore += invPopU[u];
  }
  for (auto& u: trainUsers) {
    invPopU[u] = invPopU[u]/sumPopScore;
  }

  sumPopScore = 0;
  for (auto& item: trainItems) {
    invPopI[item] = itemFreq[item]/((double)nTrainUsers);//((double)nTrainUsers)/itemFreq[item];
    sumPopScore += invPopI[item];
  }
  for (auto& item: trainItems) {
    invPopI[item] = invPopI[item]/sumPopScore;
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
  
  std::cout << "\nModelMF::trainSGDPar trainSeed: " << trainSeed 
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
  std::cout << "maxThreads: " << maxThreads << std::endl;

  //create maxThreads partitions of users
  int usersPerPart = trainUsers.size()/maxThreads;
  std::cout << "train users: " << trainUsers.size() << " usersPerPart: " 
    << usersPerPart << std::endl;
  std::vector<std::unordered_set<int>> usersPart(maxThreads); 
  int currPart = 0;
  for (int i = 0; i < trainUsers.size(); i++) {
    usersPart[currPart].insert(trainUsers[i]);
    if (i != 0 && i % usersPerPart == 0) {
      if (currPart != maxThreads - 1) {
        std::cout << "currPart: " << currPart << " i: " << i << std::endl;
        currPart++;
      }
    }
  }

  //create maxThreads partitions of items
  int itemsPerPart = trainItems.size()/maxThreads;
  std::cout << "train items: " << trainItems.size() << " itemsPerPart: " 
    << itemsPerPart << std::endl;
  std::vector<std::unordered_set<int>> itemsPart(maxThreads); 
  currPart = 0;
  for (int i = 0; i < trainItems.size(); i++) {
    itemsPart[currPart].insert(trainItems[i]);
    if (i != 0 && i % itemsPerPart == 0) {
      if (currPart != maxThreads - 1) {
        std::cout << "currPart: " << currPart << " i: " << i << std::endl;
        currPart++;
      }
    }
  }

  std::vector<std::pair<int, int>> updateSeq;

  for (iter = 0; iter < maxIter; iter++) {  
    
    start = std::chrono::system_clock::now();
    
    for (int k = 0; k < maxThreads; k++) {
      sgdUpdateBlockSeq(maxThreads, updateSeq, mt);
#pragma omp parallel for
      for (int t = 0; t < maxThreads; t++) {
        const auto& users = usersPart[updateSeq[t].first];
        const auto& items = itemsPart[updateSeq[t].second];
        for (const auto& u: users) {
          for (int ii = trainMat->rowptr[u]; ii < trainMat->rowptr[u+1]; ii++) {
            
            int item = trainMat->rowind[ii];
            if (items.count(item) == 0) {
              continue;
            }

            float itemRat = trainMat->rowval[ii];
            float r_ui_est = uFac.row(u).dot(iFac.row(item));
            float diff = itemRat - r_ui_est;
            
            float wt = invPopI[item]; 
            if (itemFreq[item] > userFreq[u]) {
              wt = invPopU[u];
            }
            //wt = (1.0 + rhoRMS*wt);
            wt = (1.0/(1.0 + rhoRMS*wt));

            //update user
            for (int i = 0; i < facDim; i++) {
              uFac(u, i) -= learnRate * (-2.0*wt*diff*iFac(item, i) + 2.0*uReg*uFac(u, i));
            }
          
            //update item
            for (int i = 0; i < facDim; i++) {
              iFac(item, i) -= learnRate * (-2.0*wt*diff*uFac(u, i) + 2.0*iReg*iFac(item, i));
            }

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
        std::cout << "ModelInvPopMF::trainSGDPar trainSeed: " << trainSeed
                  << " Iter: " << iter << " Objective: " << std::scientific << prevObj 
                  << " Train RMSE: " << RMSE(data.trainMat, invalidUsers, invalidItems)
                  << " Val RMSE: " << prevValRMSE
                  << " subIterDuration: " << subIterDuration
                  << std::endl;
      }

      if (iter % SAVE_ITER == 0 || iter == maxIter - 1) {
        std::string modelFName = std::string(data.prefix);
        //bestModel.saveFacs(modelFName);
      }

    }
     
  }
      
  //save best model found till now
  std::string modelFName = std::string(data.prefix);
  //bestModel.saveFacs(modelFName);

  std::cout << "\nBest model validation RMSE: " << bestModel.RMSE(data.valMat, 
      invalidUsers, invalidItems);
}




