#include "modelDropoutMFBias.h"

double ModelDropoutMFBias::estRating(int user, int item) {
  double rat = 0;
  int minRank = std::min(userRankMap[user], itemRankMap[item]);
  for (int k = 0; k < minRank; k++) {
    rat += uFac(user, k)*iFac(item, k);
  }
  rat += uBias[user] + iBias[item];
  return rat;
}


//have objective corresponding to this
double ModelDropoutMFBias::estRating(int user, int item, int minRank) {
  double rat = 0;
  minRank = std::min(minRank, std::min(userRankMap[user], itemRankMap[item]));
  for (int k = 0; k < minRank; k++) {
    rat += uFac(user, k)*iFac(item, k);
  }
  rat += uBias[user] + iBias[item];
  return rat;
}


double ModelDropoutMFBias::objective(const Data& data, std::unordered_set<int>& invalidUsers,
    std::unordered_set<int>& invalidItems, int minRank) {

  double rmse = 0, uRegErr = 0, iRegErr = 0, obj = 0;
  double uBiasReg = 0, iBiasReg = 0;
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

      float itemRat = trainMat->rowval[ii];
      double diff = itemRat - estRating(u, item, minRank);
      rmse += diff*diff;
    }
    int uMinRank = std::min(minRank, userRankMap[u]);
    for (int k = 0; k < uMinRank; k++) {
      uRegErr += uFac(u, k)*uFac(u, k);
    }
    uBiasReg += uBias[u]*uBias[u];
  }
  uRegErr = uRegErr*uReg;
  uBiasReg = uBiasReg*uReg;

#pragma omp parallel for reduction(+: iRegErr)
  for (int item = 0; item < nItems; item++) {
    //skip if invalid item
    if (invalidItems.count(item) > 0) {
      //found and skip
      continue;
    }
    int iMinRank = std::min(minRank, itemRankMap[item]);
    for (int k = 0; k < iMinRank; k++) {
      iRegErr += iFac(item, k)*iFac(item, k);
    }
    iBiasReg += iBias[item]*iBias[item];
  }
  iRegErr = iRegErr*iReg;
  iBiasReg = iBiasReg*iReg;

  //obj = rmse + uRegErr + iRegErr;
  obj = rmse + uRegErr + iRegErr + uBiasReg + iBiasReg;
    
  //std::cout <<"\nrmse: " << std::scientific << rmse << " uReg: " << uRegErr 
  //  << " iReg: " << iRegErr << std::endl; 

  return obj;
}


void ModelDropoutMFBias::trainSGDProbPar(const Data &data, ModelDropoutMF &bestModel, 
    std::unordered_set<int>& invalidUsers,
    std::unordered_set<int>& invalidItems) {

  std::cout << "\nModelDropoutMF::trainSGDAdapPar trainSeed: " << trainSeed;
  
  int nnz = data.trainNNZ;
  
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
    
    for (int k = userRankMap[u]; k < facDim; k++) {
      uFac(u, k) = 0;
    }
    
  }
  
  for (int item = 0; item < trainMat->ncols; item++) {
    if (invalidItems.count(item) == 0) {
      trainItems.push_back(item);
    }
    
    for (int k = itemRankMap[item]; k < facDim; k++) {
      iFac(item, k) = 0;
    }
    
  }

  //std::cout << "\nNNZ = " << nnz;
  std::chrono::time_point<std::chrono::system_clock> start, end;
  std::chrono::duration<double> duration;
  
  std::cout << "\nModelDropoutMFBias::trainSGDAdapPar trainSeed: " << trainSeed 
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

  prevObj = objective(data, invalidUsers, invalidItems, facDim);
  bestObj = prevObj;
  bestValRMSE = prevValRMSE = RMSE(data.valMat, invalidUsers,
      invalidItems, facDim);
    
  std::cout << "\nObj aftr svd: " << prevObj << " Train RMSE: " 
    << RMSE(data.trainMat, invalidUsers, invalidItems, facDim) << " Val RMSE: " 
    << bestValRMSE << std::endl;

  if (rhoRMS < EPS) {
    rhoRMS = 0.3;
  }

  for (iter = 0; iter < maxIter; iter++) {  
    
    start = std::chrono::system_clock::now();
    

    for (int k = 0; k < maxThreads; k++) {
      sgdUpdateBlockSeq(maxThreads, updateSeq, mt);
#pragma omp parallel for
      for (int t = 0; t < maxThreads; t++) {
        const auto& users = usersPart[updateSeq[t].first];
        const auto& items = itemsPart[updateSeq[t].second];
        std::uniform_real_distribution<> rdis(0, 1.0);
        for (const auto& u: users) {
          for (int ii = trainMat->rowptr[u]; ii < trainMat->rowptr[u+1]; ii++) {
            
            int item = trainMat->rowind[ii];
            if (items.count(item) == 0) {
              continue;
            }
            
            int updMinRank = std::min(userRankMap[u], itemRankMap[item]);

            if (updMinRank != facDim && rdis(mt) <= rhoRMS) {
              updMinRank = facDim;
            }

            float itemRat = trainMat->rowval[ii];
            //compute rating based on minRank factors

            float r_ui_est = adapDotProd(uFac, iFac, u, item, updMinRank);
            r_ui_est += uBias[u] + iBias[item];
            
            float diff = itemRat - r_ui_est;
            
            //update user
            for (int i = 0; i < updMinRank; i++) {
              uFac(u, i) -= learnRate * (-2.0*diff*iFac(item, i) + 2.0*uReg*uFac(u, i));
            }
          
            //update user bias
            uBias[u] -= learnRate*(-2.0*diff + 2.0*uReg*uBias[u]);
            
            //update item
            for (int i = 0; i < updMinRank; i++) {
              iFac(item, i) -= learnRate * (-2.0*diff*uFac(u, i) + 2.0*iReg*iFac(item, i));
            }
      
            //update item bias
            iBias[item] -= learnRate*(-2.0*diff + 2.0*iReg*iBias[item]);
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
              bestValRMSE, prevValRMSE, invalidUsers, invalidItems, facDim)) {
          break; 
        }
      } else {
        //TODO
        std::cerr << "Need to override following method" << std::endl;
        break;
        /*
        if (isTerminateModel(bestModel, data, iter, bestIter, bestObj, prevObj,
              invalidUsers, invalidItems, minRank)) {
          break; 
        }
        */
      }
       
      if (iter % DISP_ITER == 0) {
        std::cout << "trainSGDProbPar " 
                  << " Iter: " << iter << " Obj: " << std::scientific << prevObj 
                  << " Train: " << RMSE(data.trainMat, invalidUsers, invalidItems, facDim)
                  << " Val: " << prevValRMSE
                  << " subIterDur: " << subIterDuration
                  << std::endl;
      }

      if (iter % SAVE_ITER == 0 || iter == maxIter - 1) {
        std::string modelFName = std::string(data.prefix);
        bestModel.save(modelFName);
      }

    }
     
  }
 
  
  //save best model found till now
  std::string modelFName = std::string(data.prefix);
  bestModel.save(modelFName);

  std::cout << "\nBest model validation RMSE: " << bestModel.RMSE(data.valMat, 
      invalidUsers, invalidItems, facDim);
}



