#include "modelDropoutMF.h"


double ModelDropoutMF::estRating(int user, int item) {
  double rat = 0;
  int updMinRank = std::min(userRankMap[user], itemRankMap[item]);
  int candFacDim = ((float)facDim/8.0);
  candFacDim = candFacDim > 0 ? candFacDim: 1;
  //updMinRank = std::min(updMinRank, candFacDim);
  for (int k = 0; k < candFacDim; k++) {
    rat += uFac(user, k)*iFac(item, k);
  }
  for (int k = candFacDim; k < updMinRank; k++) {
    rat += 0.5*uFac(user, k)*iFac(item, k);
  }
  for (int k = updMinRank; k < facDim; k++) {
    rat += 0.15*uFac(user, k)*iFac(item, k);
  }
  return rat;
}


//have objective corresponding to this
/*
double ModelDropoutMF::estRating(int user, int item, int minRank) {
  double rat = 0;
  minRank = std::min(minRank, std::min(userRankMap[user], itemRankMap[item]));
  for (int k = 0; k < minRank; k++) {
    rat += uFac(user, k)*iFac(item, k);
  }
  return rat;
}

double ModelDropoutMF::RMSE(gk_csr_t *mat, std::unordered_set<int>& invalidUsers,
    std::unordered_set<int>& invalidItems, int minRank) {
  int nnz;
  double r_ui, r_ui_est, diff, rmse;

  nnz = 0;
  rmse = 0;

#pragma omp parallel for reduction(+:rmse, nnz) private(r_ui, r_ui_est, diff)
  for (int u = 0; u < nUsers; u++) {
    
    //skip if invalid user
    if (invalidUsers.count(u) > 0) {
      //found and skip
      continue;
    }

    for (int ii = mat->rowptr[u]; ii < mat->rowptr[u+1]; ii++) {
      
      int item = mat->rowind[ii];
      //skip if invalid item
      if (invalidItems.count(item) > 0 || item >= nItems) {
        //found and skip
        continue;
      }
      
      r_ui     = mat->rowval[ii];
      r_ui_est = estRating(u, item, minRank);
      diff     = r_ui - r_ui_est;
      rmse     += diff*diff;
      nnz++;
    }
  }

  rmse = sqrt(rmse/nnz);
  
  return rmse;
}


double ModelDropoutMF::objective(const Data& data, std::unordered_set<int>& invalidUsers,
    std::unordered_set<int>& invalidItems, int minRank) {

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

      float itemRat = trainMat->rowval[ii];
      double diff = itemRat - estRating(u, item, minRank);
      rmse += diff*diff;
    }
    int uMinRank = std::min(minRank, userRankMap[u]);
    for (int k = 0; k < uMinRank; k++) {
      uRegErr += uFac(u, k)*uFac(u, k);
    }
  }
  uRegErr = uRegErr*uReg;

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
  }
  iRegErr = iRegErr*iReg;

  obj = rmse + uRegErr + iRegErr;
    
  //std::cout <<"\nrmse: " << std::scientific << rmse << " uReg: " << uRegErr 
  //  << " iReg: " << iRegErr << std::endl; 

  return obj;
}
*/

/*
bool ModelDropoutMF::isTerminateModel(ModelDropoutMF& bestModel, const Data& data, int iter,
    int& bestIter, double& bestObj, double& prevObj, double& bestValRMSE,
    double& prevValRMSE, std::unordered_set<int>& invalidUsers, 
    std::unordered_set<int>& invalidItems, int minRank) {
  bool ret = false;
  double currObj = objective(data, invalidUsers, invalidItems, minRank);
  double currValRMSE = -1;
  
  if (data.valMat) {
    currValRMSE = RMSE(data.valMat, invalidUsers, invalidItems, minRank);
  } else {
    std::cerr << "\nNo validation data" << std::endl;
    exit(0);
  }
  
  //nan check
  if (currObj != currObj || currValRMSE != currValRMSE) {
    std::cout << "Found nan " << std::endl;
    //half learning rate
    if (learnRate > 1e-5) {
      //replace current model by best model
      *this = bestModel;
      learnRate = learnRate/2;
      return false;
    } else {
      return true;
    };
  }
    
  if (currValRMSE < bestValRMSE) {
    bestModel = *this;
    bestValRMSE = currValRMSE;
    bestIter = iter;
  }

  if (iter - bestIter >= 100) {
    //half the learning rate
    if (learnRate > 1e-5) {
      learnRate = learnRate/2;
    }
  }

  if (iter - bestIter >= CHANCE_ITER) {
    //can't improve validation RMSE after 500 iterations
    printf("\nNOT CONVERGED: bestIter:%d bestObj: %.10e bestValRMSE: %.10e"
        " currIter:%d currObj: %.10e currValRMSE: %.10e", 
        bestIter, bestObj, bestValRMSE, iter, currObj, currValRMSE);
    ret = true;
  }
  
  if (fabs(prevObj - currObj) < EPS) {
    //convergence
    printf("\nConverged in iteration: %d prevObj: %.10e currObj: %.10e"
        " bestValRMSE: %.10e", iter, prevObj, currObj, bestValRMSE); 
    ret = true;
  }

  /*
  if (fabs(prevValRMSE - currValRMSE) < 0.0001) {
    printf("\nvalidation RMSE in iteration: %d prev: %.10e curr: %.10e" 
        " bestValRMSE: %.10e", iter, prevValRMSE, currValRMSE, bestValRMSE); 
    ret = true;
  }
  */
/*
  prevObj = currObj;
  prevValRMSE = currValRMSE;

  return ret;
}
*/

void ModelDropoutMF::trainSGDAdapPar(const Data &data, ModelDropoutMF &bestModel, 
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
  
  std::cout << "\nModelMF::trainSGDAdapPar trainSeed: " << trainSeed 
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

  for (int rankInd = 0; rankInd < ranks.size(); rankInd++) {
    int minRank = ranks[rankInd];
    
    prevObj = objective(data, invalidUsers, invalidItems);
    bestObj = prevObj;
    bestValRMSE = prevValRMSE = RMSE(data.valMat, invalidUsers, invalidItems);
    
    std::cout << "\nObj aftr svd: " << prevObj << " Train RMSE: " 
      << RMSE(data.trainMat, invalidUsers, invalidItems) << " Val RMSE: " 
      << bestValRMSE;
    std::cout << "minRank: " << minRank << std::endl; 

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
              
              int updMinRank = std::min(minRank, std::min(userRankMap[u], itemRankMap[item]));
              int prevRankInd = 0;
              if (rankInd > 0) {
                prevRankInd = ranks[rankInd-1];
              }

              float itemRat = trainMat->rowval[ii];
              //compute rating based on minRank factors

              float r_ui_est = adapDotProd(uFac, iFac, u, item, updMinRank);
              float diff = itemRat - r_ui_est;
              
              //update user
              for (int i = prevRankInd; i < updMinRank; i++) {
                uFac(u, i) -= learnRate * (-2.0*diff*iFac(item, i) + 2.0*uReg*uFac(u, i));
              }
            
              //update item
              for (int i = prevRankInd; i < updMinRank; i++) {
                iFac(item, i) -= learnRate * (-2.0*diff*uFac(u, i) + 2.0*iReg*iFac(item, i));
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
          std::cout << "trainSGDAdapPar " 
                    << " Iter: " << iter << " Obj: " << std::scientific << prevObj 
                    << " Train: " << RMSE(data.trainMat, invalidUsers, invalidItems)
                    << " Val: " << prevValRMSE
                    << " subIterDur: " << subIterDuration
                    << std::endl;
        }

        if (iter % SAVE_ITER == 0 || iter == maxIter - 1) {
          std::string modelFName = std::string(data.prefix);
          bestModel.saveFacs(modelFName);
        }

      }
       
    }
   
    *this = bestModel;
  }    
  
  //save best model found till now
  std::string modelFName = std::string(data.prefix);
  bestModel.saveFacs(modelFName);

  std::cout << "\nBest model validation RMSE: " << bestModel.RMSE(data.valMat, 
      invalidUsers, invalidItems);
}


void ModelDropoutMF::trainSGDProbPar(const Data &data, ModelDropoutMF &bestModel, 
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
  
  std::cout << "\nModelMF::trainSGDAdapPar trainSeed: " << trainSeed 
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

  prevObj = objective(data, invalidUsers, invalidItems);
  bestObj = prevObj;
  bestValRMSE = prevValRMSE = RMSE(data.valMat, invalidUsers,
      invalidItems);
    
  std::cout << "\nObj aftr svd: " << prevObj << " Train RMSE: " 
    << RMSE(data.trainMat, invalidUsers, invalidItems) << " Val RMSE: " 
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
            float diff = itemRat - r_ui_est;
            
            //update user
            for (int i = 0; i < updMinRank; i++) {
              uFac(u, i) -= learnRate * (-2.0*diff*iFac(item, i) + 2.0*uReg*uFac(u, i));
            }
          
            //update item
            for (int i = 0; i < updMinRank; i++) {
              iFac(item, i) -= learnRate * (-2.0*diff*uFac(u, i) + 2.0*iReg*iFac(item, i));
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
                  << " Train: " << RMSE(data.trainMat, invalidUsers, invalidItems)
                  << " Val: " << prevValRMSE
                  << " subIterDur: " << subIterDuration
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


void ModelDropoutMF::trainSGDProbOrderedPar(const Data &data, ModelDropoutMF &bestModel, 
    std::unordered_set<int>& invalidUsers,
    std::unordered_set<int>& invalidItems) {

  std::cout << "\nModelDropoutMF::trainSGDProbOrderedPar trainSeed: " << trainSeed;
  
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
  
  std::cout << "\nModelDropoutMF:trainSGDProbOrderedPar trainSeed: " << trainSeed 
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

  prevObj = objective(data, invalidUsers, invalidItems);
  bestObj = prevObj;
  bestValRMSE = prevValRMSE = RMSE(data.valMat, invalidUsers,
      invalidItems);
    
  std::cout << "\nObj aftr svd: " << prevObj << " Train RMSE: " 
    << RMSE(data.trainMat, invalidUsers, invalidItems) << " Val RMSE: " 
    << bestValRMSE << std::endl;

  if (rhoRMS < EPS) {
    rhoRMS = 0.3;
  }
  
  std::cout << "rhorms: " << rhoRMS << std::endl;

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

            if (rdis(mt) <= 0.5) {
              int candFacDim = ((float)facDim/8.0);
              candFacDim = candFacDim > 0 ? candFacDim: 1;
              updMinRank = std::min(updMinRank, candFacDim);
            }

            float itemRat = trainMat->rowval[ii];
            //compute rating based on minRank factors

            float r_ui_est = adapDotProd(uFac, iFac, u, item, updMinRank);
            float diff = itemRat - r_ui_est;
            
            //update user
            for (int i = 0; i < updMinRank; i++) {
              uFac(u, i) -= learnRate * (-2.0*diff*iFac(item, i) + 2.0*uReg*uFac(u, i));
            }
          
            //update item
            for (int i = 0; i < updMinRank; i++) {
              iFac(item, i) -= learnRate * (-2.0*diff*uFac(u, i) + 2.0*iReg*iFac(item, i));
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
        std::cout << "trainSGDProbOrderedPar " 
                  << " Iter: " << iter << " Obj: " << std::scientific << prevObj 
                  << " Train: " << RMSE(data.trainMat, invalidUsers, invalidItems)
                  << " Val: " << prevValRMSE
                  << " subIterDur: " << subIterDuration
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


void ModelDropoutMF::trainSGDOnlyOrderedPar(const Data &data, ModelDropoutMF &bestModel, 
    std::unordered_set<int>& invalidUsers,
    std::unordered_set<int>& invalidItems) {

  std::cout << "\nModelDropoutMF::trainSGDOnlyOrderedPar trainSeed: " << trainSeed;
  
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
  
  std::cout << "\nModelDropoutMF:trainSGDOnlyOrderedPar trainSeed: " << trainSeed 
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

  prevObj = objective(data, invalidUsers, invalidItems);
  bestObj = prevObj;
  bestValRMSE = prevValRMSE = RMSE(data.valMat, invalidUsers,
      invalidItems);
    
  std::cout << "\nObj aftr svd: " << prevObj << " Train RMSE: " 
    << RMSE(data.trainMat, invalidUsers, invalidItems) << " Val RMSE: " 
    << bestValRMSE << std::endl;

  if (rhoRMS < EPS) {
    rhoRMS = 0.3;
  }
  
  std::cout << "rhorms: " << rhoRMS << std::endl;

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
            
            //int updMinRank = std::min(userRankMap[u], itemRankMap[item]);

            //if (updMinRank != facDim && rdis(mt) <= rhoRMS) {
            //  updMinRank = facDim;
            //}
            int updMinRank = facDim;
            if (rdis(mt) <= 0.5) {
              int candFacDim = ((float)facDim/8.0);
              candFacDim = candFacDim > 0 ? candFacDim: 1;
              updMinRank = candFacDim;
            }

            float itemRat = trainMat->rowval[ii];
            //compute rating based on minRank factors

            float r_ui_est = adapDotProd(uFac, iFac, u, item, updMinRank);
            float diff = itemRat - r_ui_est;
            
            //update user
            for (int i = 0; i < updMinRank; i++) {
              uFac(u, i) -= learnRate * (-2.0*diff*iFac(item, i) + 2.0*uReg*uFac(u, i));
            }
          
            //update item
            for (int i = 0; i < updMinRank; i++) {
              iFac(item, i) -= learnRate * (-2.0*diff*uFac(u, i) + 2.0*iReg*iFac(item, i));
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
        std::cout << "trainSGDOnlyOrderedPar " 
                  << " Iter: " << iter << " Obj: " << std::scientific << prevObj 
                  << " Train: " << RMSE(data.trainMat, invalidUsers, invalidItems)
                  << " Val: " << prevValRMSE
                  << " subIterDur: " << subIterDuration
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
