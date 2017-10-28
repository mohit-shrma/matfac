#include "modelIncrement.h"


void ModelIncrement::init() {
  currRankMapU = std::vector<int>(nUsers, 1);
  currRankMapI = std::vector<int>(nItems, 1);
}


double ModelIncrement::estRating(int user, int item) {
  double rat = 0;
  //no. of effective ranks
  int updMinRank = currRankMapI[item] < currRankMapU[user] ? currRankMapI[item] : currRankMapU[user];
  for (int k = 0; k < updMinRank; k++) {
    rat += uFac(user, k)*iFac(item, k);
  }
  return rat;
}


void ModelIncrement::train(const Data& data, Model &bestModel, 
    std::unordered_set<int>& invalidUsers,
    std::unordered_set<int>& invalidItems) {

  std::cout << "\nModelIncrement::train trainSeed: " << trainSeed;
  
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
  std::vector<int> prevRankMapU; 
  std::vector<int> prevRankMapI;
  std::vector<double> prevValRMSEU;
  std::vector<double> prevValRMSEI;
  std::vector<bool> toIncrementU;
  std::vector<bool> toIncrementI;
  
  Eigen::MatrixXf uFacPrev; 
  Eigen::MatrixXf iFacPrev;

  uFacPrev = Eigen::MatrixXf(nUsers, facDim);
  iFacPrev = Eigen::MatrixXf(nItems, facDim);
  uFacPrev = uFac;
  iFacPrev = iFac;
  prevRankMapU = std::vector<int>(nUsers, 1);
  prevRankMapI = std::vector<int>(nItems, 1);
  prevValRMSEU = std::vector<double>(nUsers, 10.0);
  prevValRMSEI = std::vector<double>(nItems, 10.0);
  toIncrementU = std::vector<bool>(nUsers, true);
  toIncrementI = std::vector<bool>(nItems, true);

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
 

  std::cout << "created invalid users and items..." << std::endl;

  for (int u = 0; u < trainMat->nrows; u++) {
    if (invalidUsers.count(u) == 0) {
      trainUsers.push_back(u);
      prevValRMSEU[u] = RMSEUser(data.graphMat, invalidUsers, invalidItems, 
          u); 
    }
  }
 
  std::cout << "got train users..." << std::endl;

  for (int item = 0; item < trainMat->ncols; item++) {
    if (invalidItems.count(item) == 0) {
      trainItems.push_back(item);
    }
    prevValRMSEI[item] = RMSEItem(data.graphMat, invalidUsers, invalidItems, 
        item);
  }
  
  std::cout << "got train items..." << std::endl;

  //std::cout << "\nNNZ = " << nnz;
  std::chrono::time_point<std::chrono::system_clock> start, end;
  std::chrono::duration<double> duration;
  
  std::cout << "\nModelIncrement:train trainSeed: " << trainSeed 
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
  std::cout << "rhoRMS: " << rhoRMS << " alpha: " << alpha << std::endl;
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
            
            int updMinRank =  currRankMapI[item] < currRankMapU[u] ? currRankMapI[item]: currRankMapU[u];

            float itemRat = trainMat->rowval[ii];
            //compute rating based on minRank factors

            //std::cout << "update rank: (" << u << "," << item << "): " << updMinRank << std::endl;
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
        std::cout << "trainInc " 
                  << " Iter: " << iter << " Obj: " << std::scientific << prevObj 
                  << " Train: " << RMSE(data.trainMat, invalidUsers, invalidItems)
                  << " Val: " << prevValRMSE
                  << " subIterDur: " << subIterDuration
                  << std::endl;
      }

      if (iter % SAVE_ITER == 0 || iter == maxIter - 1) {
        std::string modelFName = std::string(data.prefix);
        //bestModel.saveFacs(modelFName);
      }

    }


    //check whether to increase rank        
    if (iter > 0 && iter % INC_ITER == 0) {
      int incU = 0;
#pragma omp parallel for reduction(+:incU)
      for (int i = 0; i < trainUsers.size(); i++) {
        int u = trainUsers[i];
        if (toIncrementU[u]) {
          //check if improved validation rmse for user
          double currRMSE = RMSEUser(data.graphMat, invalidUsers, invalidItems, u);
          if (currRMSE < 0) {
            //not enough samples in probe mat
            toIncrementU[u] = false;
          }
          if (currRMSE < prevValRMSEU[u] && currRankMapU[u] < facDim) {
            prevRankMapU[u] = currRankMapU[u];
            prevValRMSEU[u] = currRMSE;
            currRankMapU[u] += 5;
            if (currRankMapU[u] >= facDim) {
              toIncrementU[u] = false; 
              currRankMapU[u] = facDim;
            }
            incU++;
          } else {
            toIncrementU[u] = false;
            currRankMapU[u] = prevRankMapU[u];
            uFac.row(u) = uFacPrev.row(u);
          }
        }
      }
      

      int incI = 0;
#pragma omp parallel for reduction(+:incI)
      for (int i = 0; i < trainItems.size(); i++) {
        int item = trainItems[i];
        if (toIncrementI[item]) {
          //check if improved validation rmse for user
          double currRMSE = RMSEItem(data.graphMat, invalidUsers, invalidItems, item);
          if (currRMSE < 0) {
            //not enough samples in probe mat
            toIncrementI[item] = false;
          }
          if (currRMSE < prevValRMSEI[item] && currRankMapI[item] < facDim) {
            prevRankMapI[item] = currRankMapI[item];
            prevValRMSEI[item] = currRMSE;
            currRankMapI[item] += 5;
            if (currRankMapI[item] >= facDim) {
              toIncrementI[item] = false;
              currRankMapI[item] = facDim;
            }
            incI++;
          } else {
            toIncrementI[item] = false;
            currRankMapI[item] = prevRankMapI[item];
            iFac.row(item) = iFacPrev.row(item);
          }
        }
      }

      if (iter % INC_ITER == 0 && (incU > 0 || incI > 0)) {
        std::cout << "iter: " << iter << " Incremented users: " << incU 
          << " items: " << incI << std::endl;
      }

      uFacPrev = uFac;
      iFacPrev = iFac;
    }

     
    if (0 == iter) {
      uFacPrev = uFac;
      iFacPrev = iFac;
    }

  }
 
  //save best model found till now
  std::string modelFName = std::string(data.prefix);
  //bestModel.saveFacs(modelFName);

  //std::cout << "\nBest model validation RMSE: " << bestModel.RMSE(data.valMat, 
  //    invalidUsers, invalidItems) << std::endl;
}


