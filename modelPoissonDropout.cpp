#include "modelPoissonDropout.h"

//TODO: userRankMap contains percentile
/*
double ModelPoissonDropout::estRating(int user, int item) {
  //std::cout << "ModelPoissonDropout::estRating (" << user << "," << item << ") " << std::endl;
  double rat = 0;
  bool isUMinFreq = userFreq[user] < itemFreq[item];
  int lambda = std::ceil(isUMinFreq ? userRankMap[user]*(facDim-1) : itemRankMap[item]*(facDim-1));
  if (lambda < EPS) {
    lambda = 1;
  }
  double sumWt = 0;
  int maxFac = facDim-1, maxWt = 0;
  for (int k = 0; k < facDim; k++) {
    double wt = (std::exp(-lambda)* std::pow(lambda, k)) / factorial[k];
    fDimWt[k] = wt;
    if (wt > maxWt) {
      maxWt = wt;
      maxFac = k;
    }
    sumWt += wt;
  }

  //std::cout << std::endl;
  for (int k = 0; k <= maxFac ; k++) {
  //for (int k = 0; k < facDim ; k++) {
    //fDimWt[k] = fDimWt[k]/sumWt;
    //std::cout << fDimWt[k]/sumWt << " ";
    rat += uFac(user, k)*iFac(item, k);
  }
  //std::cout << std::endl;

  return rat;
}
*/


//to be used with train 
double ModelPoissonDropout::estRating(int user, int item) {
  //std::cout << "ModelPoissonDropout::estRating (" << user << "," << item << ") " << std::endl;
  double rat = 0;
  bool isUMinFreq = userFreq[user] < itemFreq[item];
  //double scaleFreq = isUMinFreq ? (userFreq[user] - minFreq) / (maxFreq - minFreq) : (itemFreq[item] - minFreq) / (maxFreq - minFreq);
  double scaleFreq = isUMinFreq ? (userFreq[user] - meanFreq) / stdFreq : (itemFreq[item] - meanFreq) / stdFreq;
  double sigmPc = 1.0/(1.0+exp(-rhoRMS*(scaleFreq-alpha)));

  //no. of effective ranks
  int lambda = std::ceil(sigmPc*((double)facDim)); // facDim
  assert(lambda > 0);

  int k = 0;
  double cdf = 0, wt = 0, lowerRat = 0;
  for (k = 0; k <= cdfRanks[lambda-1] && k < facDim; k++) {
    lowerRat += uFac(user, k)*iFac(item, k);
  }
  rat = lowerRat;

  return rat;
}

void ModelPoissonDropout::initCDFRanks() {
  cdfRanks = std::vector<int>(facDim, 0); 
  double cdf = 0, wt = 0;
  for (int lambda = 1; lambda <= facDim; lambda++) {
    //cdf for effective rank 0
    cdf = std::exp(-lambda)*(std::pow(lambda, 0) / factorial[0]);
    int k = 0;
    for (k = 0; k < facDim; k++) {
      wt = std::exp(-lambda)*(std::pow(lambda, k+1) / factorial[k+1]);
      cdf += wt;
      if (cdf >= 0.99) {
        break;
      }
    }
    cdfRanks[lambda-1] = k; //end index of rank to be use for update
    if (k == facDim) {
      cdfRanks[lambda-1] = k-1; //end index of rank to be use for update
    }
    std::cout << "cdfRank: " << lambda-1 << " " << cdfRanks[lambda-1] << std::endl;
    
  }
  
}


void ModelPoissonDropout::train(const Data& data, Model &bestModel, 
    std::unordered_set<int>& invalidUsers,
    std::unordered_set<int>& invalidItems) {

  std::cout << "\nModelPoissonDropout ::train trainSeed: " << trainSeed;
  
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
  }
  
  for (int item = 0; item < trainMat->ncols; item++) {
    if (invalidItems.count(item) == 0) {
      trainItems.push_back(item);
    }
  }

  //std::cout << "\nNNZ = " << nnz;
  std::chrono::time_point<std::chrono::system_clock> start, end;
  std::chrono::duration<double> duration;
  
  std::cout << "\n ModelPoissonDropout:train trainSeed: " << trainSeed 
    << " invalidUsers: " << invalidUsers.size()
    << " invalidItems: " << invalidItems.size() << std::endl;
  

  //get user-item ratings from training data
  const auto uiRatings = getUIRatings(trainMat, invalidUsers, invalidItems);

  std::cout << "\nTrain NNZ after removing invalid users and items: " 
    << uiRatings.size() << std::endl;
  double subIterDuration = 0;
  
  
  int maxThreads = omp_get_max_threads(); 
  std::cout << "maxThreads: " << maxThreads << std::endl;
  
  //random engine
  std::vector<std::mt19937> rEngines;
  for (int t = 0; t < maxThreads; t++) {
    rEngines.push_back(std::mt19937 (trainSeed+t));
  }
  std::shuffle(trainUsers.begin(), trainUsers.end(), rEngines[0]);
  std::shuffle(trainItems.begin(), trainItems.end(), rEngines[0]);

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
  std::cout << "minFreq: " << minFreq << " maxFreq: " << maxFreq << std::endl;
  std::cout << "rhoRMS: " << rhoRMS << " alpha: " << alpha << std::endl;

  for (iter = 0; iter < maxIter; iter++) {  
    
    start = std::chrono::system_clock::now();

    for (int k = 0; k < maxThreads; k++) {
      sgdUpdateBlockSeq(maxThreads, updateSeq, rEngines[0]);
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

            bool isUMinFreq = userFreq[u] < itemFreq[item];
            //double scaleFreq = isUMinFreq ? (userFreq[u] - minFreq) / (maxFreq - minFreq) : (itemFreq[item] - minFreq) / (maxFreq - minFreq);
            double scaleFreq = isUMinFreq ? (userFreq[u] - meanFreq) / stdFreq : (itemFreq[item] - meanFreq) / stdFreq;
            double sigmPc = 1.0/(1.0+exp(-rhoRMS*(scaleFreq-alpha)));

            //no. of effective ranks
            int lambda = std::ceil(sigmPc*((double)facDim)); // facDim
            assert(lambda > 0);
            
            //int lambda = std::ceil(isUMinFreq ? userRankMap[u]*(facDim-1) : itemRankMap[item]*(facDim-1));
            
            std::poisson_distribution<> pdis(lambda);
            int updRank = pdis(rEngines[t]);
            if (updRank > facDim) {
              updRank = facDim;
            }
            if (updRank < EPS) {
              updRank = 1;
            }

            float itemRat = trainMat->rowval[ii];

            //compute rating based on minRank factors
            //std::cout << "update rank: (" << u << "," << item << "): " << updMinRank << std::endl;

            float r_ui_est = adapDotProd(uFac, iFac, u, item, updRank);
            float diff = itemRat - r_ui_est;
            
            //update user
            for (int i = 0; i < updRank; i++) { 
              uFac(u, i) -= learnRate * (-2.0*diff*iFac(item, i) + 2.0*uReg*uFac(u, i));
            }
          
            //update item
            for (int i = 0; i < updRank; i++) { 
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
      
      start = std::chrono::system_clock::now();
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
      end = std::chrono::system_clock::now();  
      duration =  end - start;

      if (iter % DISP_ITER == 0) {
        std::cout << "trainPoisson " 
                  << " Iter: " << iter << " Obj: " << std::scientific << prevObj 
                  << " Train: " << RMSE(data.trainMat, invalidUsers, invalidItems)
                  << " Val: " << prevValRMSE
                  << " subIterDur: " << subIterDuration 
                  << " objDur: " << duration.count()
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
      invalidUsers, invalidItems) << std::endl;


}


