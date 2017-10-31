#include "modelBPRPoissonDropout.h"

void ModelBPRPoissonDropout::initCDFRanks() {
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


/*
double ModelBPRPoissonDropout::estRating(int user, int item) {
  //std::cout << "ModelPoissonDropout::estRating (" << user << "," << item << ") " << std::endl;
  double rat = 0;
  bool isUMinFreq = userFreq[user] < itemFreq[item];
  //double scaleFreq = isUMinFreq ? (userFreq[user] - minFreq) / (maxFreq - minFreq) : (itemFreq[item] - minFreq) / (maxFreq - minFreq);
  double scaleFreq = isUMinFreq ? (userFreq[user] - meanFreq) / stdFreq : (itemFreq[item] - meanFreq) / stdFreq;
  double sigmPc = 1.0/(1.0+exp(-rhoRMS*(scaleFreq-alpha)));
  
  int updMinRank = std::ceil(sigmPc*((double)facDim));
  assert(updMinRank > 0);

  if (updMinRank > facDim) {
    updMinRank = facDim;
  }

  for (int k = 0; k < updMinRank; k++) {
    rat += uFac(user, k)*iFac(item, k);
  }

  return rat;
}
*/


double ModelBPRPoissonDropout::estRating(int user, int item) {
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


void ModelBPRPoissonDropout::train(const Data& data, Model &bestModel, 
    std::unordered_set<int>& invalidUsers,
    std::unordered_set<int>& invalidItems) {

  int u, pI, nI, nTrainInversions = 0;
  float itemRat, r_ui_est, r_uj_est;
  double bestRecall, valRecall;
  double bestHR, valHR;
  double subIterDuration = 0;
  int iter, bestIter = -1;
  int nnz = data.trainNNZ;
  
  //random engine
  std::mt19937 mt(trainSeed);
  
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

  std::unordered_set<int> trainUsers, trainItems;
  
  for (int u = 0; u < trainMat->nrows; u++) {
    if (invalidUsers.count(u) == 0) {
      trainUsers.insert(u);
    }
  }
  
  for (int item = 0; item < trainMat->ncols; item++) {
    if (invalidItems.count(item) == 0) {
      trainItems.insert(item);
    }
  }

  std::cout << "\nModelBPRPoissonDropout ::train trainSeed: " << trainSeed
    << " invalidUsers: " << invalidUsers.size() << " invalidItems: " 
    << invalidItems.size() << std::endl;
  
  //std::cout << "\nNNZ = " << nnz;
  std::chrono::time_point<std::chrono::system_clock> start, end;
  std::chrono::duration<double> duration;
  
  std::cout << "\n ModelBPRPoissonDropout:train trainSeed: " << trainSeed 
    << " invalidUsers: " << invalidUsers.size()
    << " invalidItems: " << invalidItems.size() << std::endl;

  //get user-item ratings from training data
  const auto uiRatings = getBPRUIRatings(data.trainMat);
  //index to above uiRatings pair
  std::vector<size_t> uiRatingInds(uiRatings.size());
  std::iota(uiRatingInds.begin(), uiRatingInds.end(), 0);

  std::cout << "\nTrain NNZ after removing invalid users and items: " 
    << uiRatings.size() << std::endl;
  
  std::cout << "minFreq: " << minFreq << " maxFreq: " << maxFreq << std::endl;
  std::cout << "rhoRMS: " << rhoRMS << " alpha: " << alpha << std::endl;

  valHR = hitRate(data, invalidUsers, invalidItems, data.valMat);
  std::cout << "\nValidation HR: " << valHR << std::endl;
  
  for (iter = 0; iter < maxIter; iter++) {  
    
    start = std::chrono::system_clock::now();

    if (iter % 10 == 0) {
      //shuffle the user item rating indexes
      std::shuffle(uiRatingInds.begin(), uiRatingInds.end(), mt);
    } else {
      parBlockShuffle(uiRatingInds, mt);
    }

    nTrainInversions = 0;

    for (const auto& ind: uiRatingInds) {
      u = std::get<0>(uiRatings[ind]);
      pI = std::get<1>(uiRatings[ind]);

      //sample -ve item
      nI = sampleNegItem(u, data.trainMat, trainItems);
      if (-1 == nI) {
        //failed to sample negativ item
        continue;
      }
      
      double scaleFreq = (userFreq[u] - meanFreq) / stdFreq; 
      bool isUMinFreq = userFreq[u] < itemFreq[pI] ? userFreq[u] < itemFreq[nI] : false;
      if (!isUMinFreq) {
        if (itemFreq[pI] < itemFreq[nI]) {
          scaleFreq = (itemFreq[pI] - meanFreq) / stdFreq;
        } else {
          scaleFreq = (itemFreq[nI] - meanFreq) / stdFreq; 
        }
      }
      double sigmPc = 1.0/(1.0+exp(-rhoRMS*(scaleFreq-alpha)));
            
      //no. of effective ranks
      int lambda = std::ceil(sigmPc*((double)facDim)); // facDim
      assert(lambda > 0);

      std::poisson_distribution<> pdis(lambda);
      int updRank = pdis(mt);
      if (updRank > facDim) {
        updRank = facDim;
      }
      if (updRank < EPS) {
        updRank = 1;
      }
      
      r_ui_est = adapDotProd(uFac, iFac, u, pI, updRank);
      r_uj_est = adapDotProd(uFac, iFac, u, nI, updRank);

      if (r_uj_est - r_ui_est > EPS) {
        nTrainInversions++;
      }

      double r_uij = r_ui_est - r_uj_est;
      double expCoeff = -1.0 /(1.0 + std::exp(r_uij));

      //update user
      for (int i = 0; i < updRank; i++) {
        uFac(u, i) -= learnRate*( (expCoeff*(iFac(pI, i) - iFac(nI, i)))
                                  + 2.0*uReg*(uFac(u, i)) );
      }

      //update item
      for (int i = 0; i < updRank; i++) {
        iFac(pI, i) -= learnRate*( (expCoeff*uFac(u, i)) + 2.0*iReg*iFac(pI, i));
        iFac(nI, i) -= learnRate*( (-expCoeff*uFac(u,i)) + 2.0*iReg*iFac(nI, i));
      }

    }

    end = std::chrono::system_clock::now();  
   
    duration =  end - start;
    subIterDuration = duration.count();

    if (iter % OBJ_ITER == 0 || iter == maxIter-1) {
      if (isTerminateModelHR(bestModel, data, iter, bestIter, bestHR, valHR,
            invalidUsers, invalidItems)) {
        break;
      }
    }

    if (iter % DISP_ITER == 0) {
      std::cout << "ModelBPRPoissonDropout::train trainSeed: " << trainSeed
                << " Iter: " << iter << " HR: " << std::scientific << valHR
                << " best HR: " << bestHR
                << " nTrainInversions: " << nTrainInversions
                << " subIterDuration: " << subIterDuration
                << std::endl;
    }
     
  }

  //save best model found till now
  std::string modelFName = std::string(data.prefix);
  //bestModel.saveFacs(modelFName);
}


void ModelBPRPoissonDropout::trainSigmoid(const Data& data, Model &bestModel, 
    std::unordered_set<int>& invalidUsers,
    std::unordered_set<int>& invalidItems) {

  int u, pI, nI, nTrainInversions = 0;
  float itemRat, r_ui_est, r_uj_est;
  double bestRecall, valRecall;
  double bestHR, valHR;
  double subIterDuration = 0;
  int iter, bestIter = -1;
  int nnz = data.trainNNZ;
  
  //random engine
  std::mt19937 mt(trainSeed);
  
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

  std::unordered_set<int> trainUsers, trainItems;
  
  for (int u = 0; u < trainMat->nrows; u++) {
    if (invalidUsers.count(u) == 0) {
      trainUsers.insert(u);
    }
  }
  
  for (int item = 0; item < trainMat->ncols; item++) {
    if (invalidItems.count(item) == 0) {
      trainItems.insert(item);
    }
  }

  std::cout << "\nModelBPRPoissonDropout::trainSigmoid trainSeed: " << trainSeed
    << " invalidUsers: " << invalidUsers.size() << " invalidItems: " 
    << invalidItems.size() << std::endl;
  
  //std::cout << "\nNNZ = " << nnz;
  std::chrono::time_point<std::chrono::system_clock> start, end;
  std::chrono::duration<double> duration;
  
  std::cout << "\n ModelBPRPoissonDropout:train trainSeed: " << trainSeed 
    << " invalidUsers: " << invalidUsers.size()
    << " invalidItems: " << invalidItems.size() << std::endl;

  //get non-zero ratings from training data
  const auto uiRatings = getBPRUIRatings(data.trainMat);
  //index to above uiRatings pair
  std::vector<size_t> uiRatingInds(uiRatings.size());
  std::iota(uiRatingInds.begin(), uiRatingInds.end(), 0);

  std::cout << "\nTrain NNZ after removing invalid users and items: " 
    << uiRatings.size() << std::endl;

  std::cout << "minFreq: " << minFreq << " maxFreq: " << maxFreq << std::endl;
  std::cout << "rhoRMS: " << rhoRMS << " alpha: " << alpha << std::endl;

  for (iter = 0; iter < maxIter; iter++) {  
    
    start = std::chrono::system_clock::now();

    if (iter % 10 == 0) {
      //shuffle the user item rating indexes
      std::shuffle(uiRatingInds.begin(), uiRatingInds.end(), mt);
    } else {
      parBlockShuffle(uiRatingInds, mt);
    }

    nTrainInversions = 0;

    for (const auto& ind: uiRatingInds) {
      u = std::get<0>(uiRatings[ind]);
      pI = std::get<1>(uiRatings[ind]);

      //sample -ve item
      nI = sampleNegItem(u, data.trainMat, trainItems);
      if (-1 == nI) {
        //failed to sample negativ item
        continue;
      }
      
      double scaleFreq = (userFreq[u] - meanFreq) / stdFreq; 
      bool isUMinFreq = userFreq[u] < itemFreq[pI] ? userFreq[u] < itemFreq[nI] : false;
      if (!isUMinFreq) {
        if (itemFreq[pI] < itemFreq[nI]) {
          scaleFreq = (itemFreq[pI] - meanFreq) / stdFreq;
        } else {
          scaleFreq = (itemFreq[nI] - meanFreq) / stdFreq; 
        }
      }
      double sigmPc = 1.0/(1.0+exp(-rhoRMS*(scaleFreq-alpha)));
            
      int updMinRank = std::ceil(sigmPc*((double)facDim));
      if (updMinRank < EPS) {
        updMinRank = 1;
      }
      if (updMinRank > facDim) {
        updMinRank = facDim;
      }
      
      r_ui_est = adapDotProd(uFac, iFac, u, pI, updMinRank);
      r_uj_est = adapDotProd(uFac, iFac, u, nI, updMinRank);

      if (r_uj_est - r_ui_est > EPS) {
        nTrainInversions++;
      }

      double r_uij = r_ui_est - r_uj_est;
      double expCoeff = -1.0 /(1.0 + std::exp(r_uij));

      //update user
      for (int i = 0; i < updMinRank; i++) {
        uFac(u, i) -= learnRate*( (expCoeff*(iFac(pI, i) - iFac(nI, i)))
                                  + 2.0*uReg*(uFac(u, i)) );
      }

      //update item
      for (int i = 0; i < updMinRank; i++) {
        iFac(pI, i) -= learnRate*( (expCoeff*uFac(u, i)) + 2.0*iReg*iFac(pI, i));
        iFac(nI, i) -= learnRate*( (-expCoeff*uFac(u,i)) + 2.0*iReg*iFac(nI, i));
      }

    }

    end = std::chrono::system_clock::now();  
   
    duration =  end - start;
    subIterDuration = duration.count();

    if (iter % OBJ_ITER == 0 || iter == maxIter-1) {
      if (isTerminateModelHR(bestModel, data, iter, bestIter, bestHR, valHR,
            invalidUsers, invalidItems)) {
        break;
      }
    }

    if (iter % DISP_ITER == 0) {
      std::cout << "ModelBPRPoissonDropout::train trainSeed: " << trainSeed
                << " Iter: " << iter << " HR: " << std::scientific << valHR
                << " best HR: " << bestHR
                << " nTrainInversions: " << nTrainInversions
                << " subIterDuration: " << subIterDuration
                << std::endl;
    }
     
  }

  //save best model found till now
  std::string modelFName = std::string(data.prefix);
  //bestModel.saveFacs(modelFName);
}


