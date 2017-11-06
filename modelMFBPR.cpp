#include "modelMFBPR.h"



void ModelMFBPR::gradCheck() {
  int u = 0, i = 0, j = 1;
  
  float r_ui_est = uFac.row(u).dot(iFac.row(i));
  float r_uj_est = uFac.row(u).dot(iFac.row(j));

  float funcVal = -1*std::log(1.0/(1.0 + std::exp(-(r_ui_est - r_uj_est))));
  float eps = 1e-3;
  
  float r_uij = r_ui_est - r_uj_est;
  float expCoeff = -1.0 /(1.0 + std::exp(r_uij));
  Eigen::VectorXf grad(facDim);
  //update user
  for (int k = 0; k < facDim; k++) {
    grad(k) = expCoeff*(iFac(i, k) - iFac(j, k));
  }
  
  Eigen::VectorXf uplus(uFac.row(u));
  Eigen::VectorXf uminus(uFac.row(u));

  //for (int i = 0; i < facDim; i++) {
    uplus(0) += eps;
    uminus(0) -= eps;
  //}

  r_ui_est = uplus.dot(iFac.row(i));
  r_uj_est = uplus.dot(iFac.row(j));
  float funcValPlus = -1*std::log(1.0/(1.0 + std::exp(-(r_ui_est - r_uj_est))));

  r_ui_est = uminus.dot(iFac.row(i));
  r_uj_est = uminus.dot(iFac.row(j));
  float funcValMinus = -1*std::log(1.0/(1.0 + std::exp(-(r_ui_est - r_uj_est))));

  float approxGrad = (funcValPlus - funcValMinus)/(2*eps);  
  
  float diff = abs(grad(0) - approxGrad);
  std::cout << funcVal << " " << funcValPlus << " " << funcValMinus << std::endl;
  std::cout << grad(0) << " " << approxGrad << " " << diff << std::endl;
}


std::vector<std::tuple<int, int, float>> ModelMFBPR::getBPRUIRatings(gk_csr_t* mat) {
  std::vector<std::tuple<int, int, float>> uiRatings;
  for (int u = 0; u < mat->nrows; u++) {
    for (int ii = mat->rowptr[u]; ii < mat->rowptr[u+1]; ii++) {
      int item = mat->rowind[ii];
      float rating = mat->rowval[ii];
      if (rating > 0) {
        uiRatings.push_back(std::make_tuple(u, item, rating));
      }
    }
  }
  return uiRatings;
}


int ModelMFBPR::sampleNegItem(int u, const gk_csr_t* trainMat,
    std::unordered_set<int>& trainItems) const {

  int j = -1, jj;
  int nUserItems, start, end;
  int nTrainItems = trainMat->ncols;//trainItems.size();
  int32_t *ui_rowind = trainMat->rowind;
  ssize_t *ui_rowptr = trainMat->rowptr;
  float   *ui_rowval = trainMat->rowval;
  int nTry = 0;
  nUserItems = ui_rowptr[u+1] - ui_rowptr[u];
  //sample neg item
  while(nTry < 100) {
    jj = std::rand()%nUserItems;
    if (ui_rowval[jj + ui_rowptr[u]] == 0.0) {
      //explicit 0
      j = ui_rowind[jj + ui_rowptr[u]];
      break;
    } else {
      //search for implicit 0

      if (0 == jj) {
        start = 0;
        end = ui_rowind[ui_rowptr[u]]; //first rated item by u
      } else if (nUserItems-1 == jj) {
        start = ui_rowind[ui_rowptr[u] + jj] + 1; //item next to last rated item
        end = nTrainItems;
      } else {
        start = ui_rowind[ui_rowptr[u] + jj] + 1; //item next to jjth item
        end = ui_rowind[ui_rowptr[u] + jj + 1]; //item rated after jjth item
      }

      //check for empty interval
      if (end - start > 0) {
        j = std::rand()%(end-start) + start;
      } else {
        continue;
      }

      if (trainItems.find(j) != trainItems.end()) {
        break;
      }
    }
    nTry++;
  } //end while

  if (100 == nTry) {
    j = -1;
  }

  return j;
}


void ModelMFBPR::trainHog(const Data& data, Model& bestModel,
    std::unordered_set<int>& invalidUsers,
    std::unordered_set<int>& invalidItems) {

  int nTrainInversions = 0;
  float itemRat;
  double bestRecall, valRecall;
  double bestHR, valHR;
  double subIterDuration = 0;
  int bestIter = -1;
  double trainLoss  = 0;

  std::cout << "\nModelMFBPR::trainHog trainSeed: " << trainSeed << std::endl;
  std::chrono::time_point<std::chrono::system_clock> start, end;

  //random engine
  std::mt19937 mt(trainSeed);

  //get non-zero ratings from training data
  const auto uiRatings = getBPRUIRatings(data.trainMat);
  //index to above uiRatings pair
  std::vector<size_t> uiRatingInds(uiRatings.size());
  std::iota(uiRatingInds.begin(), uiRatingInds.end(), 0);

  std::vector<std::unordered_set<int>> uISet(nUsers);
  genStats(data.trainMat, uISet, std::to_string(trainSeed));
  getInvalidUsersItems(data.trainMat, uISet, invalidUsers, invalidItems);
  for (int u = data.trainMat->nrows; u < data.nUsers; u++) {
    invalidUsers.insert(u);
  }
  for (int item = data.trainMat->ncols; item < data.nItems; item++) {
    invalidItems.insert(item);
  }

  std::cout << "\nModelMFBPR::trainHog trainSeed: " << trainSeed
    << " invalidUsers: " << invalidUsers.size()
    << " invalidItems: " << invalidItems.size() << std::endl;

  gk_csr_t *mat = data.trainMat;
  std::unordered_set<int> trainItems, testItems, valItems;
  for (int item = 0; item < mat->ncols; item++) {
    if (mat->colptr[item+1] - mat->colptr[item] > 0) {
      trainItems.insert(item);
    }
  }

  mat = data.testMat;
  for (int item = 0; item < mat->ncols; item++) {
    if (mat->colptr[item+1] - mat->colptr[item] > 0) {
      testItems.insert(item);
    }
  }

  mat = data.valMat;
  for (int item = 0; item < mat->ncols; item++) {
    if (mat->colptr[item+1] - mat->colptr[item] > 0) {
      valItems.insert(item);
    }
  }

  valHR = hitRate(data, invalidUsers, invalidItems, data.valMat);
  std::cout << "\nValidation HR: " << valHR << std::endl;
  

  for (int iter = 0; iter < maxIter; iter++) {

    start = std::chrono::system_clock::now();

    if (iter % 10 == 0) {
      //shuffle the user item rating indexes
      std::shuffle(uiRatingInds.begin(), uiRatingInds.end(), mt);
    } else {
      parBlockShuffle(uiRatingInds, mt);
    }

    nTrainInversions = 0;
    trainLoss = 0;

#pragma omp parallel for reduction(+:trainLoss,nTrainInversions)
    for (int z = 0; z < uiRatingInds.size(); z++) {
      int ind = uiRatingInds[z];
      int u = std::get<0>(uiRatings[ind]);
      int pI = std::get<1>(uiRatings[ind]);

      float r_ui_est = uFac.row(u).dot(iFac.row(pI));

      //sample -ve item
      int nI = sampleNegItem(u, data.trainMat, trainItems);
      if (-1 == nI) {
        //failed to sample negativ item
        continue;
      }

      float r_uj_est = uFac.row(u).dot(iFac.row(nI));

      if (r_uj_est - r_ui_est > EPS) {
        nTrainInversions++;
      }
      
      double r_uij = r_ui_est - r_uj_est;
      trainLoss += log(1.0 + exp(-r_uij));

      double expCoeff = -1.0 /(1.0 + std::exp(r_uij));
      if (!std::isfinite(expCoeff) || !std::isfinite(trainLoss)) {
        std::cout << "Gradient/trainLoss is not finite (decrease learn rate): " 
          << expCoeff << " " << trainLoss << std::endl;
        exit(0);
      }
      
      //update user
      for (int i = 0; i < facDim; i++) {
        uFac(u, i) -= learnRate*( (expCoeff*(iFac(pI, i) - iFac(nI, i)))
                                  + 2.0*uReg*(uFac(u, i)) );
      }

      //update item
      for (int i = 0; i < facDim; i++) {
        iFac(pI, i) -= learnRate*( (expCoeff*uFac(u, i)) + 2.0*iReg*iFac(pI, i));
        iFac(nI, i) -= learnRate*( (-expCoeff*uFac(u,i)) + 2.0*iReg*iFac(nI, i));
      }

    }


    end = std::chrono::system_clock::now();
    
    if (!std::isfinite(trainLoss)) {
      std::cout << "Training loss is not finite (decrease learn rate): " << trainLoss << std::endl;
      exit(0);
    }
   
    //decay learning rate
    learnRate = learnRate*0.9;

    std::chrono::duration<double> duration =  end - start;
    subIterDuration = duration.count();

    if (iter % OBJ_ITER == 0 || iter == maxIter-1) {
      if (isTerminateModelHR(bestModel, data, iter, bestIter, bestHR, valHR,
            invalidUsers, invalidItems)) {
        break;
      }
    }

    if (iter % DISP_ITER == 0) {
      std::cout << "ModelMFBPR::trainHog trainSeed: " << trainSeed
                << " Iter: " << iter << " HR: " << std::scientific << valHR
                << " best HR: " << bestHR
                << " nTrainInversions: " << nTrainInversions
                << " trainLoss: " << trainLoss
                << " subIterDuration: " << subIterDuration
                << std::endl;
    }
  }

  std::cout  << "\nBest model validation HR: " << bestModel.hitRate(data,
      invalidUsers, invalidItems, data.valMat) << std::endl;

}


void ModelMFBPR::train(const Data& data, Model& bestModel,
    std::unordered_set<int>& invalidUsers,
    std::unordered_set<int>& invalidItems) {

  int u, pI, nI, nTrainInversions = 0;
  float itemRat, r_ui_est, r_uj_est;
  double bestRecall, valRecall;
  double bestHR, valHR;
  double subIterDuration = 0;
  int bestIter = -1;

  std::cout << "\nModelMFBPR::train trainSeed: " << trainSeed << std::endl;
  std::chrono::time_point<std::chrono::system_clock> start, end;

  //random engine
  std::mt19937 mt(trainSeed);

  //get non-zero ratings from training data
  const auto uiRatings = getBPRUIRatings(data.trainMat);
  //index to above uiRatings pair
  std::vector<size_t> uiRatingInds(uiRatings.size());
  std::iota(uiRatingInds.begin(), uiRatingInds.end(), 0);

  std::vector<std::unordered_set<int>> uISet(nUsers);
  genStats(data.trainMat, uISet, std::to_string(trainSeed));
  getInvalidUsersItems(data.trainMat, uISet, invalidUsers, invalidItems);
  for (int u = data.trainMat->nrows; u < data.nUsers; u++) {
    invalidUsers.insert(u);
  }
  for (int item = data.trainMat->ncols; item < data.nItems; item++) {
    invalidItems.insert(item);
  }

  std::cout << "\nModelMFBPR::train trainSeed: " << trainSeed
    << " invalidUsers: " << invalidUsers.size()
    << " invalidItems: " << invalidItems.size() << std::endl;

  gk_csr_t *mat = data.trainMat;
  std::unordered_set<int> trainItems, testItems, valItems;
  for (int item = 0; item < mat->ncols; item++) {
    if (mat->colptr[item+1] - mat->colptr[item] > 0) {
      trainItems.insert(item);
    }
  }

  mat = data.testMat;
  for (int item = 0; item < mat->ncols; item++) {
    if (mat->colptr[item+1] - mat->colptr[item] > 0) {
      testItems.insert(item);
    }
  }

  mat = data.valMat;
  for (int item = 0; item < mat->ncols; item++) {
    if (mat->colptr[item+1] - mat->colptr[item] > 0) {
      valItems.insert(item);
    }
  }

  valHR = hitRate(data, invalidUsers, invalidItems, data.valMat);
  std::cout << "\nValidation HR: " << valHR << std::endl;
  
  double trainLoss  = 0;

  for (int iter = 0; iter < maxIter; iter++) {

    start = std::chrono::system_clock::now();

    if (iter % 10 == 0) {
      //shuffle the user item rating indexes
      std::shuffle(uiRatingInds.begin(), uiRatingInds.end(), mt);
    } else {
      parBlockShuffle(uiRatingInds, mt);
    }

    nTrainInversions = 0;
    trainLoss = 0;
    for (const auto& ind: uiRatingInds) {
      u = std::get<0>(uiRatings[ind]);
      pI = std::get<1>(uiRatings[ind]);

      r_ui_est = uFac.row(u).dot(iFac.row(pI));

      //sample -ve item
      nI = sampleNegItem(u, data.trainMat, trainItems);
      if (-1 == nI) {
        //failed to sample negativ item
        continue;
      }

      r_uj_est = uFac.row(u).dot(iFac.row(nI));

      if (r_uj_est - r_ui_est > EPS) {
        nTrainInversions++;
      }
      
      double r_uij = r_ui_est - r_uj_est;
      trainLoss += std::log(1.0 + std::exp(-r_uij));
      double expCoeff = -1.0 /(1.0 + std::exp(r_uij));

      if (!std::isfinite(expCoeff) || !std::isfinite(trainLoss)) {
        std::cout << "Gradient/trainLoss is not finite (decrease learn rate): " 
          << expCoeff << " " << trainLoss << std::endl;
        exit(0);
      }

      //update user
      for (int i = 0; i < facDim; i++) {
        uFac(u, i) -= learnRate*( (expCoeff*(iFac(pI, i) - iFac(nI, i)))
                                  + 2.0*uReg*(uFac(u, i)) );
      }

      //update item
      for (int i = 0; i < facDim; i++) {
        iFac(pI, i) -= learnRate*( (expCoeff*uFac(u, i)) + 2.0*iReg*iFac(pI, i));
        iFac(nI, i) -= learnRate*( (-expCoeff*uFac(u,i)) + 2.0*iReg*iFac(nI, i));
      }

    }

    end = std::chrono::system_clock::now();

    if (!std::isfinite(trainLoss)) {
      std::cout << "Training loss is not finite (decrease learn rate): " << trainLoss << std::endl;
      exit(0);
    }

    //decay learning rate
    learnRate = learnRate*0.9;
    
    std::chrono::duration<double> duration =  end - start;
    subIterDuration = duration.count();

    if (iter % OBJ_ITER == 0 || iter == maxIter-1) {
      if (isTerminateModelHR(bestModel, data, iter, bestIter, bestHR, valHR,
            invalidUsers, invalidItems)) {
        break;
      }
    }

    if (iter % DISP_ITER == 0) {
      std::cout << "ModelMFBPR::train trainSeed: " << trainSeed
                << " Iter: " << iter << " HR: " << std::scientific << valHR
                << " best HR: " << bestHR
                << " nTrainInversions: " << nTrainInversions
                << " trainLoss: " << trainLoss
                << " subIterDuration: " << subIterDuration
                << std::endl;
    }
  }

  std::cout  << "\nBest model validation HR: " << bestModel.hitRate(data,
      invalidUsers, invalidItems, data.valMat) << std::endl;

}
