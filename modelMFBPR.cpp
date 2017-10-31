#include "modelMFBPR.h"



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

  for (int iter = 0; iter < maxIter; iter++) {

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
      double expCoeff = -1.0 /(1.0 + std::exp(r_uij));

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
                << " subIterDuration: " << subIterDuration
                << std::endl;
    }
  }

  std::cout  << "\nBest model validation HR: " << bestModel.hitRate(data,
      invalidUsers, invalidItems, data.valMat) << std::endl;

}
