// TMF
#include "modelDropoutSigmoid.h"
// NOTE: this method is not being used in train method but its identical except
// for assert statement
double ModelDropoutSigmoid::estRating(int user, int item) {
  double rat = 0;
  bool isUMinFreq = userFreq[user] < itemFreq[item];
  double scaleFreq = isUMinFreq ? (userFreq[user] - meanFreq) / stdFreq
                                : (itemFreq[item] - meanFreq) / stdFreq;
  double sigmPc = 1.0 / (1.0 + exp(-rhoRMS * (scaleFreq - alpha)));

  int updMinRank = std::ceil(sigmPc * ((double)facDim));
  assert(updMinRank > 0);

  if (updMinRank > facDim) {
    updMinRank = facDim;
  }

  for (int k = 0; k < updMinRank; k++) {
    rat += uFac(user, k) * iFac(item, k);
  }

  return rat;
}

void ModelDropoutSigmoid::train(const Data &data, Model &bestModel,
                                       std::unordered_set<int> &invalidUsers,
                                       std::unordered_set<int> &invalidItems) {

  std::cout << "\nModelDropoutSigmoid ::train trainSeed: " << trainSeed;

  int nnz = data.trainNNZ;

  std::chrono::time_point<std::chrono::system_clock> startSVD, endSVD;
  startSVD = std::chrono::system_clock::now();
  // initialization with svd of the passed matrix
  // svdFrmSvdlibCSR(data.trainMat, facDim, uFac, iFac, false);

  endSVD = std::chrono::system_clock::now();
  std::chrono::duration<double> durationSVD = (endSVD - startSVD);
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

  // std::cout << "\nNNZ = " << nnz;
  std::chrono::time_point<std::chrono::system_clock> start, end;
  std::chrono::duration<double> duration;

  std::cout << "\n ModelDropoutSigmoid:train trainSeed: " << trainSeed
            << " invalidUsers: " << invalidUsers.size()
            << " invalidItems: " << invalidItems.size() << std::endl;

  // random engine
  std::mt19937 mt(trainSeed);
  // get user-item ratings from training data
  const auto uiRatings = getUIRatings(trainMat, invalidUsers, invalidItems);

  std::cout << "\nTrain NNZ after removing invalid users and items: "
            << uiRatings.size() << std::endl;
  double subIterDuration = 0;

  std::shuffle(trainUsers.begin(), trainUsers.end(), mt);
  std::shuffle(trainItems.begin(), trainItems.end(), mt);

  int maxThreads = omp_get_max_threads();
  std::cout << "maxThreads: " << maxThreads << std::endl;

  // create maxThreads partitions of users
  int usersPerPart = trainUsers.size() / maxThreads;
  std::cout << "train users: " << trainUsers.size()
            << " usersPerPart: " << usersPerPart << std::endl;
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

  // create maxThreads partitions of items
  int itemsPerPart = trainItems.size() / maxThreads;
  std::cout << "train items: " << trainItems.size()
            << " itemsPerPart: " << itemsPerPart << std::endl;
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
  bestValRMSE = prevValRMSE = RMSE(data.valMat, invalidUsers, invalidItems);

  std::cout << "\nObj aftr svd: " << prevObj << " Train RMSE: "
            << RMSE(data.trainMat, invalidUsers, invalidItems)
            << " Val RMSE: " << bestValRMSE << std::endl;
  std::cout << "rhoRMS: " << rhoRMS << " alpha: " << alpha << std::endl;
  std::cout << "minFreq: " << minFreq << " maxFreq: " << maxFreq << std::endl;
  for (iter = 0; iter < maxIter; iter++) {

    start = std::chrono::system_clock::now();

    for (int k = 0; k < maxThreads; k++) {
      sgdUpdateBlockSeq(maxThreads, updateSeq, mt);
#pragma omp parallel for
      for (int t = 0; t < maxThreads; t++) {
        const auto &users = usersPart[updateSeq[t].first];
        const auto &items = itemsPart[updateSeq[t].second];
        for (const auto &u : users) {
          for (int ii = trainMat->rowptr[u]; ii < trainMat->rowptr[u + 1];
               ii++) {

            int item = trainMat->rowind[ii];
            if (items.count(item) == 0) {
              continue;
            }

            bool isUMinFreq = userFreq[u] < itemFreq[item];
            double scaleFreq = isUMinFreq
                                   ? (userFreq[u] - meanFreq) / stdFreq
                                   : (itemFreq[item] - meanFreq) / stdFreq;
            double sigmPc = 1.0 / (1.0 + exp(-rhoRMS * (scaleFreq - alpha)));

            int updMinRank = std::ceil(sigmPc * ((double)facDim));
            if (updMinRank < EPS) {
              updMinRank = 1;
            }
            if (updMinRank > facDim) {
              updMinRank = facDim;
            }

            float itemRat = trainMat->rowval[ii];
            // compute rating based on minRank factors

            // std::cout << "update rank: (" << u << "," << item << "): " <<
            // updMinRank << std::endl;

            float r_ui_est = adapDotProd(uFac, iFac, u, item, updMinRank);
            float diff = itemRat - r_ui_est;

            // update user
            for (int i = 0; i < updMinRank; i++) {
              uFac(u, i) -= learnRate * (-2.0 * diff * iFac(item, i) +
                                         2.0 * uReg * uFac(u, i));
            }

            // update item
            for (int i = 0; i < updMinRank; i++) {
              iFac(item, i) -= learnRate * (-2.0 * diff * uFac(u, i) +
                                            2.0 * iReg * iFac(item, i));
            }
          }
        }
      }
    }

    end = std::chrono::system_clock::now();

    duration = end - start;
    subIterDuration = duration.count();

    // check objective
    if (iter % OBJ_ITER == 0 || iter == maxIter - 1) {

      if (GK_CSR_IS_VAL) {
        if (isTerminateModel(bestModel, data, iter, bestIter, bestObj, prevObj,
                             bestValRMSE, prevValRMSE, invalidUsers,
                             invalidItems)) {
          break;
        }
      } else {
        // TODO
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
        std::cout << "trainSigmoid "
                  << " Iter: " << iter << " Obj: " << std::scientific << prevObj
                  << " Train: "
                  << RMSE(data.trainMat, invalidUsers, invalidItems)
                  << " Val: " << prevValRMSE
                  << " subIterDur: " << subIterDuration << std::endl;
      }

      if (iter % SAVE_ITER == 0 || iter == maxIter - 1) {
        std::string modelFName = std::string(data.prefix);
        // bestModel.saveFacs(modelFName);
      }
    }
  }

  // save best model found till now
  std::string modelFName = std::string(data.prefix);
  // bestModel.saveFacs(modelFName);

  std::cout << "\nBest model validation RMSE: "
            << bestModel.RMSE(data.valMat, invalidUsers, invalidItems)
            << std::endl;
}
