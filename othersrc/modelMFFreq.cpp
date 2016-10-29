#include "modelMFFreq.h"

void ModelMFFreq::updateModelInval(
    const std::vector<std::tuple<int, int, float>>& uiRatings,
    std::vector<size_t>& uiRatingInds, std::unordered_set<int>& invalUsers,
    std::unordered_set<int>& invalItems, std::mt19937& mt) {

  //shuffle the user item rating indexes
  std::shuffle(uiRatingInds.begin(), uiRatingInds.end(), mt);
  const int indsSz = uiRatingInds.size();

#pragma omp parallel for
  for (int k = 0; k < indsSz; k++) {
      
    auto ind = uiRatingInds[k];
    //get user, item and rating
    int u       = std::get<0>(uiRatings[ind]);
    int item    = std::get<1>(uiRatings[ind]);
    float itemRat = std::get<2>(uiRatings[ind]);
    double r_ui_est, diff;

    if (invalUsers.find(u) == invalUsers.end()) {
      r_ui_est = dotProd(uFac[u], iFac[item], facDim);
      diff = itemRat - r_ui_est;
      //update user
      for (int i = 0; i < facDim; i++) {
        uFac[u][i] -= learnRate*(-2.0*diff*iFac[item][i] + 2.0*uReg*uFac[u][i]);
      }
    }
 
    if (invalItems.find(item) == invalItems.end()) {
      r_ui_est = dotProd(uFac[u], iFac[item], facDim);
      diff = itemRat - r_ui_est;
      //update item
      for (int i = 0; i < facDim; i++) {
        iFac[item][i] -= learnRate*(-2.0*diff*uFac[u][i] + 2.0*iReg*iFac[item][i]);
      }
    }

  }

}


void ModelMFFreq::updateModel(
    const std::vector<std::tuple<int, int, float>>& uiRatings,
    std::vector<size_t>& uiRatingInds, std::mt19937& mt) {

  //shuffle the user item rating indexes
  std::shuffle(uiRatingInds.begin(), uiRatingInds.end(), mt);
  const int indsSz = uiRatingInds.size();

#pragma omp parallel for
  for (int k = 0; k < indsSz; k++) {
      
    auto ind = uiRatingInds[k];
    //get user, item and rating
    int u       = std::get<0>(uiRatings[ind]);
    int item    = std::get<1>(uiRatings[ind]);
    float itemRat = std::get<2>(uiRatings[ind]);
   
    double r_ui_est = dotProd(uFac[u], iFac[item], facDim);
    double diff = itemRat - r_ui_est;
    //update user
    for (int i = 0; i < facDim; i++) {
      uFac[u][i] -= learnRate*(-2.0*diff*iFac[item][i] + 2.0*uReg*uFac[u][i]);
    }
  
    r_ui_est = dotProd(uFac[u], iFac[item], facDim);
    diff = itemRat - r_ui_est;
    //update item
    for (int i = 0; i < facDim; i++) {
      iFac[item][i] -= learnRate*(-2.0*diff*uFac[u][i] + 2.0*iReg*iFac[item][i]);
    }

  }

}


void ModelMFFreq::updateModelCol(
    const std::vector<std::tuple<int, int, float>>& uiRatings,
    std::vector<size_t>& uiRatingInds, std::mt19937& mt) {

  //shuffle the user item rating indexes
  std::shuffle(uiRatingInds.begin(), uiRatingInds.end(), mt);
  const int indsSz = uiRatingInds.size();

#pragma omp parallel for
  for (int k = 0; k < indsSz; k++) {
      
    auto ind = uiRatingInds[k];
    //get user, item and rating
    int u       = std::get<0>(uiRatings[ind]);
    int item    = std::get<1>(uiRatings[ind]);
    float itemRat = std::get<2>(uiRatings[ind]);
   
    double r_ui_est = dotProd(uFac[u], iFac[item], facDim);
    double diff = itemRat - r_ui_est;
    //update item
    for (int i = 0; i < facDim; i++) {
      iFac[item][i] -= learnRate*(-2.0*diff*uFac[u][i] + 2.0*iReg*iFac[item][i]);
    }

  }

}


void ModelMFFreq::updateModelRow(
    const std::vector<std::tuple<int, int, float>>& uiRatings,
    std::vector<size_t>& uiRatingInds, std::mt19937& mt) {

  //shuffle the user item rating indexes
  std::shuffle(uiRatingInds.begin(), uiRatingInds.end(), mt);
  const int indsSz = uiRatingInds.size();

#pragma omp parallel for
  for (int k = 0; k < indsSz; k++) {
      
    auto ind = uiRatingInds[k];
    //get user, item and rating
    int u       = std::get<0>(uiRatings[ind]);
    int item    = std::get<1>(uiRatings[ind]);
    float itemRat = std::get<2>(uiRatings[ind]);
   
    double r_ui_est = dotProd(uFac[u], iFac[item], facDim);
    double diff = itemRat - r_ui_est;
    //update user
    for (int i = 0; i < facDim; i++) {
      uFac[u][i] -= learnRate*(-2.0*diff*iFac[item][i] + 2.0*uReg*uFac[u][i]);
    }

  }

}


//pass invalidUsers and invalidItems before train
void ModelMFFreq::subTrain(const Data& data, Model& bestModel, 
    std::unordered_set<int>& invalidUsers, 
    std::unordered_set<int>& invalidItems) {

  std::cout << "ModelMFFreq::subTrain" << std::endl;

  int iter, bestIter = -1; 
  double bestObj, prevObj;
  double bestValRMSE, prevValRMSE;
  gk_csr_t *trainMat = data.trainMat;
  prevObj = objective(data);
  bestObj = prevObj;
  std::cout << "\nObj aftr svd: " << prevObj << std::endl;

  std::unordered_set<int> emptySet;

  //random engine
  std::mt19937 mt(trainSeed);
  //get user-item ratings from training data
  auto uiRatings = getUIRatings(trainMat);
  //index to above uiRatings pair
  std::vector<size_t> uiRatingInds(uiRatings.size());
  std::iota(uiRatingInds.begin(), uiRatingInds.end(), 0);
  
  std::cout << "No. of invalid users: " << invalidUsers.size() << std::endl;
  std::cout << "No. of invalid items: " << invalidItems.size() << std::endl;

  for (iter = 0; iter < maxIter; iter++) {  
    //shuffle the user item rating indexes
    std::shuffle(uiRatingInds.begin(), uiRatingInds.end(), mt);
    
    updateModelInval(uiRatings, uiRatingInds, invalidUsers, invalidItems, mt);

    //check objective
    if (iter % OBJ_ITER == 0 || iter == maxIter-1) {
      if (isTerminateModel(bestModel, data, iter, bestIter, bestObj, prevObj,
            emptySet, emptySet)) {
        break; 
      }

      if (iter % 50 == 0) {
        std::cout << "ModelMFFreq::subTrain trainSeed: " << trainSeed
                  << " Iter: " << iter << " Objective: " << std::scientific << prevObj 
                  << " Train RMSE: " << RMSE(data.trainMat)
                  << std::endl;
      }

      if (iter % 500 == 0 || iter == maxIter - 1) {
        bestModel.saveFacs(std::string(data.prefix));
      }

    }

  }

  //save best model found till now
  bestModel.saveFacs(std::string(data.prefix));
}


void ModelMFFreq::train(const Data& data, Model& bestModel, 
    std::unordered_set<int>& invalidUsers, 
    std::unordered_set<int>& invalidItems) {

  std::cout << "ModelMFFreq::train" << std::endl;
  gk_csr_t *trainMat = data.trainMat;

  std::vector<std::unordered_set<int>> uISet(nUsers);
  genStats(trainMat, uISet, std::to_string(trainSeed));
  getInvalidUsersItems(trainMat, uISet, invalidUsers, invalidItems);
  std::unordered_set<int> headItems = getHeadItems(trainMat, 0.8);
  std::unordered_set<int> headUsers = getHeadUsers(trainMat, 0.8);
 
  std::cout << "\nNo. of head items: " << headItems.size() << std::endl;
  std::cout << "No. of head users: " << headUsers.size() << std::endl;

  learnRate = origLearnRate;
  subTrain(data, bestModel, invalidUsers, invalidItems);
  
  //add non-head users items to invalid sets
  std::unordered_set<int> trainInvalUsers;
  std::unordered_set<int> trainInvalItems;
  
  for (int item = 0; item < trainMat->ncols; item++) {
    if (headItems.find(item) == headItems.end()) {
      trainInvalItems.insert(item);
    } 
  }
  for (auto&& item: invalidItems) {
    trainInvalItems.insert(item);
  }
  
  for (int user = 0; user < trainMat->nrows; user++) {
    if (headUsers.find(user) == headUsers.end()) {
      trainInvalUsers.insert(user);
    }
  }
  for (auto&& user: invalidUsers) {
    trainInvalUsers.insert(user);
  }

  learnRate = origLearnRate;
  subTrain(data, bestModel, trainInvalUsers, trainInvalItems); 

  //train only non-head items
  std::cout << "\nupdate non-head items ..." << std::endl;
  trainInvalItems.clear();
  for (int user = 0; user < trainMat->nrows; user++) {
    trainInvalUsers.insert(user);
  }
  for (int item = 0; item < trainMat->ncols; item++) {
    if (headItems.find(item) != headItems.end()) {
      trainInvalItems.insert(item);
    }
  }

  learnRate = origLearnRate;
  subTrain(data, bestModel, trainInvalUsers, trainInvalItems);

  //train only non-head users
  std::cout << "\nupdate non-head users ..." << std::endl;
  trainInvalUsers.clear();
  for (int item = 0; item < trainMat->ncols; item++) {
    trainInvalItems.insert(item);
  }
  for (int user = 0; user < trainMat->nrows; user++) {
    if (headUsers.find(user) != headUsers.end()) {
      trainInvalUsers.insert(user);
    }
  }
  
  learnRate = origLearnRate;
  subTrain(data, bestModel, trainInvalUsers, trainInvalItems);

  learnRate = origLearnRate;
  subTrain(data, bestModel, invalidUsers, invalidItems);
}


