#include "modelMF.h"


void ModelMF::train(const Data &data, Model &bestModel, 
    std::unordered_set<int>& invalidUsers,
    std::unordered_set<int>& invalidItems) {

  //copy passed known factors
  //uFac = data.origUFac;
  //iFac = data.origIFac;
  
  std::cout << "\nModelMF::train trainSeed: " << trainSeed;
  
  int nnz = data.trainNNZ;
  
  std::cout << "\nObj b4 svd: " << objective(data) 
    << " Train RMSE: " << RMSE(data.trainMat) 
    << " Train nnz: " << nnz << std::endl;
  
  std::chrono::time_point<std::chrono::system_clock> startSVD, endSVD;
  startSVD = std::chrono::system_clock::now();
  //initialization with svd of the passed matrix
  //svdFrmSvdlibCSR(data.trainMat, facDim, uFac, iFac, false); 
  
  endSVD = std::chrono::system_clock::now();
  std::chrono::duration<double> durationSVD =  (endSVD - startSVD) ;
  std::cout << "\nsvd duration: " << durationSVD.count();

  int u, item, iter, bestIter = -1; 
  float itemRat;
  double diff, r_ui_est;
  double bestObj, prevObj;
  double bestValRMSE, prevValRMSE;

  gk_csr_t *trainMat = data.trainMat;

  std::vector<std::unordered_set<int>> uISet(nUsers);
  genStats(trainMat, uISet, std::to_string(trainSeed));
  getInvalidUsersItems(trainMat, uISet, invalidUsers, invalidItems);
 
  //std::cout << "\nNNZ = " << nnz;
  prevObj = objective(data, invalidUsers, invalidItems);
  bestObj = prevObj;
  std::cout << "\nObj aftr svd: " << prevObj << " Train RMSE: " 
    << RMSE(data.trainMat, invalidUsers, invalidItems);


  std::chrono::time_point<std::chrono::system_clock> start, end;
  std::chrono::duration<double> duration;
  
 std::cout << "\nModelMF::train trainSeed: " << trainSeed 
    << " invalidUsers: " << invalidUsers.size()
    << " invalidItems: " << invalidItems.size() << std::endl;
  
  //random engine
  std::mt19937 mt(trainSeed);
  //get user-item ratings from training data
  auto uiRatings = getUIRatings(trainMat, invalidUsers, invalidItems);
  //index to above uiRatings pair
  std::vector<size_t> uiRatingInds(uiRatings.size());
  std::iota(uiRatingInds.begin(), uiRatingInds.end(), 0);


  std::cout << "\nTrain NNZ after removing invalid users and items: " 
    << uiRatings.size() << std::endl;
  double subIterDuration = 0;
  for (iter = 0; iter < maxIter; iter++) {  
    
    //shuffle the user item rating indexes
    std::shuffle(uiRatingInds.begin(), uiRatingInds.end(), mt);

    start = std::chrono::system_clock::now();
    for (auto&& ind: uiRatingInds) {
      //get user, item and rating
      u       = std::get<0>(uiRatings[ind]);
      item    = std::get<1>(uiRatings[ind]);
      itemRat = std::get<2>(uiRatings[ind]);
      
      //std::cout << "\nGradCheck u: " << u << " item: " << item;
      //gradCheck(u, item, itemRat);
      r_ui_est = uFac.row(u).dot(iFac.row(item));
      diff = itemRat - r_ui_est;

      //update user
      for (int i = 0; i < facDim; i++) {
        uFac[u][i] -= learnRate * (-2.0*diff*iFac[item][i] + 2.0*uReg*uFac[u][i]);
      }

    
      //update item
      for (int i = 0; i < facDim; i++) {
        iFac[item][i] -= learnRate * (-2.0*diff*uFac[u][i] + 2.0*iReg*iFac[item][i]);
      }

    }
    end = std::chrono::system_clock::now();  
   
    duration =  end - start;
    subIterDuration = duration.count();

    //check objective
    if (iter % OBJ_ITER == 0 || iter == maxIter-1) {
      if (isTerminateModel(bestModel, data, iter, bestIter, bestObj, prevObj,
            invalidUsers, invalidItems)) {
        break; 
      }

      if (iter % 50 == 0) {
        std::cout << "ModelMF::train trainSeed: " << trainSeed
                  << " Iter: " << iter << " Objective: " << std::scientific << prevObj 
                  << " Train RMSE: " << RMSE(data.trainMat, invalidUsers, invalidItems)
                  << " Val RMSE: " << prevValRMSE
                  << " subIterDuration: " << subIterDuration
                  << std::endl;
      }

      if (iter % 500 == 0 || iter == maxIter - 1) {
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


void ModelMF::trainALS(const Data &data, Model &bestModel, 
    std::unordered_set<int>& invalidUsers,
    std::unordered_set<int>& invalidItems) {

  //copy passed known factors
  //uFac = data.origUFac;
  //iFac = data.origIFac;
  
  std::cout << "\nModelMF::trainALS trainSeed: " << trainSeed;
  
  int nnz = data.trainNNZ;
  
  std::cout << "\nObj b4 svd: " << objective(data) 
    << " Train RMSE: " << RMSE(data.trainMat) 
    << " Train nnz: " << nnz << std::endl;
  
  std::chrono::time_point<std::chrono::system_clock> startSVD, endSVD;
  startSVD = std::chrono::system_clock::now();
  //initialization with svd of the passed matrix
  //svdFrmSvdlibCSR(data.trainMat, facDim, uFac, iFac, false); 
  
  endSVD = std::chrono::system_clock::now();
  std::chrono::duration<double> durationSVD =  (endSVD - startSVD) ;
  std::cout << "\nsvd duration: " << durationSVD.count();

  int u, item, iter, bestIter = -1; 
  float itemRat;
  double diff, r_ui_est;
  double bestObj, prevObj;
  double bestValRMSE, prevValRMSE;

  gk_csr_t *trainMat = data.trainMat;

  //array to hold user and item gradients
  std::vector<double> uGrad (facDim, 0);
  std::vector<double> iGrad (facDim, 0);
 
  //vector to hold user gradient accumulation
  std::vector<std::vector<double>> uGradsAcc (nUsers, 
      std::vector<double>(facDim,0)); 

  //vector to hold item gradient accumulation
  std::vector<std::vector<double>> iGradsAcc (nItems, 
      std::vector<double>(facDim,0)); 

  std::vector<std::unordered_set<int>> uISet(nUsers);
  genStats(trainMat, uISet, std::to_string(trainSeed));
  getInvalidUsersItems(trainMat, uISet, invalidUsers, invalidItems);
 
  //std::cout << "\nNNZ = " << nnz;
  prevObj = objective(data, invalidUsers, invalidItems);
  bestObj = prevObj;
  std::cout << "\nObj aftr svd: " << prevObj << " Train RMSE: " 
    << RMSE(data.trainMat, invalidUsers, invalidItems);


  std::chrono::time_point<std::chrono::system_clock> start, end;
  std::chrono::duration<double> duration;
  
 std::cout << "\nModelMF::trainALS trainSeed: " << trainSeed 
    << " invalidUsers: " << invalidUsers.size()
    << " invalidItems: " << invalidItems.size() << std::endl;
  
  //random engine
  std::mt19937 mt(trainSeed);
  //get user-item ratings from training data
  auto uiRatings = getUIRatings(trainMat, invalidUsers, invalidItems);
  //index to above uiRatings pair
  std::vector<size_t> uiRatingInds(uiRatings.size());
  std::iota(uiRatingInds.begin(), uiRatingInds.end(), 0);


  std::cout << "\nTrain NNZ after removing invalid users and items: " 
    << uiRatings.size() << std::endl;
  double subIterDuration = 0;
  for (iter = 0; iter < maxIter; iter++) { 

    start = std::chrono::system_clock::now();

    //update users
#pragma omp parallel for
    for (int u = 0; u < nUsers; u++) {
      
      //skip if invalid
      if (invalidUsers.count(u) > 0) {
        continue;
      } 
      
      Eigen::MatrixXf YTY = Eigen::MatrixXf::Zero(facDim, facDim);
      Eigen::VectorXf b = Eigen::VectorXf::Zero(facDim);
      for (int ii = trainMat->rowptr[u]; ii < trainMat->rowptr[u+1]; ii++) {
        int item = trainMat->rowind[ii];
        float rating = trainMat->rowval[ii];
        if (rating > 0) {
          //update YTY
          for (int j = 0; j < facDim; j++) {
            for (int k = 0; k < facDim; k++) {
              YTY(j, k) += iFac[item][j]*iFac[item][k];
            }
            b(j) += rating*iFac[item][j];
          }
        }
      }
      
      //add u reg
      for (int j = 0; j < facDim; j++) {
        YTY(j,j) += uReg;
      }
      
      //solve YTY * u = b
      Eigen::VectorXf ufac =  YTY.ldlt().solve(b); 
      //Eigen::VectorXf ufac =  YTY.lu().solve(b); 
      for (int j = 0; j < facDim; j++) {
        uFac[u][j] = ufac[j];
      } 
    } 

    //update items
#pragma omp parallel for
    for (int item = 0; item < nItems; item++) {
      //skip if invalid
      if (invalidItems.count(item) > 0) {
        continue;
      }
      Eigen::MatrixXf YTY = Eigen::MatrixXf::Zero(facDim, facDim);
      Eigen::VectorXf b = Eigen::VectorXf::Zero(facDim);
      
      for (int uu = trainMat->colptr[item]; uu < trainMat->colptr[item+1]; uu++) {
        int user = trainMat->colind[uu];
        float rating = trainMat->colval[uu];
        if (rating > 0) {
          //update YTY
          for (int j = 0; j < facDim; j++) {
            for (int k = 0; k < facDim; k++) {
              YTY(j, k) += uFac[user][j]*uFac[user][k];
            }
            b(j) += rating*uFac[user][j];
          }
        }
      }
      
      //add item reg
      for (int j = 0; j < facDim; j++) {
        YTY(j, j) += iReg;
      }
      
      //solve YTY * v = b
      Eigen::VectorXf ifac =  YTY.ldlt().solve(b); 
      //Eigen::VectorXf ifac =  YTY.lu().solve(b); 
      for (int j = 0; j < facDim; j++) {
        iFac[item][j] = ifac[j];
      } 

    } 

    end = std::chrono::system_clock::now();  
   
    duration =  end - start;
    subIterDuration = duration.count();

    //check objective
    if (iter % OBJ_ITER == 0 || iter == maxIter-1) {
      if (isTerminateModel(bestModel, data, iter, bestIter, bestObj, prevObj,
            invalidUsers, invalidItems)) {
        break; 
      }

      if (iter % 50 == 0) {
        std::cout << "ModelMF::trainALS trainSeed: " << trainSeed
                  << " Iter: " << iter << " Objective: " << std::scientific << prevObj 
                  << " Train RMSE: " << RMSE(data.trainMat, invalidUsers, invalidItems)
                  << " Val RMSE: " << prevValRMSE
                  << " subIterDuration: " << subIterDuration
                  << std::endl;
      }

      if (iter % 500 == 0 || iter == maxIter - 1) {
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


void ModelMF::hogTrain(const Data &data, Model &bestModel, 
    std::unordered_set<int>& invalidUsers,
    std::unordered_set<int>& invalidItems) {

  //copy passed known factors
  //uFac = data.origUFac;
  //iFac = data.origIFac;
  
  std::cout << "\nModelMF::hogTrain trainSeed: " << trainSeed;
  
  int nnz = data.trainNNZ;
  
  std::cout << "\nObj b4 svd: " << objective(data) 
    << " Train RMSE: " << RMSE(data.trainMat) 
    << " Train nnz: " << nnz << std::endl;
  
  std::chrono::time_point<std::chrono::system_clock> startSVD, endSVD;
  startSVD = std::chrono::system_clock::now();
  //initialization with svd of the passed matrix
  //svdFrmSvdlibCSR(data.trainMat, facDim, uFac, iFac, false); 
  
  endSVD = std::chrono::system_clock::now();
  std::chrono::duration<double> durationSVD =  (endSVD - startSVD) ;
  std::cout << "\nsvd duration: " << durationSVD.count();

  int iter, bestIter = -1; 
  double bestObj, prevObj;
  double bestValRMSE, prevValRMSE;

  gk_csr_t *trainMat = data.trainMat;

 
  //vector to hold user gradient accumulation
  std::vector<std::vector<double>> uGradsAcc (nUsers, 
      std::vector<double>(facDim,0)); 

  //vector to hold item gradient accumulation
  std::vector<std::vector<double>> iGradsAcc (nItems, 
      std::vector<double>(facDim,0)); 

  std::vector<std::unordered_set<int>> uISet(nUsers);
  genStats(trainMat, uISet, std::to_string(trainSeed));
  getInvalidUsersItems(trainMat, uISet, invalidUsers, invalidItems);
  
  //std::cout << "\nNNZ = " << nnz;
  prevObj = objective(data, invalidUsers, invalidItems);
  bestObj = prevObj;
  std::cout << "\nObj aftr svd: " << prevObj << " Train RMSE: " << RMSE(data.trainMat, 
      invalidUsers, invalidItems);

  std::chrono::time_point<std::chrono::system_clock> start, end;
  std::chrono::duration<double> duration;
  
  
  std::cout << "\nModelMF::hogTrain trainSeed: " << trainSeed 
    << " invalidUsers: " << invalidUsers.size()
    << " invalidItems: " << invalidItems.size() << std::endl;
  
  //random engine
  std::mt19937 mt(trainSeed);
  //get user-item ratings from training data
  auto uiRatings = getUIRatings(trainMat, invalidUsers, invalidItems);
  //index to above uiRatings pair
  std::vector<size_t> uiRatingInds(uiRatings.size());
  std::iota(uiRatingInds.begin(), uiRatingInds.end(), 0);


  std::cout << "\nTrain NNZ after removing invalid users and items: " 
    << uiRatings.size() << std::endl;
  double subIterDuration = 0;
  for (iter = 0; iter < maxIter; iter++) {  
    
    //shuffle the user item rating indexes
    std::shuffle(uiRatingInds.begin(), uiRatingInds.end(), mt);

    start = std::chrono::system_clock::now();
    const int indsSz = uiRatingInds.size();
#pragma omp parallel for
    for (int k = 0; k < indsSz; k++) {
      auto ind = uiRatingInds[k];
      //get user, item and rating
      int u       = std::get<0>(uiRatings[ind]);
      int item    = std::get<1>(uiRatings[ind]);
      float itemRat = std::get<2>(uiRatings[ind]);
      
      double r_ui_est = uFac.row(u).dot(iFac.row(item));
      double diff = itemRat - r_ui_est;

      //update user
      uFac.row(u) -= learnRate*(-2.0*diff*iFac.row(item) + 2.0*uReg*uFac.row(u));
      //for (int i = 0; i < facDim; i++) {
      //  uFac[u][i] -= learnRate*(-2.0*diff*iFac[item][i] + 2.0*uReg*uFac[u][i]);
      //}


    
      //update item
      //for (int i = 0; i < facDim; i++) {
      //  iFac[item][i] -= learnRate*(-2.0*diff*uFac[u][i] + 2.0*iReg*iFac[item][i]);
      //}
      iFac.row(item) -= learnRate*(-2.0*diff*uFac.row(u) + 2.0*iReg*iFac.row(item));
    }
    end = std::chrono::system_clock::now();  
   
    duration =  end - start;
    subIterDuration = duration.count();

    //check objective
    if (iter % OBJ_ITER == 0 || iter == maxIter-1) {
      if (isTerminateModel(bestModel, data, iter, bestIter, bestObj, prevObj,
            invalidUsers, invalidItems)) {
        break; 
      }

      if (iter % 50 == 0) {
        std::cout << "ModelMF::hogTrain trainSeed: " << trainSeed
                  << " Iter: " << iter << " Objective: " << std::scientific << prevObj 
                  << " Train RMSE: " << RMSE(data.trainMat, invalidUsers, invalidItems)
                  << " Val RMSE: " << prevValRMSE
                  << " sub duration: " << subIterDuration
                  << std::endl;
      }

      if (iter % 500 == 0 || iter == maxIter - 1) {
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


