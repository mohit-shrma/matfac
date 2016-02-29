#include "modelMFBias.h"


double ModelMFBias::estRating(int user, int item) {


}


void ModelMFBias::train(const Data& data, Model& bestModel, 
    std::unordered_set<int>& invalidUsers, 
    std::unordered_set<int>& invalidItems) {

  std::cout << "\nModelMFBias::train trainSeed: " << trainSeed;
  

  //TODO: global bias
  int nnz = data.trainNNZ;
 
  //TODO:modify these methods
  std::cout << "\nObj b4 svd: " << objective(data) 
    << " Train RMSE: " << RMSE(data.trainMat) 
    << " Train nnz: " << nnz << std::endl;
  
  std::chrono::time_point<std::chrono::system_clock> startSVD, endSVD;
  startSVD = std::chrono::system_clock::now();
  //initialization with svd of the passed matrix
  //svdFrmSvdlibCSR(data.trainMat, facDim, uFac, iFac); 
  //svdUsingLapack(data.trainMat, facDim, uFac, iFac);
  //svdFrmCSR(data.trainMat, facDim, uFac, iFac);
  //svdFrmCSRColAvg(data.trainMat, facDim, uFac, iFac);
  
  endSVD = std::chrono::system_clock::now();
  std::chrono::duration<double> durationSVD =  (endSVD - startSVD) ;
  std::cout << "\nsvd duration: " << durationSVD.count();

  int u, iter, subIter, bestIter;
  int item;
  float itemRat;
  double bestObj, prevObj;

  gk_csr_t *trainMat = data.trainMat;

  //array to hold user and item gradients
  std::vector<double> uGrad (facDim, 0);
  std::vector<double> iGrad (facDim, 0);
  
  prevObj = objective(data);
  std::cout << "\nObj aftr svd: " << prevObj << " Train RMSE: " << RMSE(data.trainMat);

  std::chrono::time_point<std::chrono::system_clock> start, end;
  
  std::vector<std::unordered_set<int>> uISet(nUsers);
  genStats(trainMat, uISet, std::to_string(trainSeed));
  getInvalidUsersItems(trainMat, uISet, invalidUsers, invalidItems);
  
  std::cout << "\nModelMF::train trainSeed: " << trainSeed 
    << " invalidUsers: " << invalidUsers.size()
    << " invalidItems: " << invalidItems.size() << std::endl;

  //random engine
  std::mt19937 mt(trainSeed);
  //get user-item ratings from training data
  auto uiRatings = getUIRatings(trainMat);
  
  for (iter = 0; iter < maxIter; iter++) {  
    start = std::chrono::system_clock::now();

    //shuffle the user item ratings
    std::shuffle(uiRatings.begin(), uiRatings.end(), mt);

    end = std::chrono::system_clock::now();  
    
    for (auto&& uiRating: uiRatings) {
      //get user, item and rating
      u       = std::get<0>(uiRating);
      item    = std::get<1>(uiRating);
      itemRat = std::get<2>(uiRating);
      
      //skip if u in invalidUsers    
      /*
      auto search = invalidUsers.find(u);
      if (search != invalidUsers.end()) {
        //found and skip
        continue;
      }
      */
      
      /*
      search = invalidItems.find(item);
      if (search != invalidItems.end()) {
        //found and skip
        continue;
      }
      */

      //compute user gradient
      computeUGrad(u, item, itemRat, uGrad);

      //update user
      //updateAdaptiveFac(uFac[u], uGrad, uGradsAcc[u]); 
      updateFac(uFac[u], uGrad); 

      //TODO: update user bias
      
      //compute item gradient
      computeIGrad(u, item, itemRat, iGrad);

      //update item
      //updateAdaptiveFac(iFac[item], iGrad, iGradsAcc[item]);
      updateFac(iFac[item], iGrad);

      //TODO: update item bias

    }
    //check objective
    if (iter % OBJ_ITER == 0 || iter == maxIter-1) {
      if (isTerminateModel(bestModel, data, iter, bestIter, bestObj, prevObj)) {
        break; 
      }
      std::cout << "\nModelMFBias::train trainSeed: " << trainSeed
                << " Iter: " << iter << " Objective: " << std::scientific << prevObj 
                << " Train RMSE: " << RMSE(data.trainMat) 
                << std::endl;
      std::chrono::duration<double> duration =  (end - start) ;
      std::cout << "\nsub duration: " << duration.count() << std::endl;
      //save best model found till now
      std::string modelFName = "ModelFull_" + std::to_string(trainSeed);
      bestModel.save(modelFName);
    }
  }

}




