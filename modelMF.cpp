#include "modelMF.h"


void ModelMF::updateAdaptiveFac(std::vector<double> &fac, std::vector<double> &grad,
    std::vector<double> &gradAcc) {
  for (int i = 0; i < facDim; i++) {
    gradAcc[i] = gradAcc[i]*rhoRMS + (1.0-rhoRMS)*grad[i]*grad[i];
    fac[i] -= (learnRate/sqrt(gradAcc[i]+0.0000001)) * grad[i];
  }
}


void ModelMF::computeUGrad(int user, int item, float r_ui, 
        std::vector<double> &uGrad) {
  //estimate and actual rating difference
  double diff = r_ui - dotProd(uFac[user], iFac[item], facDim);

  for (int i = 0; i < facDim; i++) {
    uGrad[i] = -2.0*diff*iFac[item][i] + 2.0*uReg*uFac[user][i];
  }

}


void ModelMF::computeIGrad(int user, int item, float r_ui, 
        std::vector<double> &iGrad) {
  //estimate and actual rating difference
  double diff = r_ui - dotProd(uFac[user], iFac[item], facDim);

  for (int i = 0; i < facDim; i++) {
    iGrad[i] = -2.0*diff*uFac[user][i] + 2.0*iReg*iFac[item][i];
  }

}


void ModelMF::gradCheck(int u, int item, float r_ui) {
  int i;
  std::vector<double> grad (facDim, 0);
  std::vector<double> tempFac (facDim, 0);
  double lossRight, lossLeft, gradE;

  double r_ui_est = estRating(u, item);
  double diff = r_ui - r_ui_est;
  
  //gradient w.r.t. u
  for (i = 0; i < facDim; i++) {
    grad[i] = -2.0*diff*iFac[item][i]; 
  }
  
  //perturb user with +E and compute loss
  tempFac = uFac[u];
  for(auto& v: tempFac) {
    v = v + 0.0001; 
  }
  r_ui_est = dotProd(tempFac, iFac[item], facDim);
  lossRight = (r_ui - r_ui_est)*(r_ui - r_ui_est);
  
  //perturb user with -E and compute loss
  tempFac = uFac[u];
  for(auto& v: tempFac) {
    v = v - 0.0001; 
  }
  r_ui_est = dotProd(tempFac, iFac[item], facDim);
  lossLeft = (r_ui - r_ui_est)*(r_ui - r_ui_est);

  //compute gradient and E dotprod
  gradE = 0;
  for (auto v: grad) {
    gradE += 2.0*v*0.0001;
  }
  
  if (fabs(lossRight - lossLeft - gradE) > 0.0001) {
    printf("\nu: %d lr: %f ll: %f diff: %f div: %f lDiff:%f gradE:%f",
        u, lossRight, lossLeft, lossRight - lossLeft -gradE,
        (lossRight-lossLeft)/gradE, lossRight-lossLeft, gradE); 
  }
  
  //gradient w.r.t. item
  for (i = 0; i < facDim; i++) {
    grad[i] = -2.0*diff*uFac[u][i]; 
  }
  
  //perturb item with +E and compute loss
  tempFac = iFac[item];
  for(auto& v: tempFac) {
    v = v + 0.0001; 
  }
  r_ui_est = dotProd(tempFac, uFac[u], facDim);
  lossRight = (r_ui - r_ui_est)*(r_ui - r_ui_est);
  
  //perturb user with -E and compute loss
  tempFac = iFac[item];
  for(auto& v: tempFac) {
    v = v - 0.0001; 
  }
  r_ui_est = dotProd(tempFac, uFac[u], facDim);
  lossLeft = (r_ui - r_ui_est)*(r_ui - r_ui_est);

  //compute gradient and E dotprod
  gradE = 0;
  for (auto v: grad) {
    gradE += 2.0*v*0.0001;
  }
  
  if (fabs(lossRight - lossLeft - gradE) > 0.0001) {
    printf("\nitem: %d lr: %f ll: %f diff: %f div: %f lDiff:%f gradE:%f",
        item, lossRight, lossLeft, lossRight - lossLeft -gradE,
        (lossRight-lossLeft)/gradE, lossRight-lossLeft, gradE); 
  }

}


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

  int u, item, iter, bestIter; 
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

  //std::cout << "\nNNZ = " << nnz;
  prevObj = objective(data);
  std::cout << "\nObj aftr svd: " << prevObj << " Train RMSE: " << RMSE(data.trainMat);


  std::chrono::time_point<std::chrono::system_clock> start, end;
  std::chrono::duration<double> duration;
  
  std::vector<std::unordered_set<int>> uISet(nUsers);
  genStats(trainMat, uISet, std::to_string(trainSeed));
  getInvalidUsersItems(trainMat, uISet, invalidUsers, invalidItems);
  
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
      r_ui_est = dotProd(uFac[u], iFac[item], facDim);
      diff = itemRat - r_ui_est;

      //compute user gradient
      //computeUGrad(u, item, itemRat, uGrad);
       
      for (int i = 0; i < facDim; i++) {
        uGrad[i] = -2.0*diff*iFac[item][i] + 2.0*uReg*uFac[u][i];
      }

      //update user
      //updateAdaptiveFac(uFac[u], uGrad, uGradsAcc[u]); 
      //updateFac(uFac[u], uGrad); 
      for (int i = 0; i < facDim; i++) {
        uFac[u][i] -= learnRate * uGrad[i];
      }

      r_ui_est = dotProd(uFac[u], iFac[item], facDim);
      diff = itemRat - r_ui_est;
    
      //compute item gradient
      //computeIGrad(u, item, itemRat, iGrad);
      for (int i = 0; i < facDim; i++) {
        iGrad[i] = -2.0*diff*uFac[u][i] + 2.0*iReg*iFac[item][i];
      }

      //update item
      //updateAdaptiveFac(iFac[item], iGrad, iGradsAcc[item]);
      //updateFac(iFac[item], iGrad);
      for (int i = 0; i < facDim; i++) {
        iFac[item][i] -= learnRate * iGrad[i];
      }

    }
    end = std::chrono::system_clock::now();  
   
    duration =  end - start;
    subIterDuration += duration.count();

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
                  << std::endl;
        //std::cout << "\navg sub duration: " << subIterDuration/(iter+1) << std::endl;
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


void ModelMF::uniTrain(const Data &data, Model &bestModel, 
    std::unordered_set<int>& invalidUsers,
    std::unordered_set<int>& invalidItems) {

  //copy passed known factors
  //uFac = data.origUFac;
  //iFac = data.origIFac;
  
  std::cout << "\nModelMF::uniTrain trainSeed: " << trainSeed;
  
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

  int u, item, iter, bestIter; 
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

  //std::cout << "\nNNZ = " << nnz;
  prevObj = objective(data);
  std::cout << "\nObj aftr svd: " << prevObj << " Train RMSE: " << RMSE(data.trainMat);


  std::chrono::time_point<std::chrono::system_clock> start, end;
  std::chrono::duration<double> duration;
  
  std::vector<std::unordered_set<int>> uISet(nUsers);
  genStats(trainMat, uISet, std::to_string(trainSeed));
  getInvalidUsersItems(trainMat, uISet, invalidUsers, invalidItems);
  
  std::cout << "\nModelMF::uniTrain trainSeed: " << trainSeed 
    << " invalidUsers: " << invalidUsers.size()
    << " invalidItems: " << invalidItems.size() << std::endl;
  
  //random engine
  std::mt19937 mt(trainSeed);
  //get user-item ratings from training data
  auto uiBlockRatings = getRandUIRatings(trainMat, 3, trainSeed);
  int nMatBlocks = uiBlockRatings.size();
  std::vector<int> blocks(nMatBlocks);
  std::iota(blocks.begin(), blocks.end(), 0);

  for (auto&& bId : blocks) {
    auto blockRatings = uiBlockRatings[bId];
    std::cout << "blockid: " << bId << " "  << blockRatings.size() << std::endl; 
  }

  double subIterDuration = 0;
  for (iter = 0; iter < maxIter; iter++) {  
    
    //shuffle the user item rating indexes
    std::shuffle(blocks.begin(), blocks.end(), mt);

    start = std::chrono::system_clock::now();
    
    int totalUpd = 0;
    while (totalUpd <= nnz) {
      for (auto&& bId : blocks) {
        auto blockRatings = uiBlockRatings[bId];
        if (blockRatings.size() == 0) {
          continue;
        }
        //shuffle the block
        std::shuffle(blockRatings.begin(), blockRatings.end(), mt);

        //perform 1M updates on block bId
        int upd = 0;
        while (upd < 1000000) {
          for (auto&& uiRating: blockRatings) {
            //get user, item and rating
            u       = std::get<0>(uiRating);
            item    = std::get<1>(uiRating);
            itemRat = std::get<2>(uiRating);
        
            r_ui_est = dotProd(uFac[u], iFac[item], facDim);
            diff = itemRat - r_ui_est;
            for (int i = 0; i < facDim; i++) {
              uGrad[i] = -2.0*diff*iFac[item][i] + 2.0*uReg*uFac[u][i];
            }
            for (int i = 0; i < facDim; i++) {
              uFac[u][i] -= learnRate * uGrad[i];
            }
        
            r_ui_est = dotProd(uFac[u], iFac[item], facDim);
            diff = itemRat - r_ui_est;
            for (int i = 0; i < facDim; i++) {
              iGrad[i] = -2.0*diff*uFac[u][i] + 2.0*iReg*iFac[item][i];
            }
            for (int i = 0; i < facDim; i++) {
              iFac[item][i] -= learnRate * iGrad[i];
            }

            upd++;
            if (upd >= 1000000) {
              break;
            }
          }    
        }
        totalUpd += upd;
      }
    }
    end = std::chrono::system_clock::now();  
   
    duration =  end - start;
    subIterDuration += duration.count();

    //check objective
    if (iter % OBJ_ITER == 0 || iter == maxIter-1) {
      if (isTerminateModel(bestModel, data, iter, bestIter, bestObj, prevObj,
            bestValRMSE, prevValRMSE,
            invalidUsers, invalidItems)) {
        break; 
      }
      std::cout << "\nModelMF::train trainSeed: " << trainSeed
                << " Iter: " << iter << " Objective: " << std::scientific << prevObj 
                << " Train RMSE: " << RMSE(data.trainMat, invalidUsers, invalidItems)
                << " Val RMSE: " << prevValRMSE
                << std::endl;
      std::cout << "\navg sub duration: " << subIterDuration/(iter+1) << std::endl;
      //save best model found till now
      std::string modelFName = std::string(data.prefix);
      bestModel.saveFacs(modelFName);
    }
  
  }

}


void ModelMF::partialTrain(const Data &data, Model &bestModel, 
    std::unordered_set<int>& invalidUsers,
    std::unordered_set<int>& invalidItems) {

  //copy passed known factors
  //uFac = data.origUFac;
  //iFac = data.origIFac;
  
  std::cout << "\nModelMF::partialTrain trainSeed: " << trainSeed;

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

  int u, iter, subIter, bestIter;
  int item, nUserItems, itemInd;
  float itemRat;
  double bestObj, prevObj;

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

  //std::cout << "\nNNZ = " << nnz;
  prevObj = objective(data);
  std::cout << "\nObj aftr svd: " << prevObj << " Train RMSE: " << RMSE(data.trainMat);

  //random engine
  std::mt19937 mt(trainSeed);
  //user dist
  std::uniform_int_distribution<int> uDist(0, nUsers-1);
  std::uniform_int_distribution<int> iDist(0, nItems-1);
  
  std::vector<std::unordered_set<int>> uISet(nUsers);
  //sample 20% of nnz to be excluded from training
  std::cout << "\nIgnoring ui pairs: " << nnz*0.2 << std::endl;
  for (int i = 0; i < nnz*0.2; i++) {
      //sample u
      u  = uDist(mt); 
      //sample item rated by user
      nUserItems =  trainMat->rowptr[u+1] - trainMat->rowptr[u];
      if (0 == nUserItems) {
        i--;
        continue;
      }
      itemInd = iDist(mt)%nUserItems;
      item = trainMat->rowind[trainMat->rowptr[u] + itemInd];
      
      //check if item present in uIset[u]
      auto search = uISet[u].find(item);
      if (search != uISet[u].end()) {
        //found and ignore the current iteration
        i--;
        continue;
      }

      uISet[u].insert(item);
  }

  genStats(trainMat, uISet, std::to_string(trainSeed));
  getInvalidUsersItems(trainMat, uISet, invalidUsers, invalidItems);

  std::cout << "\nModelMF::partialTrain trainSeed: " << trainSeed 
    << " invalidUsers: " << invalidUsers.size()
    << " invalidItems: " << invalidItems.size() << std::endl;

  std::chrono::time_point<std::chrono::system_clock> start, end;

  for (iter = 0; iter < maxIter; iter++) {  
    int setFound  = 0;
    start = std::chrono::system_clock::now();
    for (subIter = 0; subIter < nnz; subIter++) {
      
      //sample u
      u = uDist(mt);
      //skip if u in invalidUsers    
      auto search = invalidUsers.find(u);
      if (search != invalidUsers.end()) {
        //found and skip
        continue;
      }

      //sample item rated by user
      nUserItems =  trainMat->rowptr[u+1] - trainMat->rowptr[u];
      itemInd = iDist(mt)%nUserItems; 
      item = trainMat->rowind[trainMat->rowptr[u] + itemInd];
      
      //check if item present in uIset[u]
      search = uISet[u].find(item);
      if (search != uISet[u].end()) {
        //found and ignore the current iteration
        setFound++;
        continue;
      }

      itemRat = trainMat->rowval[trainMat->rowptr[u] + itemInd]; 
    
      //std::cout << "\nGradCheck u: " << u << " item: " << item;
      //gradCheck(u, item, itemRat);

      //compute user gradient
      computeUGrad(u, item, itemRat, uGrad);

      //update user
      //updateAdaptiveFac(uFac[u], uGrad, uGradsAcc[u]); 
      updateFac(uFac[u], uGrad); 

      //compute item gradient
      computeIGrad(u, item, itemRat, iGrad);

      //update item
      //updateAdaptiveFac(iFac[item], iGrad, iGradsAcc[item]);
      updateFac(iFac[item], iGrad);

    }

    //check objective
    if (iter % OBJ_ITER == 0 || iter == maxIter-1) {
      if (isTerminateModel(bestModel, data, iter, bestIter, bestObj, prevObj)) {
        break; 
      }
      std::cout << "\nModelMF::partialTrain trainSeed: " << trainSeed
                << " Iter: " << iter << " Objective: " << std::scientific << prevObj 
                << " Train RMSE: " << RMSE(data.trainMat) << " skipped: " << setFound
                << std::endl;
      end = std::chrono::system_clock::now();  
      std::chrono::duration<double> duration =  (end - start) ;
      std::cout << "\nsub duration: " << duration.count();
      //save best model found till now
      std::string modelFName = "ModelPartial_" + std::to_string(trainSeed);
      bestModel.save(modelFName);
    }
  
  }
  
  //std::cout << "\nNum Iter: " << iter << " Best Iter: " << bestIter
  //  << " Best obj: " << std::scientific << bestObj ;

}


//train on a submatrix inclusive start exclusive end
void ModelMF::subTrain(const Data &data, Model &bestModel,
                    int uStart, int uEnd, int iStart, int iEnd) {

  //copy passed known factors
  //uFac = data.origUFac;
  //iFac = data.origIFac;
  
  std::cout << "\nModelMF::subTrain";
  
  std::cout << "\nObj b4 svd: " << objectiveSubMat(data, uStart, uEnd, iStart, iEnd) 
    << " Train RMSE: " << subMatRMSE(data.trainMat, uStart, uEnd, iStart, iEnd);
 
  std::chrono::time_point<std::chrono::system_clock> startSVD, endSVD;
  startSVD = std::chrono::system_clock::now();
  //initialization with svd of the passed matrix
  
  //svdFrmCSR(data.trainMat, facDim, uFac, iFac);
  //svdFrmCSRColAvg(data.trainMat, facDim, uFac, iFac);
  svdFrmSvdlibCSR(data.trainMat, facDim, uFac, iFac, false);

  endSVD = std::chrono::system_clock::now();
  std::chrono::duration<double> durationSVD =  (endSVD - startSVD) ;
  std::cout << "\nsvd duration: " << durationSVD.count();

  int u, ii, iter, subIter, bestIter, nSubUsers, nSubItems;
  int item, nUserItems, itemInd;
  float itemRat;
  double bestObj, prevObj;
  int nnz = nnzSubMat(data.trainMat, uStart, uEnd, iStart, iEnd);

  std::cout << "\nsubmat nnz: " << nnz << std::endl;

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

  //set to hold rated items in set
  std::set<int> ratedItems;
  //set to hold unrated items in set
  std::set<int> unRatedItems;
  //set to hold users w/o rating
  std::set<int> unRatedUsers;

  //vector to hold rated items in submatrix for users
  std::vector<std::vector<int>> uRatedItems (nUsers);
  int uNoItemsCount = 0;
  for (u = uStart; u < uEnd; u++) {
    for (ii = trainMat->rowptr[u]; ii < trainMat->rowptr[u+1]; ii++) {
      item = trainMat->rowind[ii];
      if (isInsideBlock(u, item, uStart, uEnd, iStart, iEnd)) {
        uRatedItems[u].push_back(item);
        ratedItems.insert(item);
      }
    }
    if (uRatedItems[u].size() == 0) {
      uNoItemsCount++;
      unRatedUsers.insert(u);
    }
  }

  for (item = iStart; item < iEnd; item++) {
    auto search = ratedItems.find(item);
    if (search == ratedItems.end()) {
      //not found 
      unRatedItems.insert(item);
    }
  }

  nSubUsers = uEnd - uStart; 
  nSubItems = iEnd - iStart;
  std::cout << "\nUsers without rating in submat: " << uNoItemsCount 
    << " " << unRatedUsers.size(); 
  std::cout << "\nItems without rating in submat: " 
    << nSubItems - ratedItems.size() << " " << unRatedItems.size();

  //std::cout << "\nNNZ = " << nnz;
  prevObj = objectiveSubMat(data, uStart, uEnd, iStart, iEnd);
  std::cout << "\nObj aftr svd: " << prevObj << " Train RMSE: " << RMSE(data.trainMat);

  std::chrono::time_point<std::chrono::system_clock> start, end;

  for (iter = 0; iter < maxIter; iter++) {  
    start = std::chrono::system_clock::now();
    
    for (subIter = 0; subIter < nnz; subIter++) {
      
      //sample u
      u = uStart + (std::rand() % nSubUsers);
      while(uRatedItems[u].size() == 0) {
        u = uStart + (std::rand() % nSubUsers);
      }

      //sample item rated by user
      nUserItems =  uRatedItems[u].size();
      itemInd = std::rand()%nUserItems; 
      item = uRatedItems[u][itemInd];
      itemRat = trainMat->rowval[trainMat->rowptr[u] + itemInd]; 
      
      //std::cout << "\nGradCheck u: " << u << " item: " << item;
      //gradCheck(u, item, itemRat);

      //compute user gradient
      computeUGrad(u, item, itemRat, uGrad);

      //update user
      //updateAdaptiveFac(uFac[u], uGrad, uGradsAcc[u]); 
      updateFac(uFac[u], uGrad); 

      //compute item gradient
      computeIGrad(u, item, itemRat, iGrad);

      //update item
      //updateAdaptiveFac(iFac[item], iGrad, iGradsAcc[item]);
      updateFac(iFac[item], iGrad);
     }

    //check objective
    if (iter % OBJ_ITER == 0 || iter == maxIter-1) {
      if (isTerminateModelSubMat(bestModel, data, iter, bestIter, bestObj, 
            prevObj, uStart, uEnd, iStart, iEnd)) {
        break; 
      }
      std::cout << "\nIter: " << iter << " Objective: " << std::scientific << prevObj 
                << " Train subMat RMSE: " 
                << subMatRMSE(data.trainMat, uStart, uEnd, iStart, iEnd) 
                << " Train subMat Non-Obs RMSE: "
                << subMatKnownRankNonObsErrWSet(data, uStart, uEnd, iStart, 
                    iEnd, unRatedUsers, unRatedItems) 
                << std::endl;
      end = std::chrono::system_clock::now();  
      std::chrono::duration<double> duration =  (end - start) ;
      std::cout << "\nSubiter duration: " << duration.count();
    }
  
  } 
   

  //std::cout << "\nNum Iter: " << iter << " Best Iter: " << bestIter
  //  << " Best obj: " << std::scientific << bestObj ;

}


//train on full but dont update passed range of users and items factors
//start inclusive end exclusive
void ModelMF::fixTrain(const Data &data, Model &bestModel, int uStart, 
    int uEnd, int iStart, int iEnd) {

  //copy passed known factors
  //uFac = data.origUFac;
  //iFac = data.origIFac;
  
  std::cout << "\nModelMF::fixTrain";
  
  int nnz = data.trainNNZ;
  int subMatNNZ = nnzSubMat(data.trainMat, uStart, uEnd, iStart, iEnd);
  nnz = nnz - subMatNNZ;
  std::cout << "\nTrain NNZ = " << nnz << " subMatNNZ: " << subMatNNZ;
 
  /*
  std::cout << "\nObj b4 svd: " << objective(data) 
    << " Train RMSE: " << RMSE(data.trainMat) 
    << " Train nnz: " << nnz << std::endl;
  std::chrono::time_point<std::chrono::system_clock> startSVD, endSVD;
  startSVD = std::chrono::system_clock::now();
  //initialization with svd of the passed matrix
  svdFrmSvdlibCSR(data.trainMat, facDim, uFac, iFac, false); 
  
  endSVD = std::chrono::system_clock::now();
  std::chrono::duration<double> durationSVD =  (endSVD - startSVD) ;
  std::cout << "\nsvd duration: " << durationSVD.count();
  */

  int u, iter, subIter, bestIter;
  int item, nUserItems, itemInd;
  float itemRat;
  double bestObj, prevObj;

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

  prevObj = objective(data);
  std::cout << "\nObj aftr svd: " << prevObj << " Train RMSE: " << RMSE(data.trainMat);

  std::chrono::time_point<std::chrono::system_clock> start, end;


  for (iter = 0; iter < maxIter; iter++) {  
    start = std::chrono::system_clock::now();
    for (subIter = 0; subIter < data.trainNNZ; subIter++) {
      
      //sample u
      u = std::rand() % nUsers;
      
      //sample item rated by user
      nUserItems =  trainMat->rowptr[u+1] - trainMat->rowptr[u];
      itemInd = std::rand()%nUserItems; 
      item = trainMat->rowind[trainMat->rowptr[u] + itemInd];
      itemRat = trainMat->rowval[trainMat->rowptr[u] + itemInd]; 
   
      if ((u >= uStart && u < uEnd) && (item >= iStart && item < iEnd)) {
        //subIter--;
        continue;
      }

      //std::cout << "\nGradCheck u: " << u << " item: " << item;
      //gradCheck(u, item, itemRat);

      if (u < uStart || u >= uEnd ) {
        //update as outside the user block
        //compute user gradient
        computeUGrad(u, item, itemRat, uGrad);

        //update user
        //updateAdaptiveFac(uFac[u], uGrad, uGradsAcc[u]); 
        updateFac(uFac[u], uGrad); 
      }

      if (item < iStart || item >= iEnd) {
        //update as outside the item block
        //compute item gradient
        computeIGrad(u, item, itemRat, iGrad);

        //update item
        //updateAdaptiveFac(iFac[item], iGrad, iGradsAcc[item]);
        updateFac(iFac[item], iGrad);
      }

    }

    //check objective
    if (iter % OBJ_ITER == 0 || iter == maxIter-1) {
      if (isTerminateModel(bestModel, data, iter, bestIter, bestObj, prevObj)) {
        break; 
      }
      std::cout << "\nIter: " << iter << " Objective: " << std::scientific << prevObj 
                << " Train RMSE: " << RMSE(data.trainMat)
                << "\n subMat Non-Obs RMSE("<<uStart<<","<<uEnd<<","<<iStart<<","<<iEnd<<"): "
                << subMatKnownRankNonObsErr(data, uStart, uEnd, iStart, iEnd) 
                << "\n subMat Non-Obs RMSE("<<uEnd<<","<<nUsers<<","<<iStart<<","<<iEnd<<"): "
                << subMatKnownRankNonObsErr(data, uEnd, nUsers, iStart, iEnd) 
                << "\n subMat Non-Obs RMSE("<<uStart<<","<<uEnd<<","<<iEnd<<","<<nItems<<"): "
                << subMatKnownRankNonObsErr(data, uStart, uEnd, iEnd, nItems)
                << "\n subMat Non-Obs RMSE("<<uEnd<<","<<nUsers<<","<<iEnd<<","<<nItems<<"): "
                << subMatKnownRankNonObsErr(data, uEnd, nUsers, iEnd, nItems)
                << std::endl;
      end = std::chrono::system_clock::now();  
      std::chrono::duration<double> duration =  (end - start) ;
      std::cout << "\nsub duration: " << duration.count();
    }
  
  }
  
  //std::cout << "\nNum Iter: " << iter << " Best Iter: " << bestIter
  //  << " Best obj: " << std::scientific << bestObj ;

}


void ModelMF::subExTrain(const Data &data, Model &bestModel,
                    int uStart, int uEnd, int iStart, int iEnd) {

  //copy passed known factors
  //uFac = data.origUFac;
  //iFac = data.origIFac;
  
  std::cout << "\nModelMF::train";
  
  /*
  std::cout << "\nObj b4 svd: " << objectiveSubMat(data) << " Train RMSE: " << RMSE(data.trainMat);
 
  std::chrono::time_point<std::chrono::system_clock> startSVD, endSVD;
  startSVD = std::chrono::system_clock::now();
  //initialization with svd of the passed matrix
  
  //svdFrmCSR(data.trainMat, facDim, uFac, iFac);
  svdFrmCSRColAvg(data.trainMat, facDim, uFac, iFac);
  
  endSVD = std::chrono::system_clock::now();
  std::chrono::duration<double> durationSVD =  (endSVD - startSVD) ;
  std::cout << "\nsvd duration: " << durationSVD.count();
  */

  int u, iter, subIter, bestIter;
  int item, nUserItems, itemInd;
  float itemRat;
  double bestObj, prevObj;
  int nnz = nnzSubMat(data.trainMat, uStart, uEnd, iStart, iEnd);

  std::cout << "\nsubmat nnz: " << nnz << std::endl;

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

  //std::cout << "\nNNZ = " << nnz;
  prevObj = objectiveSubMat(data, uStart, uEnd, iStart, iEnd);
  std::cout << "\nObj aftr svd: " << prevObj << " Train RMSE: " << RMSE(data.trainMat);

  std::chrono::time_point<std::chrono::system_clock> start, end;
  start = std::chrono::system_clock::now();

  for (iter = 0; iter < maxIter; iter++) {  
    for (subIter = 0; subIter < nnz; subIter++) {
      
      //sample u
      u = std::rand() % nUsers;
      
      //sample item rated by user
      nUserItems =  trainMat->rowptr[u+1] - trainMat->rowptr[u];
      itemInd = std::rand()%nUserItems; 
      item = trainMat->rowind[trainMat->rowptr[u] + itemInd];
     
      //continue if sampled user and item pair falls within the block
      if (isInsideBlock(u, item, uStart, uEnd, iStart, iEnd)) {
        continue;
      }
      
      itemRat = trainMat->rowval[trainMat->rowptr[u] + itemInd]; 
      //std::cout << "\nGradCheck u: " << u << " item: " << item;
      //gradCheck(u, item, itemRat);

      //compute user gradient
      computeUGrad(u, item, itemRat, uGrad);

      //update user
      //updateAdaptiveFac(uFac[u], uGrad, uGradsAcc[u]); 
      updateFac(uFac[u], uGrad); 

      //compute item gradient
      computeIGrad(u, item, itemRat, iGrad);

      //update item
      //updateAdaptiveFac(iFac[item], iGrad, iGradsAcc[item]);
      updateFac(iFac[item], iGrad);

     }

    //check objective
    if (iter % OBJ_ITER == 0 || iter == maxIter-1) {
      if (isTerminateModelExSubMat(bestModel, data, iter, bestIter, bestObj, 
            prevObj, uStart, uEnd, iStart, iEnd)) {
        break; 
      }
      std::cout << "\nIter: " << iter << " Objective: " << std::scientific << prevObj 
                << " Train ExSubMat RMSE: " << subMatExRMSE(data.trainMat, 
                    uStart, uEnd, iStart, iEnd);
    }
  
  } 
   
  end = std::chrono::system_clock::now();  

  std::chrono::duration<double> duration =  (end - start) ;
  std::cout << "\nduration: " << duration.count();
  //std::cout << "\nNum Iter: " << iter << " Best Iter: " << bestIter
  //  << " Best obj: " << std::scientific << bestObj ;

}


