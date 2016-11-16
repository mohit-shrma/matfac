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

  int u, item, iter, bestIter = -1, i; 
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
  for (int u = trainMat->nrows; u < data.nUsers; u++) {
    invalidUsers.insert(u);
  }
  for (int item = trainMat->ncols; item < data.nItems; item++) {
    invalidItems.insert(item);
  }

  //std::cout << "\nNNZ = " << nnz;
  prevObj = objective(data, invalidUsers, invalidItems);
  bestObj = prevObj;
  bestValRMSE = prevValRMSE = RMSE(data.valMat, invalidUsers, invalidItems);
  std::cout << "\nObj aftr svd: " << prevObj << " Train RMSE: " 
    << RMSE(data.trainMat, invalidUsers, invalidItems) << " Val RMSE: " 
    << bestValRMSE;

  std::chrono::time_point<std::chrono::system_clock> start, end;
  std::chrono::duration<double> duration;
  
 std::cout << "\nModelMF::train trainSeed: " << trainSeed 
    << " invalidUsers: " << invalidUsers.size()
    << " invalidItems: " << invalidItems.size() << std::endl;
  
  //random engine
  std::mt19937 mt(trainSeed);
  //get user-item ratings from training data
  const auto uiRatings = getUIRatings(trainMat, invalidUsers, invalidItems);
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
    for (const auto& ind: uiRatingInds) {
      //get user, item and rating
      u       = std::get<0>(uiRatings[ind]);
      item    = std::get<1>(uiRatings[ind]);
      itemRat = std::get<2>(uiRatings[ind]);
      
      //r_ui_est = dotProd(uFac[u], iFac[item], facDim);
      r_ui_est = 0;
      for (i = 0; i < facDim; i++) {
        r_ui_est += uFac[u][i]*iFac[item][i];
      }
      diff = itemRat - r_ui_est;
       
      //update user
      for (i = 0; i < facDim; i++) {
        uFac[u][i] -= learnRate * (-2.0*diff*iFac[item][i] + 2.0*uReg*uFac[u][i]);
      }
          
      //r_ui_est = dotProd(uFac[u], iFac[item], facDim);
      //diff = itemRat - r_ui_est;

      //update item
      for (i = 0; i < facDim; i++) {
        iFac[item][i] -= learnRate * (-2.0*diff*uFac[u][i] + 2.0*iReg*iFac[item][i]);
      }

    }
    end = std::chrono::system_clock::now();  
   
    duration =  end - start;
    subIterDuration = duration.count();

    //check objective
    if (iter % OBJ_ITER == 0 || iter == maxIter-1) {
      
      if (isTerminateModel(bestModel, data, iter, bestIter, bestObj, prevObj,
            bestValRMSE, prevValRMSE, invalidUsers, invalidItems)) {
        break; 
      }
       
      /* 
      if (isTerminateModel(bestModel, data, iter, bestIter, bestObj, prevObj,
            invalidUsers, invalidItems)) {
        break; 
      }
      */

      if (iter % DISP_ITER == 0) {
        std::cout << "ModelMF::train trainSeed: " << trainSeed
                  << " Iter: " << iter << " Objective: " << std::scientific << prevObj 
                  << " Train RMSE: " << RMSE(data.trainMat, invalidUsers, invalidItems)
                  << " Val RMSE: " << prevValRMSE
                  << " subIterDuration: " << subIterDuration
                  << std::endl;
      }

      if (iter % DISP_ITER == 0 || iter == maxIter - 1) {
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


#pragma omp parallel
{

    Eigen::MatrixXf YTY = Eigen::MatrixXf::Zero(facDim, facDim);
    Eigen::VectorXf b = Eigen::VectorXf::Zero(facDim);

    int u, user, item, uu, ii, j, k;
    float rating;
    
    //update users
#pragma omp for
    for (u = 0; u < nUsers; u++) {
      
      //skip if invalid
      if (invalidUsers.count(u) > 0) {
        continue;
      } 
      YTY.fill(0);
      b.fill(0);

      
      for (ii = trainMat->rowptr[u]; ii < trainMat->rowptr[u+1]; ii++) {
        item = trainMat->rowind[ii];
        rating = trainMat->rowval[ii];
        if (rating > 0) {
          //update YTY
          for (j = 0; j < facDim; j++) {
            for (k = 0; k < facDim; k++) {
              YTY(j, k) += iFac[item][j]*iFac[item][k];
            }
            b(j) += rating*iFac[item][j];
          }
        }
      }
      
      //add u reg
      for (j = 0; j < facDim; j++) {
        YTY(j,j) += uReg;
      }
      
      //solve YTY * u = b
      Eigen::VectorXf ufac =  YTY.ldlt().solve(b); 
      //Eigen::VectorXf ufac =  YTY.lu().solve(b); 
      for (j = 0; j < facDim; j++) {
        uFac[u][j] = ufac[j];
      } 
    } 

    //update items
#pragma omp for
    for (item = 0; item < nItems; item++) {
      //skip if invalid
      if (invalidItems.count(item) > 0) {
        continue;
      }
     
      YTY.fill(0);
      b.fill(0);
      
      for (uu = trainMat->colptr[item]; uu < trainMat->colptr[item+1]; uu++) {
        user = trainMat->colind[uu];
        rating = trainMat->colval[uu];
        if (rating > 0) {
          //update YTY
          for (j = 0; j < facDim; j++) {
            for (k = 0; k < facDim; k++) {
              YTY(j, k) += uFac[user][j]*uFac[user][k];
            }
            b(j) += rating*uFac[user][j];
          }
        }
      }
      
      //add item reg
      for (j = 0; j < facDim; j++) {
        YTY(j, j) += iReg;
      }
      
      //solve YTY * v = b
      Eigen::VectorXf ifac =  YTY.ldlt().solve(b); 
      //Eigen::VectorXf ifac =  YTY.lu().solve(b); 
      for (j = 0; j < facDim; j++) {
        iFac[item][j] = ifac[j];
      } 

    } 

}
    end = std::chrono::system_clock::now();  
   
    duration =  end - start;
    subIterDuration = duration.count();

    //check objective
    if (iter % OBJ_ITER == 0 || iter == maxIter-1) {
      
      if (isTerminateModel(bestModel, data, iter, bestIter, bestObj, prevObj,
            bestValRMSE, prevValRMSE, invalidUsers, invalidItems)) {
        break; 
      }
      
      /*
      if (isTerminateModel(bestModel, data, iter, bestIter, bestObj, prevObj,
            invalidUsers, invalidItems)) {
        break; 
      }
      */
      
      if (iter % DISP_ITER == 0) {
        std::cout << "ModelMF::trainALS trainSeed: " << trainSeed
                  << " Iter: " << iter << " Objective: " << std::scientific << prevObj 
                  << " Train RMSE: " << RMSE(data.trainMat, invalidUsers, invalidItems)
                  << " Val RMSE: " << prevValRMSE
                  << " subIterDuration: " << subIterDuration
                  << std::endl;
      }

      if (iter % DISP_ITER == 0 || iter == maxIter - 1) {
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


void ModelMF::trainCCDPP(const Data &data, Model &bestModel, 
    std::unordered_set<int>& invalidUsers,
    std::unordered_set<int>& invalidItems) {

  //copy passed known factors
  //uFac = data.origUFac;
  //iFac = data.origIFac;
  
  std::cout << "\nModelMF::trainCCDPP trainSeed: " << trainSeed;
  
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
  
 std::cout << "\nModelMF::trainCCDPP trainSeed: " << trainSeed 
    << " invalidUsers: " << invalidUsers.size()
    << " invalidItems: " << invalidItems.size() << std::endl;
  
  //random engine
  std::mt19937 mt(trainSeed);
  //get user-item ratings from training data
  auto uiRatings = getUIRatings(trainMat, invalidUsers, invalidItems);
  //index to above uiRatings pair
  std::vector<size_t> uiRatingInds(uiRatings.size());
  std::iota(uiRatingInds.begin(), uiRatingInds.end(), 0);

  //residual mat
  gk_csr_t *res = gk_csr_Dup(trainMat);

  std::cout << "\nTrain NNZ after removing invalid users and items: " 
    << uiRatings.size() << std::endl;
  double subIterDuration = 0;
  
  //fill uFac as 0
#pragma omp parallel for
  for (int u = 0; u < nUsers; u++) {
    for (int k = 0; k < facDim; k++) {
      uFac[u][k] = 0;
    }
  }


  for (iter = 0; iter < maxIter; iter++) { 

    start = std::chrono::system_clock::now();
   
    double rankFunDec = 0, funDecMax = 0;

    for (int t = 0; t < facDim; t++) {
      const int k = t;
      std::vector<double> u_k(nUsers, 0), v_k(nItems, 0);  
      
      for (int u = 0; u < nUsers; u++) {
        u_k[u] = uFac[u][k];
      }

      for (int item = 0; item < nItems; item++) {
        v_k[item] = iFac[item][k];
      }
      
      double innerFunDecCur = 0, innerFunDecMax = 0;
      
      for (int subIter = 0; subIter < 5; subIter++) {
        
        innerFunDecCur = 0;      

        //update u
#pragma omp parallel for reduction(+: innerFunDecCur)
        for (int u = 0; u < nUsers; u++) {
          if (invalidUsers.count(u) > 0) {
            continue;
          }
          double num = 0, denom = uReg, newV;
          for (int ii = res->rowptr[u]; ii < res->rowptr[u+1]; ii++) {
            int item = res->rowind[ii];
            num += res->rowval[ii] * v_k[item];
            denom += v_k[item]*v_k[item];
          }
          newV = num/denom;
          innerFunDecCur += denom*(u_k[u] - newV)*(u_k[u] - newV);
          u_k[u] = newV;
        }

        //update v
#pragma omp parallel for reduction(+: innerFunDecCur)
        for (int item = 0; item < nItems; item++) {
          if (invalidItems.count(item) > 0 || item >= res->ncols) {
            continue;
          }
          double num = 0, denom = iReg, newV;
          for (int uu = res->colptr[item]; uu < res->colptr[item+1]; uu++) {
            int u = res->colind[uu];
            num += res->colval[uu] * u_k[u];
            denom += u_k[u]*u_k[u];
          }
          newV = num/denom;
          innerFunDecCur += denom*(v_k[item] - newV)*(v_k[item] - newV);
          v_k[item] = newV;
        }

        if (innerFunDecCur < funDecMax*EPS) {
          break;
        }

        rankFunDec += innerFunDecCur;
        if (innerFunDecCur > innerFunDecMax) {
          innerFunDecMax = innerFunDecCur;
        }

        // the fundec of the first inner iter of the first rank of the first
        // outer iteration could be too large!!
        if (!(iter == 0 && k == 0 && subIter == 0)) {
          if (innerFunDecCur > funDecMax) {
            funDecMax = innerFunDecCur;
          }
        }

      }
        //update residual res
#pragma omp parallel for
        for (int u = 0; u < nUsers; u++) {
          for (int ii = res->rowptr[u]; ii < res->rowptr[u+1]; ii++) {
            int item = res->rowind[ii];
            res->rowval[ii] += uFac[u][k]*iFac[item][k] - u_k[u]*v_k[item]; 
          }
        }
        
        //update residual res^T
#pragma omp parallel for
        for (int item = 0; item < nItems; item++) {
          if (item >= res->ncols) {continue;}
          for (int uu = res->colptr[item]; uu < res->colptr[item+1]; uu++) {
            int u = res->colind[uu];
            res->colval[uu] += uFac[u][k]*iFac[item][k] - u_k[u]*v_k[item]; 
          }
        } 
      
       //update factors
#pragma omp parallel for 
        for (int u = 0; u < nUsers; u++) {
          uFac[u][k] = u_k[u];
        }

#pragma omp parallel for
        for (int item = 0; item < nItems; item++) {
          iFac[item][k] = v_k[item];
        }

    }


    end = std::chrono::system_clock::now();  
   
    duration =  end - start;
    subIterDuration = duration.count();

    //check objective
    if (iter % OBJ_ITER == 0 || iter == maxIter-1) {
      
      if (isTerminateModel(bestModel, data, iter, bestIter, bestObj, prevObj,
            bestValRMSE, prevValRMSE, invalidUsers, invalidItems)) {
        break; 
      }
      
     
      /*
      if (isTerminateModel(bestModel, data, iter, bestIter, bestObj, prevObj,
            invalidUsers, invalidItems)) {
        break; 
      }
      */
      if (iter % DISP_ITER == 0) {
        std::cout << "ModelMF::trainCCDPP trainSeed: " << trainSeed
                  << " Iter: " << iter << " Objective: " << std::scientific << prevObj 
                  << " Train RMSE: " << RMSE(data.trainMat, invalidUsers, invalidItems)
                  << " Val RMSE: " << prevValRMSE
                  << " subIterDuration: " << subIterDuration
                  << std::endl;
      }

      if (iter % DISP_ITER == 0 || iter == maxIter - 1) {
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
  
  gk_csr_Free(&res);
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
  bestValRMSE = prevValRMSE = RMSE(data.valMat, invalidUsers, invalidItems);
  std::cout << "\nObj aftr svd: " << prevObj << " Train RMSE: " 
    << RMSE(data.trainMat, invalidUsers, invalidItems) << " Val RMSE: " 
    << bestValRMSE;

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
  const int indsSz = uiRatingInds.size();
  
  for (iter = 0; iter < maxIter; iter++) {  
    
    //shuffle the user item rating indexes
    std::shuffle(uiRatingInds.begin(), uiRatingInds.end(), mt);

    start = std::chrono::system_clock::now();
#pragma omp parallel for
    for (int k = 0; k < indsSz; k++) {
      //auto ind = uiRatingInds[k];
      //get user, item and rating
      const int u       = std::get<0>(uiRatings[uiRatingInds[k]]);
      const int item    = std::get<1>(uiRatings[uiRatingInds[k]]);
      const float itemRat = std::get<2>(uiRatings[uiRatingInds[k]]);
      
      //double r_ui_est = dotProd(uFac[u], iFac[item], facDim);
      double r_ui_est = 0;
      for (int i = 0; i < facDim; i++) {
        r_ui_est += uFac[u][i]*iFac[item][i];
      }
      const double diff = itemRat - r_ui_est;

      //update user
      for (int i = 0; i < facDim; i++) {
        uFac[u][i] -= learnRate*(-2.0*diff*iFac[item][i] + 2.0*uReg*uFac[u][i]);
      }

      //r_ui_est = dotProd(uFac[u], iFac[item], facDim);
      //diff = itemRat - r_ui_est;
    
      //update item
      for (int i = 0; i < facDim; i++) {
        iFac[item][i] -= learnRate*(-2.0*diff*uFac[u][i] + 2.0*iReg*iFac[item][i]);
      }
    }
    end = std::chrono::system_clock::now();  
   
    duration =  end - start;
    subIterDuration = duration.count();

    //check objective
    if (iter % OBJ_ITER == 0 || iter == maxIter-1) {
      
      if (isTerminateModel(bestModel, data, iter, bestIter, bestObj, prevObj,
            bestValRMSE, prevValRMSE, invalidUsers, invalidItems)) {
        break; 
      }
      
      /*
      if (isTerminateModel(bestModel, data, iter, bestIter, bestObj, prevObj,
            invalidUsers, invalidItems)) {
        break; 
      }
      */

      if (iter % DISP_ITER == 0) {
        std::cout << "ModelMF::hogTrain trainSeed: " << trainSeed
                  << " Iter: " << iter << " Objective: " << std::scientific << prevObj 
                  << " Train RMSE: " << RMSE(data.trainMat, invalidUsers, invalidItems)
                  << " Val RMSE: " << prevValRMSE
                  << " sub duration: " << subIterDuration
                  << std::endl;
      }

      if (iter % DISP_ITER == 0 || iter == maxIter - 1) {
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


void ModelMF::hogAdapTrain(const Data &data, Model &bestModel, 
    std::unordered_set<int>& invalidUsers,
    std::unordered_set<int>& invalidItems) {

  //copy passed known factors
  //uFac = data.origUFac;
  //iFac = data.origIFac;
  
  std::cout << "\nModelMF::hogAdapTrain trainSeed: " << trainSeed;
  
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

  int iter, bestIter; 
  double bestObj, prevObj;
  double bestValRMSE, prevValRMSE;

  gk_csr_t *trainMat = data.trainMat;

 
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
  std::cout << "\nrhoRMS: " << rhoRMS << std::endl;
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
      
      double r_ui_est = dotProd(uFac[u], iFac[item], facDim);
      double diff = itemRat - r_ui_est;
      
      double gradi = 0;
      //update user
      for (int i = 0; i < facDim; i++) {
        gradi = -2.0*diff*iFac[item][i] + 2.0*uReg*uFac[u][i]; 
        uGradsAcc[u][i] = uGradsAcc[u][i]*rhoRMS + (1.0 - rhoRMS)*gradi*gradi;
        uFac[u][i] -= (learnRate/sqrt(uGradsAcc[u][i] + 1e-6))*gradi;
      }

      r_ui_est = dotProd(uFac[u], iFac[item], facDim);
      diff = itemRat - r_ui_est;
    
      //update item
      for (int i = 0; i < facDim; i++) {
        gradi = -2.0*diff*uFac[u][i] + 2.0*iReg*iFac[item][i];
        iGradsAcc[item][i] = iGradsAcc[item][i]*rhoRMS + (1.0 - rhoRMS)*gradi*gradi;
        iFac[item][i] -= (learnRate/sqrt(iGradsAcc[item][i] + 1e-6))*gradi;
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


