#include "model.h"


void Model::updateFac(std::vector<double> &fac, std::vector<double> &grad) {
  for (int i = 0; i < facDim; i++) {
    fac[i] -= learnRate * grad[i];
  }
}


std::string Model::modelSignature() {
  
  std::string sign = std::to_string(nUsers) + "X" + std::to_string(nItems)
    + "_" + std::to_string(facDim) 
    + "_" + std::to_string(uReg) + "_" + std::to_string(iReg)
    + "_" + std::to_string(learnRate);
  
  return sign;
}


void Model::display() {
  std::cout << "nUsers: " << nUsers << " nItems: " << nItems << std::endl;
  std::cout << "facDim: " << facDim << std::endl;
  std::cout << "uReg: " << uReg << " iReg: " << iReg << std::endl;
  std::cout << "learnRate: " << learnRate << std::endl;
  std::cout << "trainSeed: " << trainSeed;
} 


void Model::save(std::string prefix) {

  std::string modelSign = modelSignature();

  //save user latent factors
  std::string uFacName = prefix + "_uFac_" + modelSign + ".mat";
  writeMat(uFac, nUsers, facDim, uFacName.c_str());
  
  //save item latent factors
  std::string iFacName = prefix + "_iFac_" + modelSign +  ".mat";
  writeMat(iFac, nItems, facDim, iFacName.c_str());

  //save user bias
  std::string uBFName = prefix + "_uBias_" + modelSign + ".vec";
  writeVector(uBias, uBFName.c_str());

  //save item bias
  std::string iBFName = prefix + "_iBias_" + modelSign + ".vec";
  writeVector(iBias, iBFName.c_str());

  //save global bias 
  std::vector<double> gBias = {mu};
  std::string gBFName = prefix + "_" + modelSign + "_gBias";
  writeVector(gBias, gBFName.c_str());
}


void Model::load(std::string prefix) {
  std::string modelSign = modelSignature();
  //read user latent factors
  std::string uFacName = prefix + "_uFac_" + modelSign + ".mat";
  readMat(uFac, nUsers, facDim, uFacName.c_str());
  
  //read item latent factors
  std::string iFacName = prefix + "_iFac_" + modelSign +  ".mat";
  readMat(iFac, nItems, facDim, iFacName.c_str());

  //read user bias
  std::string uBFName = prefix + "_uBias_" + modelSign + ".vec";
  uBias = readDVector(uBFName.c_str());

  //read item bias
  std::string iBFName = prefix + "_iBias_" + modelSign + ".vec";
  iBias = readDVector(iBFName.c_str());

  //read global bias 
  std::vector<double> gBias;
  std::string gBFName = prefix + "_" + modelSign + "_gBias";
  gBias = readDVector(gBFName.c_str());
  mu = gBias[0];  
}


void Model::saveFacs(std::string prefix) {
  std::string modelSign = modelSignature();
  //save user latent factors
  std::string uFacName = prefix + "_uFac_" + modelSign + ".mat";
  writeMat(uFac, nUsers, facDim, uFacName.c_str());
  
  //save item latent factors
  std::string iFacName = prefix + "_iFac_" + modelSign +  ".mat";
  writeMat(iFac, nItems, facDim, iFacName.c_str());
}


void Model::loadFacs(std::string prefix) {
  std::string modelSign = modelSignature();
  //read user latent factors
  std::string uFacName = prefix + "_uFac_" + modelSign + ".mat";
  readMat(uFac, nUsers, facDim, uFacName.c_str());
  
  //read item latent factors
  std::string iFacName = prefix + "_iFac_" + modelSign +  ".mat";
  readMat(iFac, nItems, facDim, iFacName.c_str());
}


void Model::load(const char* uFacName, const char* iFacName, const char* uBFName,
    const char* iBFName, const char*gBFName) {
  //read user latent factors
  readMat(uFac, nUsers, facDim, uFacName);
  
  //read item latent factors
  readMat(iFac, nItems, facDim, iFacName);

  //read user bias
  uBias = readDVector(uBFName);

  //read item bias
  iBias = readDVector(iBFName);

  //read global bias 
  std::vector<double> gBias;
  gBias = readDVector(gBFName);
  mu = gBias[0];  
}


void Model::load(const char* uFacName, const char *iFacName) {
  std::cout << "\nLoading user factors: " << uFacName;
  readMat(uFac, nUsers, facDim, uFacName);
  std::cout << "\nLoading item factors: " << iFacName;
  readMat(iFac, nItems, facDim, iFacName);
}


double Model::RMSE(gk_csr_t *mat) {
  int u, i, ii, nnz;
  float r_ui;
  double r_ui_est, diff, rmse;

  nnz = 0;
  rmse = 0;
  for (u = 0; u < nUsers; u++) {
    for (ii = mat->rowptr[u]; ii < mat->rowptr[u+1]; ii++) {
      i = mat->rowind[ii];
      r_ui = mat->rowval[ii];
      r_ui_est = estRating(u, i);
      diff = r_ui - r_ui_est;
      rmse += diff*diff;
      nnz++;
    }
  }
  rmse = sqrt(rmse/nnz);
  
  return rmse;
}


double Model::RMSE(gk_csr_t *mat, std::unordered_set<int>& invalidUsers,
    std::unordered_set<int>& invalidItems) {
  int u, i, ii, nnz;
  float r_ui;
  double r_ui_est, diff, rmse;

  nnz = 0;
  rmse = 0;
  for (u = 0; u < nUsers; u++) {
    //skip if invalid user
    auto search = invalidUsers.find(u);
    if (search != invalidUsers.end()) {
      //found and skip
      continue;
    }
    for (ii = mat->rowptr[u]; ii < mat->rowptr[u+1]; ii++) {
      i = mat->rowind[ii];
      //skip if invalid item
      search = invalidItems.find(i);
      if (search != invalidItems.end() || i >= nItems) {
        //found and skip
        continue;
      }
      
      r_ui = mat->rowval[ii];
      r_ui_est = estRating(u, i);
      diff = r_ui - r_ui_est;
      rmse += diff*diff;
      nnz++;
    }
  }
  rmse = sqrt(rmse/nnz);
  
  return rmse;
}


double Model::RMSE(gk_csr_t *mat, std::unordered_set<int>& invalidUsers,
    std::unordered_set<int>& invalidItems, Model& origModel) {
  int u, ii, item, nnz;
  float r_ui;
  double r_ui_est, diff, rmse;

  nnz = 0;
  rmse = 0;
  for (u = 0; u < nUsers; u++) { 
    //skip if invalid user
    auto search = invalidUsers.find(u);
    if (search != invalidUsers.end()) {
      //found and skip
      continue;
    }
    for (ii = mat->rowptr[u]; ii <  mat->rowptr[u+1]; ii++) {
      item = mat->rowind[ii];
      //skip if invalid item
      search = invalidItems.find(item);
      if (search != invalidItems.end() || item >= nItems) {
        //found and skip
        continue;
      }
      r_ui = origModel.estRating(u, item);
      r_ui_est = estRating(u, item);
      diff = r_ui - r_ui_est;
      rmse += diff*diff;
      nnz++;
    }
  }
  rmse = sqrt(rmse/nnz);
  
  return rmse;
}


double Model::RMSE(std::vector<std::tuple<int, int, float>>& trainRatings) {
  int u, item, nnz;
  float itemRat;
  double diff, rmse;

  rmse = 0;
  nnz = trainRatings.size();

  for (auto&& uiRating: trainRatings) {
    u = std::get<0>(uiRating);
    item = std::get<1>(uiRating);
    itemRat = std::get<2>(uiRating);
    diff = itemRat - estRating(u, item);
    rmse += diff*diff;
  }
  
  rmse = sqrt(rmse/nnz);
  
  return rmse;
}


double Model::estRating(int user, int item) {
  return dotProd(uFac[user], iFac[item], facDim);
}


double Model::estAvgRating(int user, std::unordered_set<int>& invalidItems) {
  double avgRat = 0;
  int itemCount = 0;
  for (int item = 0; item < nItems; item++) {
    if (invalidItems.find(item) != invalidItems.end()) {
      //invalid
      continue;
    }
    avgRat += estRating(user, item);
    itemCount++;
  }
  return avgRat/itemCount;
}


//compute RMSE with in the submatrix
double Model::subMatRMSE(gk_csr_t *mat, int uStart, int uEnd, 
    int iStart, int iEnd) {
  double r_ui_est, diff, rmse, r_ui;
  int u, ii, item, nnz;
  
  rmse = 0;
  nnz = 0;
  
  for (u = uStart; u < uEnd; u++) {
    for (ii = mat->rowptr[u]; ii < mat->rowptr[u+1]; ii++) {
      item = mat->rowind[ii];
      if (!isInsideBlock(u, item, uStart, uEnd, iStart, iEnd)) {
        continue;
      }
      r_ui = mat->rowval[ii];
      r_ui_est = estRating(u, item);
      diff = r_ui - r_ui_est;
      rmse += diff*diff;
      nnz++;
    }
  }

  rmse = sqrt(rmse/nnz);
  return rmse;
}


//compute rmse outside the submatrix
double Model::subMatExRMSE(gk_csr_t *mat, int uStart, int uEnd, 
    int iStart, int iEnd) {
  double r_ui_est, diff, rmse, r_ui;
  int u, ii, item, nnz;
  
  rmse = 0;
  nnz = 0;
  
  for (u = 0; u < mat->nrows; u++) {
    for (ii = mat->rowptr[u]; ii < mat->rowptr[u+1]; ii++) {
      item = mat->rowind[ii];
      //if inside block then continue
      if (isInsideBlock(u, item, uStart, uEnd, iStart, iEnd)) {
        continue;
      }
      r_ui = mat->rowval[ii];
      r_ui_est = estRating(u, item);
      diff = r_ui - r_ui_est;
      rmse += diff*diff;
      nnz++;
    }
  }

  rmse = sqrt(rmse/nnz);
  return rmse;
}


double Model::fullRMSE(const Data& data) {
  int u, i, ii, nnz;
  float r_ui;
  double r_ui_est, diff, rmse;
  gk_csr_t *mat = data.trainMat;

  nnz = 0;
  rmse = 0;
  
  for (u = 0; u < nUsers; u++) {
    for (ii = mat->rowptr[u]; ii < mat->rowptr[u+1]; ii++) {
      i = mat->rowind[ii];
      r_ui = mat->rowval[ii];
      r_ui_est = estRating(u, i);
      diff = r_ui - r_ui_est;
      rmse += diff*diff;
      nnz++;
    }
  }

  mat = data.testMat;
  for (u = 0; u < nUsers; u++) {
    for (ii = mat->rowptr[u]; ii < mat->rowptr[u+1]; ii++) {
      i = mat->rowind[ii];
      r_ui = mat->rowval[ii];
      r_ui_est = estRating(u, i);
      diff = r_ui - r_ui_est;
      rmse += diff*diff;
      nnz++;
    }
  }

  rmse = sqrt(rmse/nnz);
  
  return rmse;
}


bool Model::isTerminateModel(Model& bestModel, const Data& data, int iter,
    int& bestIter, double& bestObj, double& prevObj) {
  bool ret = false;
  double currObj = objective(data);
  if (iter > 0) {
    
    if (currObj < bestObj) {
      bestModel = *this;
      bestObj = currObj;
      bestIter = iter;
    }

    if (iter - bestIter >= 50) {
      //can't go lower than best objective after 500 iterations
      printf("\nNOT CONVERGED: bestIter:%d bestObj: %.10e"
          " currIter:%d currObj: %.10e", bestIter, bestObj, iter, currObj);
      ret = true;
    }
    
    if (fabs(prevObj - currObj) < EPS) {
      //convergence
      printf("\nConverged in iteration: %d prevObj: %.10e currObj: %.10e", iter,
              prevObj, currObj); 
      ret = true;
    }
  }

  if (iter == 0) {
    bestObj = currObj;
    bestIter = iter;
  }

  prevObj = currObj;

  return ret;
}


bool Model::isTerminateModel(Model& bestModel, const Data& data, int iter,
    int& bestIter, double& bestObj, double& prevObj, 
    std::unordered_set<int>& invalidUsers, std::unordered_set<int>& invalidItems) {
  bool ret = false;
  double currObj = objective(data, invalidUsers, invalidItems);

  if (iter > 0) {
    
    if (currObj < bestObj) {
      bestModel = *this;
      bestObj = currObj;
      bestIter = iter;
    }

    if (iter - bestIter >= 100) {
      //half the learning rate
      if (learnRate > 1e-5) {
        learnRate = learnRate/2;
      }
    }

    if (iter - bestIter >= 500) {
      //can't go lower than best objective after 500 iterations
      printf("\nNOT CONVERGED: bestIter:%d bestObj: %.10e"
          " currIter:%d currObj: %.10e", bestIter, bestObj, iter, currObj);
      ret = true;
    }
    
    if (fabs(prevObj - currObj) < EPS) {
      //convergence
      printf("\nConverged in iteration: %d prevObj: %.10e currObj: %.10e", iter,
              prevObj, currObj); 
      ret = true;
    }
  }

  if (iter == 0) {
    bestObj = currObj;
    bestIter = iter;
  }

  prevObj = currObj;

  return ret;
}


bool Model::isTerminateModel(Model& bestModel, const Data& data, int iter,
    int& bestIter, double& bestObj, double& prevObj, 
    std::unordered_set<int>& invalidUsers, std::unordered_set<int>& invalidItems,
    std::vector<std::tuple<int, int, float>>& trainRatings) {
  bool ret = false;
  double currObj = objective(data, invalidUsers, invalidItems, trainRatings);

  if (iter > 0) {
    
    if (currObj < bestObj) {
      bestModel = *this;
      bestObj = currObj;
      bestIter = iter;
    }

    if (iter - bestIter >= 100) {
      //half the learning rate
      if (learnRate > 1e-5) {
        learnRate = learnRate/2;
      } else if (learnRate < 1e-5) {
        learnRate = 1e-5;
      }
    }

    if (iter - bestIter >= 500) {
      //can't go lower than best objective after 500 iterations
      printf("\nNOT CONVERGED: bestIter:%d bestObj: %.10e"
          " currIter:%d currObj: %.10e", bestIter, bestObj, iter, currObj);
      ret = true;
    }
    
    if (fabs(prevObj - currObj) < EPS) {
      //convergence
      printf("\nConverged in iteration: %d prevObj: %.10e currObj: %.10e", iter,
              prevObj, currObj); 
      ret = true;
    }
  }

  if (iter == 0) {
    bestObj = currObj;
    bestIter = iter;
  }

  prevObj = currObj;

  return ret;
}


bool Model::isTerminateModel(Model& bestModel, const Data& data, int iter,
    int& bestIter, double& bestObj, double& prevObj, double& bestValRMSE,
    double& prevValRMSE, std::unordered_set<int>& invalidUsers, 
    std::unordered_set<int>& invalidItems) {
  bool ret = false;
  double currObj = objective(data, invalidUsers, invalidItems);
  double currValRMSE = -1;
  
  if (data.valMat) {
    currValRMSE = RMSE(data.valMat, invalidUsers, invalidItems);
  } else {
    std::cerr << "\nNo validation data" << std::endl;
    exit(0);
  }

  if (iter > 0) {
    
    if (currValRMSE < bestValRMSE) {
      bestModel = *this;
      bestValRMSE = currValRMSE;
      bestIter = iter;
    }

    if (iter - bestIter >= 50) {
      //can't improve validation RMSE after 500 iterations
      printf("\nNOT CONVERGED: bestIter:%d bestObj: %.10e bestValRMSE: %.10e"
          " currIter:%d currObj: %.10e currValRMSE: %.10e", 
          bestIter, bestObj, bestValRMSE, iter, currObj, currValRMSE);
      ret = true;
    }
    
    if (fabs(prevObj - currObj) < EPS) {
      //convergence
      printf("\nConverged in iteration: %d prevObj: %.10e currObj: %.10e"
          " bestValRMSE: %.10e", iter, prevObj, currObj, bestValRMSE); 
      ret = true;
    }

    if (fabs(prevValRMSE - currValRMSE) < 0.0001) {
      printf("\nvalidation RMSE in iteration: %d prev: %.10e curr: %.10e" 
          " bestValRMSE: %.10e", iter, prevValRMSE, currValRMSE, bestValRMSE); 
      ret = true;
    }

  }

  if (iter == 0) {
    bestObj = currObj;
    bestValRMSE = currValRMSE;
    bestIter = iter;
  }

  prevObj = currObj;
  prevValRMSE = currValRMSE;

  return ret;
}


bool Model::isTerminateModelSubMat(Model& bestModel, const Data& data, int iter,
    int& bestIter, double& bestObj, double& prevObj, int uStart, int uEnd,
    int iStart, int iEnd) {
  bool ret = false;
  double currObj = objectiveSubMat(data, uStart, uEnd, iStart, iEnd);
  if (iter > 0) {
    
    if (currObj < bestObj) {
      bestModel = *this;
      bestObj = currObj;
      bestIter = iter;
    }

    if (iter - bestIter >= 500) {
      //can't go lower than best objective after 500 iterations
      printf("\nNOT CONVERGED: bestIter:%d bestObj: %.10e"
          " currIter:%d currObj: %.10e", bestIter, bestObj, iter, currObj);
      ret = true;
    }
    
    if (fabs(prevObj - currObj) < EPS) {
      //convergence
      printf("\nConverged in iteration: %d prevObj: %.10e currObj: %.10e", iter,
              prevObj, currObj); 
      ret = true;
    }
  }

  if (iter == 0) {
    bestObj = currObj;
    bestIter = iter;
  }

  prevObj = currObj;

  return ret;
}


bool Model::isTerminateModelExSubMat(Model& bestModel, const Data& data, int iter,
    int& bestIter, double& bestObj, double& prevObj, int uStart, int uEnd,
    int iStart, int iEnd) {
  bool ret = false;
  double currObj = objectiveExSubMat(data, uStart, uEnd, iStart, iEnd);
  if (iter > 0) {
    
    if (currObj < bestObj) {
      bestModel = *this;
      bestObj = currObj;
      bestIter = iter;
    }

    if (iter - bestIter >= 500) {
      //can't go lower than best objective after 500 iterations
      printf("\nNOT CONVERGED: bestIter:%d bestObj: %.10e"
          " currIter:%d currObj: %.10e", bestIter, bestObj, iter, currObj);
      ret = true;
    }
    
    if (fabs(prevObj - currObj) < EPS) {
      //convergence
      printf("\nConverged in iteration: %d prevObj: %.10e currObj: %.10e", iter,
              prevObj, currObj); 
      ret = true;
    }
  }

  if (iter == 0) {
    bestObj = currObj;
    bestIter = iter;
  }

  prevObj = currObj;

  return ret;
}


//compute objective on train mat for basic mf model
double Model::objective(const Data& data) {

  int u, ii, item;
  float itemRat;
  double rmse = 0, uRegErr = 0, iRegErr = 0, obj = 0, diff = 0;
  gk_csr_t *trainMat = data.trainMat;

  for (u = 0; u < nUsers; u++) {
    for (ii = trainMat->rowptr[u]; ii < trainMat->rowptr[u+1]; ii++) {
      item = trainMat->rowind[ii];
      itemRat = trainMat->rowval[ii];
      diff = itemRat - estRating(u, item);
      rmse += diff*diff;
    }
    uRegErr += dotProd(uFac[u], uFac[u], facDim);
  }
  uRegErr = uRegErr*uReg;
  
  for (item = 0; item < nItems; item++) {
    iRegErr += dotProd(iFac[item], iFac[item], facDim);
  }
  iRegErr = iRegErr*iReg;

  obj = rmse + uRegErr + iRegErr;
    
  //std::cout <<"\nrmse: " << std::scientific << rmse << " uReg: " << uRegErr << " iReg: " << iRegErr ; 

  return obj;
}


double Model::objective(const Data& data, std::unordered_set<int>& invalidUsers,
    std::unordered_set<int>& invalidItems, 
    std::vector<std::tuple<int, int, float>>& trainRatings) {

  int u, item;
  float itemRat;
  double rmse = 0, uRegErr = 0, iRegErr = 0, obj = 0, diff = 0;

  for (auto&& uiRating: trainRatings) {
    u = std::get<0>(uiRating);
    item = std::get<1>(uiRating);
    itemRat = std::get<2>(uiRating);
    diff = itemRat - estRating(u, item);
    rmse += diff*diff;
  }
  for (u = 0; u < nUsers; u++) {
    //skip if invalid user
    auto search = invalidUsers.find(u);
    if (search != invalidUsers.end()) {
      //found and skip
      continue;
    }
    uRegErr += dotProd(uFac[u], uFac[u], facDim);
  }
  uRegErr = uRegErr*uReg;

  for (item = 0; item < nItems; item++) {
    //skip if invalid item
    auto search = invalidItems.find(item);
    if (search != invalidItems.end()) {
      //found and skip
      continue;
    }
    iRegErr += dotProd(iFac[item], iFac[item], facDim);
  }
  iRegErr = iRegErr*iReg;

  obj = rmse + uRegErr + iRegErr;
    
  //std::cout <<"\nrmse: " << std::scientific << rmse << " uReg: " << uRegErr << " iReg: " << iRegErr ; 

  return obj;
}


double Model::objective(const Data& data, std::unordered_set<int>& invalidUsers,
    std::unordered_set<int>& invalidItems) {

  int u, ii, item;
  float itemRat;
  double rmse = 0, uRegErr = 0, iRegErr = 0, obj = 0, diff = 0;
  gk_csr_t *trainMat = data.trainMat;

  for (u = 0; u < nUsers; u++) {
    //skip if invalid user
    auto search = invalidUsers.find(u);
    if (search != invalidUsers.end()) {
      //found and skip
      continue;
    }
    for (ii = trainMat->rowptr[u]; ii < trainMat->rowptr[u+1]; ii++) {
      item = trainMat->rowind[ii];
      //skip if invalid item
      search = invalidItems.find(item);
      if (search != invalidItems.end()) {
        //found and skip
        continue;
      }

      itemRat = trainMat->rowval[ii];
      diff = itemRat - estRating(u, item);
      rmse += diff*diff;
    }
    uRegErr += dotProd(uFac[u], uFac[u], facDim);
  }
  uRegErr = uRegErr*uReg;
  
  for (item = 0; item < nItems; item++) {
    //skip if invalid item
    auto search = invalidItems.find(item);
    if (search != invalidItems.end()) {
      //found and skip
      continue;
    }
    iRegErr += dotProd(iFac[item], iFac[item], facDim);
  }
  iRegErr = iRegErr*iReg;

  obj = rmse + uRegErr + iRegErr;
    
  //std::cout <<"\nrmse: " << std::scientific << rmse << " uReg: " << uRegErr << " iReg: " << iRegErr ; 

  return obj;
}


double Model::objectiveSubMat(const Data& data, int uStart, int uEnd,
    int iStart, int iEnd) {

  int u, ii, item;
  float itemRat;
  double rmse = 0, uRegErr = 0, iRegErr = 0, obj = 0, diff = 0;
  gk_csr_t *trainMat = data.trainMat;

  for (u = uStart; u < uEnd; u++) {
    for (ii = trainMat->rowptr[u]; ii < trainMat->rowptr[u+1]; ii++) {
      item = trainMat->rowind[ii];
      if (!isInsideBlock(u, item, uStart, uEnd, iStart, iEnd)) {
        continue;
      }
      itemRat = trainMat->rowval[ii];
      diff = itemRat - estRating(u, item);
      rmse += diff*diff;
    }
    uRegErr += dotProd(uFac[u], uFac[u], facDim);
  }
  uRegErr = uRegErr*uReg;
  
  for (item = iStart; item < iEnd; item++) {
    iRegErr += dotProd(iFac[item], iFac[item], facDim);
  }
  iRegErr = iRegErr*iReg;

  obj = rmse + uRegErr + iRegErr;
    
  //std::cout <<"\nrmse: " << std::scientific << rmse << " uReg: " << uRegErr << " iReg: " << iRegErr ; 

  return obj;
}


double Model::objectiveExSubMat(const Data& data, int uStart, int uEnd,
    int iStart, int iEnd) {

  int u, ii, item;
  float itemRat;
  double rmse = 0, uRegErr = 0, iRegErr = 0, obj = 0, diff = 0;
  gk_csr_t *trainMat = data.trainMat;

  for (u = 0; u < nUsers; u++) {
    for (ii = trainMat->rowptr[u]; ii < trainMat->rowptr[u+1]; ii++) {
      item = trainMat->rowind[ii];
      //skip if sampled user and item are in the submatrix 
      if (isInsideBlock(u, item, uStart, uEnd, iStart, iEnd)) {
        continue;
      }
      itemRat = trainMat->rowval[ii];
      diff = itemRat - estRating(u, item);
      rmse += diff*diff;
    }
    uRegErr += dotProd(uFac[u], uFac[u], facDim);
  }
  uRegErr = uRegErr*uReg;
  
  for (item = 0; item < nItems; item++) {
    if (item >= iStart && item < iEnd) {
      continue;
    }
    iRegErr += dotProd(iFac[item], iFac[item], facDim);
  }
  iRegErr = iRegErr*iReg;

  obj = rmse + uRegErr + iRegErr;
    
  //std::cout <<"\nrmse: " << std::scientific << rmse << " uReg: " << uRegErr << " iReg: " << iRegErr ; 

  return obj;
}


double Model::fullLowRankErr(const Data& data) {
  double r_ui_est, r_ui_orig, diff, rmse;
  rmse = 0;
  for (int u = 0; u < nUsers; u++) {
    for (int item = 0; item < nItems; item++) {
      r_ui_est = estRating(u, item);
      r_ui_orig = dotProd(data.origUFac[u], data.origIFac[item], data.origFacDim);
      diff = r_ui_orig - r_ui_est;
      rmse += diff*diff;
    }
  }
  rmse = sqrt(rmse/(nUsers*nItems));
  return rmse;
}


double Model::subMatKnownRankErr(const Data& data, int uStart, int uEnd,
    int iStart, int iEnd) {
  double r_ui_est, r_ui_orig, diff, rmse;
  rmse = 0;
  for (int u = uStart; u <= uEnd; u++) {
    for (int item = iStart; item <= iEnd; item++) {
      r_ui_est = estRating(u, item);
      r_ui_orig = dotProd(data.origUFac[u], data.origIFac[item], data.origFacDim);
      diff = r_ui_orig - r_ui_est;
      rmse += diff*diff;
    }
  }
  rmse = sqrt(rmse/((uEnd-uStart+1)*(iEnd-iStart+1)));
  return rmse;
}


//start inclusive, end exclusive
double Model::subMatKnownRankNonObsErr(const Data& data, int uStart, int uEnd,
    int iStart, int iEnd) {
  
  double r_ui_est, r_ui_orig, diff, seKnown, seUnknown, rmseUnknown;
  int u, ii, item, nnzKnown;

  seKnown = 0;
  seUnknown = 0;
  nnzKnown = 0;

  gk_csr_t *mat = data.trainMat;

  for (u = uStart; u < uEnd; u++) {
    for (ii = mat->rowptr[u]; ii < mat->rowptr[u+1]; ii++) {
      item = mat->rowind[ii];
      if (!isInsideBlock(u, item, uStart, uEnd, iStart, iEnd)) {
        continue;
      }
      r_ui_est = estRating(u, item);
      r_ui_orig = dotProd(data.origUFac[u], data.origIFac[item], data.origFacDim);
      diff = r_ui_orig - r_ui_est;
      seKnown += diff*diff;
      nnzKnown++;
    }
  }

  for (u = uStart; u < uEnd; u++) {
    for (item = iStart; item < iEnd; item++) {
      r_ui_est = estRating(u, item);
      r_ui_orig = dotProd(data.origUFac[u], data.origIFac[item], data.origFacDim);
      diff = r_ui_orig - r_ui_est;
      seUnknown += diff*diff;
    }
  }

  seUnknown = seUnknown - seKnown;
  rmseUnknown = sqrt(seUnknown/(((uEnd-uStart)*(iEnd-iStart)) - nnzKnown));

  return rmseUnknown;
}


//start inclusive, end exclusive
double Model::subMatKnownRankNonObsErrWSet(const Data& data, int uStart, int uEnd,
    int iStart, int iEnd, std::set<int> exUSet, std::set<int> exISet) {
  
  double r_ui_est, r_ui_orig, diff, seKnown, seUnknown, rmseUnknown;
  int u, ii, item, nnzKnown, nnzUnknown;

  seKnown = 0;
  seUnknown = 0;
  nnzKnown = 0;
  nnzUnknown = 0;
  gk_csr_t *mat = data.trainMat;

  for (u = uStart; u < uEnd; u++) {
    auto search = exUSet.find(u);
    if (search != exUSet.end()) {
      //found in set
      continue;
    }
    for (ii = mat->rowptr[u]; ii < mat->rowptr[u+1]; ii++) {
      item = mat->rowind[ii];
      search = exISet.find(item);
      if (search != exISet.end()) {
        //found in set
        continue;
      }
      if (!isInsideBlock(u, item, uStart, uEnd, iStart, iEnd)) {
        continue;
      }
      r_ui_est = estRating(u, item);
      r_ui_orig = dotProd(data.origUFac[u], data.origIFac[item], data.origFacDim);
      diff = r_ui_orig - r_ui_est;
      seKnown += diff*diff;
      nnzKnown++;
    }
  }

  for (u = uStart; u < uEnd; u++) {
    auto search = exUSet.find(u);
    if (search != exUSet.end()) {
      //found in set
      continue;
    }
    for (item = iStart; item < iEnd; item++) {
      search = exISet.find(item);
      if (search != exISet.end()) {
        //found in set
        continue;
      }
      r_ui_est = estRating(u, item);
      r_ui_orig = dotProd(data.origUFac[u], data.origIFac[item], data.origFacDim);
      diff = r_ui_orig - r_ui_est;
      seUnknown += diff*diff;
      nnzUnknown++;
    }
  }

  seUnknown = seUnknown - seKnown;
  nnzUnknown = nnzUnknown - nnzKnown;
  std::cout << "\n("<< uStart << "," << uEnd << "," << iStart 
    << "," << iEnd << ") " << "seUnknown: " << seUnknown << " seKnown: " << seKnown 
    << " nnzKnown: " << nnzKnown  << " nnzUnknown: " << nnzUnknown;
  rmseUnknown = sqrt(seUnknown/nnzUnknown);

  return rmseUnknown;
}

std::vector<std::tuple<int, int, float>> Model::getUIRatings(gk_csr_t* mat, 
    std::unordered_set<int>& invalidUsers, 
    std::unordered_set<int>& invalidItems) {
  std::vector<std::tuple<int, int, float>> uiRatings;
  for (int u = 0; u < mat->nrows; u++) {
    //skip if in invalid users
    auto search = invalidUsers.find(u);
    if (search != invalidUsers.end()) {
      //found and skip
      continue;
    }
    
    for (int ii = mat->rowptr[u]; ii < mat->rowptr[u+1]; ii++) {
      int item = mat->rowind[ii];
      //skip if in invalid items
      search = invalidItems.find(item);
      if (search != invalidItems.end()) {
        //found and skip
        continue;
      }
      uiRatings.push_back(std::make_tuple(u, item, estRating(u, item)));
    }
  }
  return uiRatings;
}


void Model::updateMatWRatings(gk_csr_t *mat) {
  
  for (int u = 0; u < mat->nrows; u++) {
    for (int ii = mat->rowptr[u]; ii < mat->rowptr[u+1]; ii++) {
      int item = mat->rowind[ii];
      mat->rowval[ii] = estRating(u, item);
    }
  }

}


//define constructor
Model::Model(const Params& params) {

  nUsers    = params.nUsers;
  nItems    = params.nItems;
  facDim    = params.facDim;
  uReg      = params.uReg;
  iReg      = params.iReg;
  learnRate = params.learnRate;
  rhoRMS    = params.rhoRMS;
  maxIter   = params.maxIter;
  trainSeed = -1;

  std::default_random_engine generator (params.seed);
  std::uniform_real_distribution<double> dist (0.0, 1.0);

  //init user latent factors
  uFac.assign(nUsers, std::vector<double>(facDim, 0));
  for (auto& uf: uFac) {
    for (auto& v: uf) {
      v = dist(generator);    
    }
  }


  //init item latent factors
  iFac.assign(nItems, std::vector<double>(facDim, 0));
  for (auto& itemf: iFac) {
    for (auto& v: itemf) {
      v = dist(generator);
    }
  }
 
  //init user biases
  uBias = std::vector<double>(nUsers, 0);
  for (auto& v: uBias) {
    v = dist(generator);  
  }

  //init item biases
  iBias = std::vector<double>(nItems, 0);
  for (auto& v: iBias) {
    v = dist(generator);
  }

}


Model::Model(int p_nUsers, int p_nItems, const Params& params):Model(params) {
  nUsers    = p_nUsers;
  nItems    = p_nItems;
}


Model::Model(const Params& params, int seed) : Model(params) {
  trainSeed = seed;
}


Model::Model(const Params& params, const char* uFacName, const char* iFacName,
    int seed):Model(params, seed) {
  std::cout << "\nLoading user factors: " << uFacName;
  readMat(uFac, nUsers, facDim, uFacName);
  std::cout << "\nLoading item factors: " << iFacName;
  readMat(iFac, nItems, facDim, iFacName);
}


Model::Model(const Params& params, const char* uFacName, const char* iFacName,
    const char* uBFName, const char *iBFName, const char *gBFName, 
    int seed):Model(params, seed) {
  std::cout << "\nLoading user factors: " << uFacName;
  readMat(uFac, nUsers, facDim, uFacName);
  std::cout << "\nLoading item factors: " << iFacName;
  readMat(iFac, nItems, facDim, iFacName);

  std::cout << "\nLoading user bias: " << uBFName;
  uBias = readDVector(uBFName);
  std::cout << "\nuBias norm: " << normVec(uBias);
  
  std::cout << "\nLoading item bias: " << iBFName;
  iBias = readDVector(iBFName);
  std::cout << "\niBias norm: " << normVec(iBias);

  //read global bias
  std::cout << "\nLoading global bias...";
  std::vector<double> gBias;
  gBias = readDVector(gBFName);
  mu = gBias[0]; 
  std::cout << "\nglobal bias: " << mu << std::endl;
}




