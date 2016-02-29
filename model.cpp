#include "model.h"


void Model::save(std::string prefix) {
  //save user latent factors
  std::string uFacName = prefix + "_uFac_" + std::to_string(nUsers) + "_" 
    + std::to_string(facDim) + "_" + std::to_string(uReg) + ".mat";
  writeMat(uFac, nUsers, facDim, uFacName.c_str());
  
  //save item latent factors
  std::string iFacName = prefix + "_iFac_" + std::to_string(nItems) + "_" 
    + std::to_string(facDim) + "_" + std::to_string(iReg) + ".mat";
  writeMat(iFac, nItems, facDim, iFacName.c_str());

  //save user bias
  std::string uBFName = prefix + "_uBias_" + std::to_string(nUsers) + "_" 
    + std::to_string(uReg) + ".vec";
  writeVector(uBias, uBFName.c_str());

  //save item bias
  std::string iBFName = prefix + "_iBias_" + std::to_string(nItems) + "_" 
    + std::to_string(iReg) + ".vec";
  writeVector(iBias, iBFName.c_str());

  //TODO:save global bias 
  std::vector<double> gBias = {mu};
  std::string gBFName = prefix + "_gBias_" + std::to_string(nItems) + "_" 
    + std::to_string(iReg) + ".vec";
  writeVector(gBias, gBFName.c_str());
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
      r_ui_est = dotProd(uFac[u], iFac[i], facDim);
      diff = r_ui - r_ui_est;
      rmse += diff*diff;
      nnz++;
    }
  }
  rmse = sqrt(rmse/nnz);
  
  return rmse;
}


double Model::estRating(int user, int item) {
  return dotProd(uFac[user], iFac[item], facDim);
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
      r_ui_est = dotProd(uFac[u], iFac[item], facDim);
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
      r_ui_est = dotProd(uFac[u], iFac[item], facDim);
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
      r_ui_est = dotProd(uFac[u], iFac[i], facDim);
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
      r_ui_est = dotProd(uFac[u], iFac[i], facDim);
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
      diff = itemRat - dotProd(uFac[u], iFac[item], facDim);
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
      diff = itemRat - dotProd(uFac[u], iFac[item], facDim);
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
      diff = itemRat - dotProd(uFac[u], iFac[item], facDim);
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
      r_ui_est = dotProd(uFac[u], iFac[item], facDim);
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
      r_ui_est = dotProd(uFac[u], iFac[item], facDim);
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
      r_ui_est = dotProd(uFac[u], iFac[item], facDim);
      r_ui_orig = dotProd(data.origUFac[u], data.origIFac[item], data.origFacDim);
      diff = r_ui_orig - r_ui_est;
      seKnown += diff*diff;
      nnzKnown++;
    }
  }

  for (u = uStart; u < uEnd; u++) {
    for (item = iStart; item < iEnd; item++) {
      r_ui_est = dotProd(uFac[u], iFac[item], facDim);
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
      r_ui_est = dotProd(uFac[u], iFac[item], facDim);
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
      r_ui_est = dotProd(uFac[u], iFac[item], facDim);
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
 
  //init user latent factors
  uFac.assign(nUsers, std::vector<double>(facDim, 0));
  for (auto& uf: uFac) {
    for (auto& v: uf) {
      v = (double)std::rand() / (double) (1.0 + RAND_MAX);    
    }
  }


  //init item latent factors
  iFac.assign(nItems, std::vector<double>(facDim, 0));
  for (auto& itemf: iFac) {
    for (auto& v: itemf) {
      v = (double)std::rand() / (double) (1.0 + RAND_MAX);
    }
  }
 
  //init user biases
  uBias = std::vector<double>(nUsers, 0);
  for (auto& v: uBias) {
    v = (double)std::rand() / (double) (1.0 + RAND_MAX);
  }

  //init item biases
  iBias = std::vector<double>(nItems, 0);
  for (auto& v: iBias) {
    v = (double)std::rand() / (double) (1.0 + RAND_MAX);
  }


  //TODO: global bias
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
    const char* iBFName, const char *uBFName, 
    const, int seed):Model(params, seed) {
  std::cout << "\nLoading user factors: " << uFacName;
  readMat(uFac, nUsers, facDim, uFacName);
  std::cout << "\nLoading item factors: " << iFacName;
  readMat(iFac, nItems, facDim, iFacName);
  std::cout << "\nLoading item bias..." << iBFName;
  iBias = readVector(iBFName);
  std::cout << "\nLoading user bias..." << uBFName;
  uBias = readVector(uBFName);
  //TODO: read global bias
}




