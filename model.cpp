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
    + "_" + std::to_string(origLearnRate);
  
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
  std::cout << "user bias norm: " << uBias.norm() << std::endl;


  //save item bias
  std::string iBFName = prefix + "_iBias_" + modelSign + ".vec";
  writeVector(iBias, iBFName.c_str());
  std::cout << "item bias norm: " << iBias.norm() << std::endl;

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
  uBias = readEigVector(uBFName.c_str());
  std::cout << "user bias norm: " << uBias.norm() << std::endl;

  //read item bias
  std::string iBFName = prefix + "_iBias_" + modelSign + ".vec";
  iBias = readEigVector(iBFName.c_str());
  std::cout << "item bias norm: " << iBias.norm() << std::endl;

  //read global bias 
  std::vector<double> gBias;
  std::string gBFName = prefix + "_" + modelSign + "_gBias";
  gBias = readDVector(gBFName.c_str());
  mu = gBias[0];  
}


void Model::saveFacs(std::string prefix) {
  std::cout << "Saving model... " << prefix << std::endl;
  std::string modelSign = modelSignature();
  //save user latent factors
  std::string uFacName = prefix + "_uFac_" + modelSign + ".mat";
  writeMat(uFac, nUsers, facDim, uFacName.c_str());
  std::cout << "uFac Norm: " << uFac.norm() << std::endl;
  
  //save item latent factors
  std::string iFacName = prefix + "_iFac_" + modelSign +  ".mat";
  writeMat(iFac, nItems, facDim, iFacName.c_str());
  std::cout << "iFac Norm: " << iFac.norm() << std::endl;
}


void Model::loadFacs(std::string prefix) {
  
  std::string modelSign = modelSignature();
  //read user latent factors
  std::string uFacName = prefix + "_uFac_" + modelSign + ".mat";
  std::cout << "Loading user factors: " << uFacName << std::endl;
  //load if file exists
  if (isFileExist(uFacName.c_str())) {
    readMat(uFac, nUsers, facDim, uFacName.c_str());
    std::cout << "uFac Norm: " << uFac.norm() << std::endl;
  } else {
    std::cout << "File doesn't exist: " << uFacName << std::endl;
  }

  //read item latent factors
  std::string iFacName = prefix + "_iFac_" + modelSign +  ".mat";
  std::cout << "Loading item factors: " << iFacName << std::endl;
  if (isFileExist(iFacName.c_str())) { 
    readMat(iFac, nItems, facDim, iFacName.c_str());
    std::cout << "iFac Norm: " << iFac.norm() << std::endl;
  } else {
    std::cout << "File doesn't exist: " << iFacName << std::endl;
  }

}


void Model::saveBinFacs(std::string prefix) {
  std::string modelSign = modelSignature();
  //save user latent factors
  std::string uFacName = prefix + "_uFac_" + modelSign + ".binmat";
  writeMatBin(uFac, nUsers, facDim, uFacName.c_str());
  
  //save item latent factors
  std::string iFacName = prefix + "_iFac_" + modelSign +  ".binmat";
  writeMatBin(iFac, nItems, facDim, iFacName.c_str());
}


void Model::loadBinFacs(std::string prefix) {
  std::string modelSign = modelSignature();
  //read user latent factors
  std::string uFacName = prefix + "_uFac_" + modelSign + ".binmat";
  //load if file exists
  if (isFileExist(uFacName.c_str())) {
    std::cout << "Loading user factors: " << uFacName << std::endl;
    readMatBin(uFac, nUsers, facDim, uFacName.c_str());
  }

  //read item latent factors
  std::string iFacName = prefix + "_iFac_" + modelSign +  ".binmat";
  if (isFileExist(iFacName.c_str())) { 
    std::cout << "Loading item factors: " << iFacName << std::endl;
    readMatBin(iFac, nItems, facDim, iFacName.c_str());
  }
}


void Model::load(const char* uFacName, const char* iFacName, const char* uBFName,
    const char* iBFName, const char*gBFName) {
  //read user latent factors
  readMat(uFac, nUsers, facDim, uFacName);
  
  //read item latent factors
  readMat(iFac, nItems, facDim, iFacName);

  //read user bias
  uBias = readEigVector(uBFName);

  //read item bias
  iBias = readEigVector(iBFName);

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
  int nnz;
  double r_ui, r_ui_est, diff, rmse;

  nnz = 0;
  rmse = 0;

#pragma omp parallel for reduction(+:rmse, nnz) private(r_ui, r_ui_est, diff)
  for (int u = 0; u < nUsers; u++) {
    
    //skip if invalid user
    if (invalidUsers.count(u) > 0) {
      //found and skip
      continue;
    }

    for (int ii = mat->rowptr[u]; ii < mat->rowptr[u+1]; ii++) {
      
      int item = mat->rowind[ii];
      //skip if invalid item
      if (invalidItems.count(item) > 0 || item >= nItems) {
        //found and skip
        continue;
      }
      
      r_ui     = mat->rowval[ii];
      r_ui_est = estRating(u, item);
      diff     = r_ui - r_ui_est;
      rmse     += diff*diff;
      nnz++;
    }
  }

  rmse = sqrt(rmse/nnz);
  
  return rmse;
}


double Model::RMSEUser(gk_csr_t *mat, std::unordered_set<int>& invalidUsers,
    std::unordered_set<int>& invalidItems, int u) {
  int nnz;
  double r_ui, r_ui_est, diff, rmse;

  nnz = 0;
  rmse = 0;
 
  //skip if invalid user 
  if (invalidUsers.count(u) > 0 || u >= mat->nrows) {
    //found and skip
    //std::cout << "RMSEUser: invalid user " << u << std::endl;
    return -1.0;
  }

  for (int ii = mat->rowptr[u]; ii < mat->rowptr[u+1]; ii++) {
    
    int item = mat->rowind[ii];
    //skip if invalid item
    if (invalidItems.count(item) > 0 || item >= nItems) {
      //found and skip
      continue;
    }
    
    r_ui     = mat->rowval[ii];
    r_ui_est = estRating(u, item);
    diff     = r_ui - r_ui_est;
    rmse     += diff*diff;
    nnz++;
  }

  rmse = sqrt(rmse/nnz);
  return rmse;
}


double Model::RMSEItem(gk_csr_t *mat, std::unordered_set<int>& invalidUsers,
    std::unordered_set<int>& invalidItems, int item) {
  int nnz;
  double r_ui, r_ui_est, diff, rmse;

  nnz = 0;
  rmse = 0;
    
  //skip if invalid user
  if (invalidItems.count(item) > 0 || item >= mat->ncols) {
    //found and skip
    //std::cout << "RMSEItem: invalid item " << item << std::endl;
    return -1.0;
  }

  for (int uu = mat->colptr[item]; uu < mat->colptr[item+1]; uu++) {

    int u = mat->colind[uu];
    //skip if invalid item
    if (invalidUsers.count(u) > 0 || u >= nUsers) {
      //found and skip
      continue;
    }
    
    r_ui     = mat->colval[uu];
    r_ui_est = estRating(u, item);
    diff     = r_ui - r_ui_est;
    rmse     += diff*diff;
    nnz++;
  }

  rmse = sqrt(rmse/nnz);
  
  return rmse;
}


std::pair<double, double> Model::hiLoNorms(std::unordered_set<int>& items) {

  int count = 0;
  float hiNorm = 0;
  float loNorm = 0;

  for (auto& item: items) {
      for (int k = 0; k < facDim/2; k++) {
        loNorm += iFac(item, k)*iFac(item, k);   
      }
      for (int k = facDim/2; k < facDim; k++) {
        hiNorm += iFac(item, k)*iFac(item, k);
      }
      loNorm += sqrt(loNorm);
      hiNorm += sqrt(hiNorm);
  }

  return std::make_pair(hiNorm/items.size(), loNorm/items.size());
}


std::pair<int, double> Model::RMSE(gk_csr_t *mat, std::unordered_set<int>& filtItems,
    std::unordered_set<int>& invalidUsers, std::unordered_set<int>& invalidItems) {
  
  int nnz;
  double r_ui, r_ui_est, diff, rmse;
  
  nnz = 0;
  rmse = 0;

#pragma omp parallel for reduction(+:rmse, nnz) private(r_ui, r_ui_est, diff)
  for (int u = 0; u < nUsers; u++) {
    
    //skip if invalid user
    if (invalidUsers.count(u) > 0) {
      //found and skip
      continue;
    }

    for (int ii = mat->rowptr[u]; ii < mat->rowptr[u+1]; ii++) {
      
      int item = mat->rowind[ii];
      
      //skip if invalid item
      if (invalidItems.count(item) > 0 || item >= nItems) {
        //found and skip
        continue;
      }

      if (filtItems.count(item) == 0) {
        //filtered item not found, skip
        continue;
      }

      r_ui     = mat->rowval[ii];
      r_ui_est = estRating(u, item);
      diff     = r_ui - r_ui_est;
      rmse     += diff*diff;
      nnz++;

    }

  }
 
  rmse = sqrt(rmse/nnz);
  
  return std::make_pair(nnz, rmse);
}


std::pair<int, double> Model::SE(gk_csr_t *mat, std::unordered_set<int>& filtItems,
    std::unordered_set<int>& invalidUsers, std::unordered_set<int>& invalidItems) {
  
  int nnz;
  double r_ui, r_ui_est, diff, rmse;
  
  nnz = 0;
  rmse = 0;

#pragma omp parallel for reduction(+:rmse, nnz) private(r_ui, r_ui_est, diff)
  for (int u = 0; u < nUsers; u++) {
    
    //skip if invalid user
    if (invalidUsers.count(u) > 0) {
      //found and skip
      continue;
    }

    for (int ii = mat->rowptr[u]; ii < mat->rowptr[u+1]; ii++) {
      
      int item = mat->rowind[ii];
      
      //skip if invalid item
      if (invalidItems.count(item) > 0 || item >= nItems) {
        //found and skip
        continue;
      }

      if (filtItems.count(item) == 0) {
        //filtered item not found, skip
        continue;
      }

      r_ui     = mat->rowval[ii];
      r_ui_est = estRating(u, item);
      diff     = r_ui - r_ui_est;
      rmse     += diff*diff;
      nnz++;

    }

  }
 
  //rmse = sqrt(rmse/nnz);
  
  return std::make_pair(nnz, rmse);
}


std::pair<int, double> Model::RMSEU(gk_csr_t *mat, std::unordered_set<int>& filtUsers,
    std::unordered_set<int>& invalidUsers, std::unordered_set<int>& invalidItems) {
  
  int nnz;
  double r_ui, r_ui_est, diff, rmse;
  
  nnz = 0;
  rmse = 0;

#pragma omp parallel for reduction(+:rmse, nnz) private(r_ui, r_ui_est, diff)
  for (int u = 0; u < nUsers; u++) {
    
    //skip if invalid user and not in filtered user
    if (invalidUsers.count(u) > 0 || 0 == filtUsers.count(u)) {
      continue;
    }

    for (int ii = mat->rowptr[u]; ii < mat->rowptr[u+1]; ii++) {
      
      int item = mat->rowind[ii];
      
      //skip if invalid item
      if (invalidItems.count(item) > 0 || item >= nItems) {
        //found and skip
        continue;
      }

      r_ui     = mat->rowval[ii];
      r_ui_est = estRating(u, item);
      diff     = r_ui - r_ui_est;
      rmse     += diff*diff;
      nnz++;

    }

  }
 
  rmse = sqrt(rmse/nnz);
  
  return std::make_pair(nnz, rmse);
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
  return uFac.row(user).dot(iFac.row(item));
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


double Model::estAvgRating(std::unordered_set<int>& invalidUsers, 
    std::unordered_set<int>& invalidItems) {
  double avgRat = 0;
  int count = 0;
  for (int user = 0; user < nUsers; user++) {
    if (invalidUsers.find(user) != invalidUsers.end()) {
      continue;
    }
    for (int item = 0; item < nItems; item++) {
      if (invalidItems.find(item) != invalidItems.end()) {
        //invalid
        continue;
      }
      avgRat += estRating(user, item);
      count++;
    }
  }
  return avgRat/count;
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

  if (iter - bestIter >= CHANCE_ITER) {
    //can't go lower than best objective after 500 iterations
    //printf("\nNOT CONVERGED: bestIter:%d bestObj: %.10e"
    //    " currIter:%d currObj: %.10e", bestIter, bestObj, iter, currObj);
    ret = true;
  }
  
  if (fabs(prevObj - currObj) < EPS) {
    //convergence
    printf("\nConverged in iteration: %d prevObj: %.10e currObj: %.10e", iter,
            prevObj, currObj); 
    ret = true;
  }

  prevObj = currObj;

  return ret;
}


double Model::NDCG(std::unordered_set<int>& invalidUsers,
    std::unordered_set<int>& invalidItems, gk_csr_t* testMat) {
  
  int N = 10, nValUsers = 0;
  double ndcg = 0;
  
  auto sortDescUPredRatings = [] (const ItemActPredTriplet& s1, 
      const ItemActPredTriplet& s2) {
    return std::get<2>(s1) > std::get<2>(s2);
  };
  
  auto sortDescUActRatings = [] (const ItemActPredTriplet& s1, 
      const ItemActPredTriplet& s2) {
    return std::get<1>(s1) > std::get<1>(s2);
  };

#pragma omp parallel for reduction(+:ndcg, nValUsers)
  for (int u = 0; u < testMat->nrows; u++) {
    
    if (invalidUsers.count(u) > 0) {
      continue;
    }
    
    std::vector<ItemActPredTriplet> uIRatings;

    for (int ii = testMat->rowptr[u]; ii < testMat->rowptr[u+1]; ii++) {
      int item = testMat->rowind[ii];
      if (invalidItems.count(item) > 0) {continue;}
      float rat = testMat->rowval[ii];
      float predRat = estRating(u, item);
      uIRatings.push_back(ItemActPredTriplet(item, rat, predRat));    
      std::push_heap(uIRatings.begin(), uIRatings.end(), sortDescUPredRatings);
      if (uIRatings.size() > N) {
        std::pop_heap(uIRatings.begin(), uIRatings.end(), sortDescUPredRatings);
        uIRatings.pop_back();
      }
    }
    
    if (uIRatings.size() < 2) {
      continue;
    }

    std::sort(uIRatings.begin(), uIRatings.end(), sortDescUPredRatings);
    
    float u_ndcg = 0.0;
    for (int i = 0; i < N && i < uIRatings.size(); i++) {
      float rel = std::get<1>(uIRatings[i]);
      u_ndcg += (std::pow(2.0, rel) - 1) / std::log2((i+1)+1); 
    }

    std::sort(uIRatings.begin(), uIRatings.end(), sortDescUActRatings);

    float u_dcg_max = 0.0;
    for (int i = 0; i < N && i < uIRatings.size(); i++) {
      float rel = std::get<1>(uIRatings[i]);
      u_dcg_max += (std::pow(2.0, rel) - 1) / std::log2((i+1)+1); 
    }
    
    if (u_dcg_max > EPS) {
      ndcg += u_ndcg/u_dcg_max;
    } else {
      continue;
    }
    
    nValUsers++;
  }
  
  //std::cout << "nValUsers: " << nValUsers << std::endl;

  return ndcg/nValUsers;
}


std::pair<int, double> Model::NDCGU(std::unordered_set<int>& filtUsers,
    std::unordered_set<int>& invalidUsers,
    std::unordered_set<int>& invalidItems, gk_csr_t* testMat) {
  
  int N = 10, nValUsers = 0;
  double ndcg = 0;
  
  auto sortDescUPredRatings = [] (const ItemActPredTriplet& s1, 
      const ItemActPredTriplet& s2) {
    return std::get<2>(s1) > std::get<2>(s2);
  };

  auto sortDescUActRatings = [] (const ItemActPredTriplet& s1, 
      const ItemActPredTriplet& s2) {
    return std::get<1>(s1) > std::get<1>(s2);
  };

#pragma omp parallel for reduction(+:ndcg, nValUsers)
  for (int u = 0; u < testMat->nrows; u++) {
    
    if (invalidUsers.count(u) > 0 || filtUsers.count(u) == 0) {
      continue;
    }
    
    std::vector<ItemActPredTriplet> uIRatings;

    for (int ii = testMat->rowptr[u]; ii < testMat->rowptr[u+1]; ii++) {
      int item = testMat->rowind[ii];
      if (invalidItems.count(item) > 0) {
        continue;
      }
      float rat = testMat->rowval[ii];
      float predRat = estRating(u, item);
      uIRatings.push_back(ItemActPredTriplet(item, rat, predRat));    
      std::push_heap(uIRatings.begin(), uIRatings.end(), sortDescUPredRatings);
      if (uIRatings.size() > N) {
        std::pop_heap(uIRatings.begin(), uIRatings.end(), sortDescUPredRatings);
        uIRatings.pop_back();
      }
    }
   
    if (uIRatings.size() < 2) {
      continue;
    }

    std::sort(uIRatings.begin(), uIRatings.end(), sortDescUPredRatings);
    
    float u_ndcg = 0.0;
    for (int i = 0; i < N && i < uIRatings.size(); i++) {
      float rel = std::get<1>(uIRatings[i]);
      u_ndcg += (std::pow(2.0, rel) - 1) / std::log2((i+1)+1); 
    }

    std::sort(uIRatings.begin(), uIRatings.end(), sortDescUActRatings);

    float u_dcg_max = 0.0;
    for (int i = 0; i < N && i < uIRatings.size(); i++) {
      float rel = std::get<1>(uIRatings[i]);
      u_dcg_max += (std::pow(2.0, rel) - 1) / std::log2((i+1)+1); 
    }

    if (u_dcg_max > EPS) {
      ndcg += u_ndcg/u_dcg_max;
    } else {
      continue;
    }
    
    nValUsers++;
  }
  //std::cout << "ndcg: " << ndcg << " nValUsers: " << nValUsers << std::endl;
  return std::make_pair(nValUsers, ndcg/nValUsers);
}


std::pair<int, double> Model::NDCGI(std::unordered_set<int>& filtItems, 
    std::unordered_set<int>& invalidUsers,
    std::unordered_set<int>& invalidItems, gk_csr_t* testMat) {
  
  int N = 10, nValUsers = 0;
  double ndcg = 0;
  
  auto sortDescUPredRatings = [] (const ItemActPredTriplet& s1, 
      const ItemActPredTriplet& s2) {
    return std::get<2>(s1) > std::get<2>(s2);
  };
  
  auto sortDescUActRatings = [] (const ItemActPredTriplet& s1, 
      const ItemActPredTriplet& s2) {
    return std::get<1>(s1) > std::get<1>(s2);
  };

#pragma omp parallel for reduction(+:ndcg, nValUsers)
  for (int u = 0; u < testMat->nrows; u++) {
    
    if (invalidUsers.count(u) > 0) {
      continue;
    }
    
    std::vector<ItemActPredTriplet> uIRatings;

    for (int ii = testMat->rowptr[u]; ii < testMat->rowptr[u+1]; ii++) {
      int item = testMat->rowind[ii];
      if (invalidItems.count(item) > 0 || filtItems.count(item) == 0) {
        continue;
      }
      float rat = testMat->rowval[ii];
      float predRat = estRating(u, item);
      uIRatings.push_back(ItemActPredTriplet(item, rat, predRat));    
      std::push_heap(uIRatings.begin(), uIRatings.end(), sortDescUPredRatings);
      if (uIRatings.size() > N) {
        std::pop_heap(uIRatings.begin(), uIRatings.end(), sortDescUPredRatings);
        uIRatings.pop_back();
      }
    }
    
    if (uIRatings.size() < 2) {
      continue;
    }

    std::sort(uIRatings.begin(), uIRatings.end(), sortDescUPredRatings);
    
    float u_ndcg = 0.0;
    for (int i = 0; i < N && i < uIRatings.size(); i++) {
      float rel = std::get<1>(uIRatings[i]);
      u_ndcg += (std::pow(2.0, rel) - 1) / std::log2((i+1)+1); 
    }

    std::sort(uIRatings.begin(), uIRatings.end(), sortDescUActRatings);
    
    float u_dcg_max = 0.0;
    for (int i = 0; i < N && i < uIRatings.size(); i++) {
      float rel = std::get<1>(uIRatings[i]);
      u_dcg_max += (std::pow(2.0, rel) - 1) / std::log2((i+1)+1); 
    }

    if (u_dcg_max > EPS) {
      ndcg += u_ndcg/u_dcg_max;
    } else {
      continue;
    }
    
    nValUsers++;
  }
  //std::cout << "ndcg: " << ndcg << " nValUsers: " << nValUsers << std::endl;
  return std::make_pair(nValUsers, ndcg/nValUsers);
}


double Model::arHR(const Data& data, std::unordered_set<int>& invalidUsers,
    std::unordered_set<int>& invalidItems, gk_csr_t* testMat) {
  
  gk_csr_t* trainMat = data.trainMat;
  int N  = 1000;
  double nHits = 0, nValUsers = 0;
#pragma omp parallel for reduction(+:nHits, nValUsers)
  for (int u = 0; u < trainMat->nrows; u++) {
    
    if (invalidUsers.count(u) > 0) {
      continue;
    }

    int testItem = testMat->rowind[testMat->rowptr[u]];
    std::unordered_set<int> uTrItems;
    std::vector<std::pair<int, double>> topNItemRat;

    for (int ii = trainMat->rowptr[u]; ii < trainMat->rowptr[u+1]; ii++) {
      int item = trainMat->rowind[ii];
      uTrItems.insert(item);
    }

    std::make_heap(topNItemRat.begin(), topNItemRat.end(), descComp);
    
    for (int item = 0; item < trainMat->ncols; item++) {
      if (uTrItems.count(item) || invalidItems.count(item) > 0) {
        continue;
      }
      topNItemRat.push_back(std::make_pair(item, estRating(u, item))); 
      std::push_heap(topNItemRat.begin(), topNItemRat.end(), descComp);
      
      if (topNItemRat.size() > N) {
        std::pop_heap(topNItemRat.begin(), topNItemRat.end(), descComp);
        topNItemRat.pop_back();
      }
    }
    
    std::sort(topNItemRat.begin(), topNItemRat.end(), descComp);
    
    for (int pos = 0; pos < topNItemRat.size(); pos++) {
      if (testItem == topNItemRat[pos].first) {
        //hit
        nHits += 1.0/(pos+1);
        break;
      }    
    }

    nValUsers++;
  }
  
  //std::cout << "nHits: " << nHits << " nValUsers: " << nValUsers << std::endl;

  return (double)nHits/(double)nValUsers;
}


std::pair<double, double> Model::arHRI(const Data& data, std::unordered_set<int>& filtItems, 
    std::unordered_set<int>& invalidUsers,
    std::unordered_set<int>& invalidItems, gk_csr_t* testMat) {
  
  gk_csr_t* trainMat = data.trainMat;
  int N  = 1000;
  double nHits = 0, nValUsers = 0;

#pragma omp parallel for reduction(+:nHits, nValUsers)
  for (int u = 0; u < trainMat->nrows; u++) {
    
    if (invalidUsers.count(u) > 0) {
      continue;
    }

    int testItem = testMat->rowind[testMat->rowptr[u]];
    //ignore if test item is not in items to be filtered
    if (filtItems.count(testItem) == 0) {
      continue;
    }

    std::unordered_set<int> uTrItems;
    std::vector<std::pair<int, double>> topNItemRat;

    for (int ii = trainMat->rowptr[u]; ii < trainMat->rowptr[u+1]; ii++) {
      int item = trainMat->rowind[ii];
      uTrItems.insert(item);
    }

    std::make_heap(topNItemRat.begin(), topNItemRat.end(), descComp);
    
    for (int item = 0; item < trainMat->ncols; item++) {
      if (uTrItems.count(item) || invalidItems.count(item) > 0) {
        continue;
      }
      topNItemRat.push_back(std::make_pair(item, estRating(u, item))); 
      std::push_heap(topNItemRat.begin(), topNItemRat.end(), descComp);
      
      if (topNItemRat.size() > N) {
        std::pop_heap(topNItemRat.begin(), topNItemRat.end(), descComp);
        topNItemRat.pop_back();
      }
    }
    
    std::sort(topNItemRat.begin(), topNItemRat.end(), descComp);
    
    for (int pos = 0; pos < topNItemRat.size(); pos++) {
      if (testItem == topNItemRat[pos].first) {
        //hit
        nHits += 1.0/(pos+1);
        break;
      }    
    }
    
    nValUsers++;
  }
  
  //std::cout << "nHits: " << nHits << " nValUsers: " << nValUsers << std::endl;
  auto countHR = std::make_pair(nHits, (double)nHits/(double)nValUsers);
  return countHR;
}


std::pair<double, double> Model::arHRU(const Data& data, std::unordered_set<int>& filtUsers,
    std::unordered_set<int>& invalidUsers,
    std::unordered_set<int>& invalidItems, gk_csr_t* testMat) {
  
  gk_csr_t* trainMat = data.trainMat;
  int N  = 1000;
  double nHits = 0, nValUsers = 0;

#pragma omp parallel for reduction(+:nHits, nValUsers)
  for (int u = 0; u < trainMat->nrows; u++) {
    
    if (invalidUsers.count(u) > 0 || filtUsers.count(u) == 0) {
      continue;
    }

    int testItem = testMat->rowind[testMat->rowptr[u]];
    std::unordered_set<int> uTrItems;
    std::vector<std::pair<int, double>> topNItemRat;

    for (int ii = trainMat->rowptr[u]; ii < trainMat->rowptr[u+1]; ii++) {
      int item = trainMat->rowind[ii];
      uTrItems.insert(item);
    }

    std::make_heap(topNItemRat.begin(), topNItemRat.end(), descComp);
    
    for (int item = 0; item < trainMat->ncols; item++) {
      if (uTrItems.count(item) || invalidItems.count(item) > 0) {
        continue;
      }
      topNItemRat.push_back(std::make_pair(item, estRating(u, item))); 
      std::push_heap(topNItemRat.begin(), topNItemRat.end(), descComp);
      
      if (topNItemRat.size() > N) {
        std::pop_heap(topNItemRat.begin(), topNItemRat.end(), descComp);
        topNItemRat.pop_back();
      }
    }
    
    std::sort(topNItemRat.begin(), topNItemRat.end(), descComp);
    
    for (int pos = 0; pos < topNItemRat.size(); pos++) {
      if (testItem == topNItemRat[pos].first) {
        //hit
        nHits += 1.0/(pos+1);
        break;
      }    
    }

    nValUsers++;
  }
  
  //std::cout << "nHits: " << nHits << " nValUsers: " << nValUsers << std::endl;
  auto countHR = std::make_pair(nHits, (double)nHits/(double)nValUsers);
  return countHR;
}


double Model::hitRate(const Data& data, std::unordered_set<int>& invalidUsers,
    std::unordered_set<int>& invalidItems, gk_csr_t* testMat) {
  
  gk_csr_t* trainMat = data.trainMat;
  int N  = 10;
  int nHits = 0, nValUsers = 0;
#pragma omp parallel for reduction(+:nHits, nValUsers)
  for (int u = 0; u < trainMat->nrows; u++) {
    
    if (invalidUsers.count(u) > 0) {
      continue;
    }

    int testItem = testMat->rowind[testMat->rowptr[u]];
    std::unordered_set<int> uTrItems;
    std::vector<std::pair<int, double>> topNItemRat;

    for (int ii = trainMat->rowptr[u]; ii < trainMat->rowptr[u+1]; ii++) {
      int item = trainMat->rowind[ii];
      uTrItems.insert(item);
    }

    std::make_heap(topNItemRat.begin(), topNItemRat.end(), descComp);
    
    for (int item = 0; item < trainMat->ncols; item++) {
      if (uTrItems.count(item) || invalidItems.count(item) > 0) {
        continue;
      }
      topNItemRat.push_back(std::make_pair(item, estRating(u, item))); 
      std::push_heap(topNItemRat.begin(), topNItemRat.end(), descComp);
      
      if (topNItemRat.size() > N) {
        std::pop_heap(topNItemRat.begin(), topNItemRat.end(), descComp);
        topNItemRat.pop_back();
      }
    }
    
    std::sort(topNItemRat.begin(), topNItemRat.end(), descComp);
    
    for (int pos = 0; pos < topNItemRat.size(); pos++) {
      if (testItem == topNItemRat[pos].first) {
        //hit
        nHits++;
        break;
      }    
    }

    nValUsers++;
  }
  
  //std::cout << "nHits: " << nHits << " nValUsers: " << nValUsers << std::endl;

  return (double)nHits/(double)nValUsers;
}


std::pair<int, double> Model::hitRateI(const Data& data, std::unordered_set<int>& filtItems, 
    std::unordered_set<int>& invalidUsers,
    std::unordered_set<int>& invalidItems, gk_csr_t* testMat) {
  
  gk_csr_t* trainMat = data.trainMat;
  int N  = 10;
  int nHits = 0, nValUsers = 0, nValItems = 0;

#pragma omp parallel for reduction(+:nHits, nValUsers, nValItems)
  for (int u = 0; u < trainMat->nrows; u++) {
    
    if (invalidUsers.count(u) > 0) {
      continue;
    }

    int testItem = testMat->rowind[testMat->rowptr[u]];
    if (filtItems.count(testItem) == 0) {
      continue;
    }

    std::unordered_set<int> uTrItems;
    std::vector<std::pair<int, double>> topNItemRat;

    for (int ii = trainMat->rowptr[u]; ii < trainMat->rowptr[u+1]; ii++) {
      int item = trainMat->rowind[ii];
      uTrItems.insert(item);
    }

    std::make_heap(topNItemRat.begin(), topNItemRat.end(), descComp);
    
    for (int item = 0; item < trainMat->ncols; item++) {
      if (uTrItems.count(item) || invalidItems.count(item) > 0) {
        continue;
      }
      topNItemRat.push_back(std::make_pair(item, estRating(u, item))); 
      std::push_heap(topNItemRat.begin(), topNItemRat.end(), descComp);
      
      if (topNItemRat.size() > N) {
        std::pop_heap(topNItemRat.begin(), topNItemRat.end(), descComp);
        topNItemRat.pop_back();
      }
    }
    
    std::sort(topNItemRat.begin(), topNItemRat.end(), descComp);
    
    for (int pos = 0; pos < topNItemRat.size(); pos++) {
      if (testItem == topNItemRat[pos].first) {
        //hit
        nHits++;
        break;
      }    
    }
    
    nValItems++;
    nValUsers++;
  }
  
  //std::cout << "nHits: " << nHits << " nValUsers: " << nValUsers << std::endl;
  auto countHR = std::make_pair(nHits, (double)nHits/(double)nValItems);
  return countHR;
}


std::pair<int, double> Model::hitRateU(const Data& data, std::unordered_set<int>& filtUsers,
    std::unordered_set<int>& invalidUsers,
    std::unordered_set<int>& invalidItems, gk_csr_t* testMat) {
  
  gk_csr_t* trainMat = data.trainMat;
  int N  = 10;
  int nHits = 0, nValUsers = 0;

#pragma omp parallel for reduction(+:nHits, nValUsers)
  for (int u = 0; u < trainMat->nrows; u++) {
    
    if (invalidUsers.count(u) > 0 || filtUsers.count(u) == 0) {
      continue;
    }

    int testItem = testMat->rowind[testMat->rowptr[u]];
    std::unordered_set<int> uTrItems;
    std::vector<std::pair<int, double>> topNItemRat;

    for (int ii = trainMat->rowptr[u]; ii < trainMat->rowptr[u+1]; ii++) {
      int item = trainMat->rowind[ii];
      uTrItems.insert(item);
    }

    std::make_heap(topNItemRat.begin(), topNItemRat.end(), descComp);
    
    for (int item = 0; item < trainMat->ncols; item++) {
      if (uTrItems.count(item) || invalidItems.count(item) > 0) {
        continue;
      }
      topNItemRat.push_back(std::make_pair(item, estRating(u, item))); 
      std::push_heap(topNItemRat.begin(), topNItemRat.end(), descComp);
      
      if (topNItemRat.size() > N) {
        std::pop_heap(topNItemRat.begin(), topNItemRat.end(), descComp);
        topNItemRat.pop_back();
      }
    }
    
    std::sort(topNItemRat.begin(), topNItemRat.end(), descComp);
    
    for (int pos = 0; pos < topNItemRat.size(); pos++) {
      if (testItem == topNItemRat[pos].first) {
        //hit
        nHits++;
        break;
      }    
    }

    nValUsers++;
  }
  
  //std::cout << "nHits: " << nHits << " nValUsers: " << nValUsers << std::endl;
  auto countHR = std::make_pair(nHits, (double)nHits/(double)nValUsers);
  return countHR;
}


bool Model::isTerminateModelHR(Model& bestModel, const Data& data, int iter,
    int& bestIter, double& bestHR, double& prevHR, 
    std::unordered_set<int>& invalidUsers, 
    std::unordered_set<int>& invalidItems) {
  bool ret = false;
  double currHR = hitRate(data, invalidUsers, invalidItems, data.valMat);

  if (currHR > bestHR) {
    bestModel = *this;
    bestHR = currHR;
    bestIter = iter;
  }

  if (iter - bestIter >= 100) {
    //half the learning rate
    if (learnRate > 1e-5) {
      learnRate = learnRate/2;
    }
  }

  if (iter - bestIter >= CHANCE_ITER) {
    //can't go lower than best objective after 500 iterations
    //printf("\nNOT CONVERGED: bestIter:%d bestObj: %.10e"
    //    " currIter:%d currObj: %.10e", bestIter, bestObj, iter, currObj);
    ret = true;
  }
  
  
  /*
  if (fabs(prevHR - currHR) < EPS) {
    //convergence
    printf("\nConverged in iteration: %d prevHR: %.10e currHR: %.10e", iter,
            prevHR, currHR); 
    ret = true;
  }
  */

  prevHR = currHR;

  return ret;
}


bool Model::isTerminateModelNDCG(Model& bestModel, const Data& data, int iter,
    int& bestIter, double& bestNDCG, double& prevNDCG, 
    std::unordered_set<int>& invalidUsers, 
    std::unordered_set<int>& invalidItems) {
  bool ret = false;
  double currNDCG = NDCG(invalidUsers, invalidItems, data.valMat);

  if (currNDCG > bestNDCG) {
    bestModel = *this;
    bestNDCG = currNDCG;
    bestIter = iter;
  }

  if (iter - bestIter >= 100) {
    //half the learning rate
    if (learnRate > 1e-5) {
      learnRate = learnRate/2;
    }
  }

  if (iter - bestIter >= CHANCE_ITER) {
    //can't go lower than best objective after 500 iterations
    //printf("\nNOT CONVERGED: bestIter:%d bestObj: %.10e"
    //    " currIter:%d currObj: %.10e", bestIter, bestObj, iter, currObj);
    ret = true;
  }
  
  
  /*
  if (fabs(prevNDCG - currNDCG) < EPS) {
    //convergence
    printf("\nConverged in iteration: %d prevNDCG: %.10e currNDCG: %.10e", iter,
            prevNDCG, currNDCG); 
    ret = true;
  }
  */

  prevNDCG = currNDCG;

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
  
  //nan check
  if (currObj != currObj || currValRMSE != currValRMSE) {
    std::cout << "Found nan " << std::endl;
    //half learning rate
    if (learnRate > 1e-5) {
      //replace current model by best model
      *this = bestModel;
      learnRate = learnRate/2;
      return false;
    } else {
      return true;
    };
  }
    
  if (currValRMSE < bestValRMSE) {
    bestModel = *this;
    bestValRMSE = currValRMSE;
    bestIter = iter;
  }

  if (iter - bestIter >= 100) {
    //half the learning rate
    if (learnRate > 1e-5) {
      learnRate = learnRate/2;
    }
  }

  if (iter - bestIter >= CHANCE_ITER) {
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

  /*
  if (fabs(prevValRMSE - currValRMSE) < 0.0001) {
    printf("\nvalidation RMSE in iteration: %d prev: %.10e curr: %.10e" 
        " bestValRMSE: %.10e", iter, prevValRMSE, currValRMSE, bestValRMSE); 
    ret = true;
  }
  */

  prevObj = currObj;
  prevValRMSE = currValRMSE;

  return ret;
}


bool Model::isTerminateModelSing(Model& bestModel, const Data& data, int iter,
    int& bestIter, double& bestObj, double& prevObj, double& bestValRMSE,
    double& prevValRMSE, std::unordered_set<int>& invalidUsers, 
    std::unordered_set<int>& invalidItems) {
  bool ret = false;
  double currObj = objectiveSing(data, invalidUsers, invalidItems);
  double currValRMSE = -1;
  
  if (data.valMat) {
    currValRMSE = RMSE(data.valMat, invalidUsers, invalidItems);
  } else {
    std::cerr << "\nNo validation data" << std::endl;
    exit(0);
  }
  
  //nan check
  if (currObj != currObj || currValRMSE != currValRMSE) {
    std::cout << "Found nan " << std::endl;
    //half learning rate
    if (learnRate > 1e-5) {
      //replace current model by best model
      *this = bestModel;
      learnRate = learnRate/2;
      return false;
    } else {
      return true;
    };
  }
    
  if (currValRMSE < bestValRMSE) {
    bestModel = *this;
    bestValRMSE = currValRMSE;
    bestIter = iter;
  }

  if (iter - bestIter >= 100) {
    //half the learning rate
    if (learnRate > 1e-5) {
      learnRate = learnRate/2;
    }
  }

  if (iter - bestIter >= CHANCE_ITER) {
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

  /*
  if (fabs(prevValRMSE - currValRMSE) < 0.0001) {
    printf("\nvalidation RMSE in iteration: %d prev: %.10e curr: %.10e" 
        " bestValRMSE: %.10e", iter, prevValRMSE, currValRMSE, bestValRMSE); 
    ret = true;
  }
  */

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
    uRegErr += uFac.row(u).dot(uFac.row(u));
  }
  uRegErr = uRegErr*uReg;
  
  for (item = 0; item < nItems; item++) {
    iRegErr += iFac.row(item).dot(iFac.row(item));;
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
    uRegErr += uFac.row(u).dot(uFac.row(u));
  }
  uRegErr = uRegErr*uReg;

  for (item = 0; item < nItems; item++) {
    //skip if invalid item
    auto search = invalidItems.find(item);
    if (search != invalidItems.end()) {
      //found and skip
      continue;
    }
    iRegErr += iFac.row(u).dot(iFac.row(u));
  }
  iRegErr = iRegErr*iReg;

  obj = rmse + uRegErr + iRegErr;
    
  //std::cout <<"\nrmse: " << std::scientific << rmse << " uReg: " << uRegErr << " iReg: " << iRegErr ; 

  return obj;
}


double Model::objective(const Data& data, std::unordered_set<int>& invalidUsers,
    std::unordered_set<int>& invalidItems) {

  double rmse = 0, uRegErr = 0, iRegErr = 0, obj = 0;
  gk_csr_t *trainMat = data.trainMat;

#pragma omp parallel for reduction(+:rmse, uRegErr)
  for (int u = 0; u < nUsers; u++) {
    //skip if invalid user
    if (invalidUsers.count(u) > 0) {
      //found and skip
      continue;
    }
    for (int ii = trainMat->rowptr[u]; ii < trainMat->rowptr[u+1]; ii++) {
      int item = trainMat->rowind[ii];
      //skip if invalid item
      if (invalidItems.count(item) > 0) {
        continue;
      }

      float itemRat = trainMat->rowval[ii];
      double diff = itemRat - estRating(u, item);
      rmse += diff*diff;
    }
    uRegErr += uFac.row(u).dot(uFac.row(u));
  }
  uRegErr = uRegErr*uReg;
  
#pragma omp parallel for reduction(+: iRegErr)
  for (int item = 0; item < nItems; item++) {
    //skip if invalid item
    if (invalidItems.count(item) > 0) {
      //found and skip
      continue;
    }
    iRegErr += iFac.row(item).dot(iFac.row(item));
  }
  iRegErr = iRegErr*iReg;

  obj = rmse + uRegErr + iRegErr;
    
  //std::cout <<"\nrmse: " << std::scientific << rmse << " uReg: " << uRegErr 
  //  << " iReg: " << iRegErr << std::endl; 

  return obj;
}


double Model::objectiveSing(const Data& data, std::unordered_set<int>& invalidUsers,
    std::unordered_set<int>& invalidItems) {

  double rmse = 0, uRegErr = 0, iRegErr = 0, obj = 0;
  gk_csr_t *trainMat = data.trainMat;

#pragma omp parallel for reduction(+:rmse, uRegErr)
  for (int u = 0; u < nUsers; u++) {
    //skip if invalid user
    if (invalidUsers.count(u) > 0) {
      //found and skip
      continue;
    }
    for (int ii = trainMat->rowptr[u]; ii < trainMat->rowptr[u+1]; ii++) {
      int item = trainMat->rowind[ii];
      //skip if invalid item
      if (invalidItems.count(item) > 0) {
        continue;
      }

      float itemRat = trainMat->rowval[ii];
      double diff = itemRat - estRating(u, item);
      rmse += diff*diff;
    }
    for (int k = 0; k < facDim; k++) {
      uRegErr += uFac(u, k)*uFac(u, k)*singularVals(k); 
    }
  }
  
#pragma omp parallel for reduction(+: iRegErr)
  for (int item = 0; item < nItems; item++) {
    //skip if invalid item
    if (invalidItems.count(item) > 0) {
      //found and skip
      continue;
    }
    for (int k = 0; k < facDim; k++) {
      iRegErr += iFac(item, k)*iFac(item, k)*singularVals(k);
    }
  }

  obj = rmse + uRegErr + iRegErr;
    
  //std::cout <<"\nrmse: " << std::scientific << rmse << " uReg: " << uRegErr 
  //  << " iReg: " << iRegErr << std::endl; 

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
    uRegErr += uFac.row(u).dot(uFac.row(u));
  }
  uRegErr = uRegErr*uReg;
  
  for (item = iStart; item < iEnd; item++) {
    iRegErr += iFac.row(item).dot(iFac.row(item));
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
    uRegErr += uFac.row(u).dot(uFac.row(u)); 
  }
  uRegErr = uRegErr*uReg;
  
  for (item = 0; item < nItems; item++) {
    if (item >= iStart && item < iEnd) {
      continue;
    }
    iRegErr += iFac.row(item).dot(iFac.row(item));
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
      r_ui_orig = dotProd(data.origUFac[u], data.origIFac[item], data.facDim);
      diff = r_ui_orig - r_ui_est;
      rmse += diff*diff;
    }
  }
  rmse = sqrt(rmse/(nUsers*nItems));
  return rmse;
}


double Model::fullLowRankErr(const Data& data, 
    std::unordered_set<int>& invalidUsers, std::unordered_set<int>& invalidItems) {
  double rmse = 0, count = 0;
#pragma omp parallel for reduction(+ : rmse, count) 
  for (int u = 0; u < nUsers; u++) {
    //skip if invalid user
    if (invalidUsers.find(u) != invalidUsers.end()) {
      continue;
    }
    
    std::vector<bool> ratedItems(nItems, false);
    for (int ii = data.trainMat->rowptr[u]; ii < data.trainMat->rowptr[u+1]; ii++) {
      int item = data.trainMat->rowind[ii];
      ratedItems[item] = true;
    }

    double r_ui_est, r_ui_orig, diff;

    for (int item = 0; item < nItems; item++) {
      //skip if invalid item
      if (invalidItems.find(item) != invalidItems.end()) {
        continue;
      }

      //skip if rated item
      if (ratedItems[item]) {
        continue;
      }

      r_ui_est = estRating(u, item);
      r_ui_orig = dotProd(data.origUFac[u], data.origIFac[item], data.facDim);
      diff = r_ui_orig - r_ui_est;
      rmse += diff*diff;
      count += 1;
    }
  }
  rmse = sqrt(rmse/count);
  return rmse;
}


double Model::fullLowRankErr(const Data& data, 
    std::unordered_set<int>& invalidUsers, std::unordered_set<int>& invalidItems,
    Model& origModel) {
  double rmse = 0, count = 0;
#pragma omp parallel for reduction(+ : rmse, count) 
  for (int u = 0; u < nUsers; u++) {
    //skip if invalid user
    if (invalidUsers.find(u) != invalidUsers.end()) {
      continue;
    }
    
    std::vector<bool> ratedItems(nItems, false);
    for (int ii = data.trainMat->rowptr[u]; ii < data.trainMat->rowptr[u+1]; ii++) {
      int item = data.trainMat->rowind[ii];
      ratedItems[item] = true;
    }
    
    double r_ui_est, r_ui_orig, diff;

    for (int item = 0; item < nItems; item++) {
      //skip if invalid item
      if (invalidItems.find(item) != invalidItems.end()) {
        continue;
      }
      
      //skip if rated item
      if (ratedItems[item]) {
        continue;
      }

      r_ui_est = estRating(u, item);
      r_ui_orig = origModel.estRating(u, item);
      diff = r_ui_orig - r_ui_est;
      rmse += diff*diff;
      count += 1;
    }
  }
  rmse = sqrt(rmse/count);
  return rmse;
}


double Model::subMatKnownRankErr(const Data& data, int uStart, int uEnd,
    int iStart, int iEnd) {
  double r_ui_est, r_ui_orig, diff, rmse;
  rmse = 0;
  for (int u = uStart; u <= uEnd; u++) {
    for (int item = iStart; item <= iEnd; item++) {
      r_ui_est = estRating(u, item);
      r_ui_orig = dotProd(data.origUFac[u], data.origIFac[item], data.facDim);
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
      r_ui_orig = dotProd(data.origUFac[u], data.origIFac[item], data.facDim);
      diff = r_ui_orig - r_ui_est;
      seKnown += diff*diff;
      nnzKnown++;
    }
  }

  for (u = uStart; u < uEnd; u++) {
    for (item = iStart; item < iEnd; item++) {
      r_ui_est = estRating(u, item);
      r_ui_orig = dotProd(data.origUFac[u], data.origIFac[item], data.facDim);
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
      r_ui_orig = dotProd(data.origUFac[u], data.origIFac[item], data.facDim);
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
      r_ui_orig = dotProd(data.origUFac[u], data.origIFac[item], data.facDim);
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


void Model::updateMatWRatings(gk_csr_t *mat) {
  
  for (int u = 0; u < mat->nrows; u++) {
    for (int ii = mat->rowptr[u]; ii < mat->rowptr[u+1]; ii++) {
      int item = mat->rowind[ii];
      if (item < nItems) {
        mat->rowval[ii] = estRating(u, item);
      }
    }
  }

}


void Model::updateMatWRatingsGaussianNoise(gk_csr_t *mat) {
  std::mt19937 mt(1); //seed for gaussian noise is 1
  std::normal_distribution<> gauss(0, 0.01);
  for (int u = 0; u < mat->nrows; u++) {
    for (int ii = mat->rowptr[u]; ii < mat->rowptr[u+1]; ii++) {
      int item = mat->rowind[ii];
      if (item < nItems) {
        mat->rowval[ii] = estRating(u, item) + gauss(mt);
      }
    }
  }
}


std::vector<std::pair<double, double>> Model::itemsMeanVar(gk_csr_t* mat) {
  
  std::vector<std::pair<double, double>> meanVar;
  for (int item = 0; item < mat->ncols; item++) { 
    meanVar.push_back(std::make_pair(0,0));
  }

#pragma omp parallel for
  for (int item = 0; item < mat->ncols; item++) { 
    
    double itemMean = 0;
    double itemVar = 0;
    double nRatings = mat->colptr[item+1] - mat->colptr[item]; 

    for (int uu = mat->colptr[item]; uu < mat->colptr[item+1]; uu++) {
      int user = mat->colind[uu];
      float rating = estRating(user, item);
      itemMean += rating;
    }
    itemMean = itemMean/nRatings;

    for (int uu = mat->colptr[item]; uu < mat->colptr[item+1]; uu++) {
      int user = mat->colind[uu];
      float rating = estRating(user, item);
      float diff = rating - itemMean;
      itemVar += diff*diff;  
    }
    //itemVar = itemVar/(nRatings - 1); //unbiased
    itemVar = itemVar/(nRatings);
    
    meanVar[item] = std::make_pair(itemMean, itemVar);
  }
  
  return meanVar;
}


std::vector<std::pair<double, double>> Model::usersMeanVar(gk_csr_t* mat) {
  
  std::vector<std::pair<double, double>> meanVar;
  for (int u = 0; u < mat->nrows; u++) {
    meanVar.push_back(std::make_pair(0,0));
  }

#pragma omp parallel for
  for (int u = 0; u < mat->nrows; u++) {
    
    double uMean = 0;
    double uVar = 0;
    double nRatings = mat->rowptr[u+1] - mat->rowptr[u]; 

    for (int ii = mat->rowptr[u]; ii < mat->rowptr[u+1]; ii++) {
      int item = mat->rowind[ii];
      float rating = estRating(u, item);
      uMean += rating;
    }
    uMean = uMean/nRatings;

    for (int ii = mat->rowptr[u]; ii < mat->rowptr[u+1]; ii++) {
      int item = mat->rowind[ii];
      float rating = estRating(u, item);
      float diff = rating - uMean;
      uVar += diff*diff;  
    }
    //uVar = uVar/(nRatings - 1); //unbiased
    uVar = uVar/(nRatings);
    
    meanVar[u] = std::make_pair(uMean, uVar);
  }
  
  return meanVar;
}


void Model::initInfreqFactors(const Params& params, const Data& data) {
  std::vector<int> itemFreq(nItems, 0);
  std::vector<int> userFreq(nUsers, 0);
#pragma omp parallel for
  for (int item = 0; item < nItems; item++) {
    itemFreq[item] = (data.trainMat->colptr[item+1] - 
        data.trainMat->colptr[item]);
  }

#pragma omp parallel for
  for (int u = 0; u < nUsers; u++) {
    userFreq[u] = data.trainMat->rowptr[u+1] - data.trainMat->rowptr[u];
  }
  
  float lb = -0.01, ub = 0.01;
  std::default_random_engine generator (params.seed);
  std::uniform_real_distribution<double> dist (lb, ub);
  std::cout << "Infreq lb = " << lb << " ub = " << ub << std::endl;


  for (int u = 0; u < nUsers; u++) {
    if (userFreq[u] <= 30) {
      for (int k = 0; k < facDim; k++) {
        uFac(u,k) = dist(generator);
      }
    }
  }

  for (int i = 0; i < nItems; i++) {
    if (itemFreq[i] < 30) {
      for (int k = 0; k < facDim; k++) {
        iFac(i,k) = dist(generator);
      }
    }
  }

}


Model::Model(int nUsers, int nItems, int facDim) : nUsers(nUsers), 
  nItems(nItems), facDim(facDim) {}


//define constructor
Model::Model(const Params& params) {

  nUsers    = params.nUsers;
  nItems    = params.nItems;
  facDim    = params.facDim;
  uReg      = params.uReg;
  iReg      = params.iReg;
  sing_a    = params.uReg;
  sing_b    = params.iReg;
  learnRate = params.learnRate;
  origLearnRate = params.learnRate;
  rhoRMS    = params.rhoRMS;
  alpha     = params.alpha;
  maxIter   = params.maxIter;
  trainSeed = -1;

  std::default_random_engine generator (params.seed);
  float lb = -0.01, ub = 0.01;
  std::uniform_real_distribution<double> dist (lb, ub);
  std::cout << "lb = " << lb << " ub = " << ub << std::endl;

  //init user latent factors
  uFac = Eigen::MatrixXf(nUsers, facDim);
  for (int u = 0; u < nUsers; u++) {
    for (int k = 0; k < facDim; k++) {
      uFac(u,k) = dist(generator);
    }
  }

  //init item latent factors
  iFac = Eigen::MatrixXf(nItems, facDim);
  for (int i = 0; i < nItems; i++) {
    for (int k = 0; k < facDim; k++) {
      iFac(i,k) = dist(generator);
    }
  }

  //init user biases
  uBias = Eigen::VectorXf(nUsers);
  for (int u = 0; u < nUsers;  u++) {
    uBias(u) = dist(generator);
  }

  //init item biases
  iBias = Eigen::VectorXf(nItems);
  for (int i = 0; i < nItems; i++) {
    iBias(i) = dist(generator);
  }
  
  //init singular vals
  singularVals = Eigen::VectorXf(facDim);
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
  uBias = readEigVector(uBFName);
  std::cout << "\nuBias norm: " << uBias.norm();
  
  std::cout << "\nLoading item bias: " << iBFName;
  iBias = readEigVector(iBFName);
  std::cout << "\niBias norm: " << iBias.norm();

  //read global bias
  std::cout << "\nLoading global bias...";
  std::vector<double> gBias;
  gBias = readDVector(gBFName);
  mu = gBias[0]; 
  std::cout << "\nglobal bias: " << mu << std::endl;
}




