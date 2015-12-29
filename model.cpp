#include "model.h"


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


double Model::subMatRMSE(gk_csr_t *mat, int uStart, int uEnd, 
    int iStart, int iEnd) {
  double r_ui_est, diff, rmse, r_ui;
  int u, ii, item, nnz;
  
  rmse = 0;
  nnz = 0;
  
  for (u = uStart; u < uEnd; u++) {
    for (ii = mat->rowptr[u]; ii < mat->rowptr[u+1]; ii++) {
      item = mat->rowind[ii];
      if (item < iStart || item >= iEnd) {
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


double Model::subMatExRMSE(gk_csr_t *mat, int uStart, int uEnd, 
    int iStart, int iEnd) {
  double r_ui_est, diff, rmse, r_ui;
  int u, ii, item, nnz;
  
  rmse = 0;
  nnz = 0;
  
  for (u = uStart; u < uEnd; u++) {
    for (ii = mat->rowptr[u]; ii < mat->rowptr[u+1]; ii++) {
      item = mat->rowind[ii];
      if (item >= iStart && item < iEnd) {
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
      if (item < iStart || item >= iEnd) {
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
  


}


Model::Model(int p_nUsers, int p_nItems, const Params& params) {

  nUsers    = p_nUsers;
  nItems    = p_nItems;
  facDim    = params.facDim;
  uReg      = params.uReg;
  iReg      = params.iReg;
  learnRate = params.learnRate;
  rhoRMS    = params.rhoRMS;
  maxIter   = params.maxIter;

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
  
}


