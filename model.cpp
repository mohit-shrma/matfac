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
      r_ui_est = std::inner_product(uFac[u].begin(), uFac[u].end(), 
          iFac[i].begin(), 0.0);
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
      r_ui_est = std::inner_product(uFac[u].begin(), uFac[u].end(), 
          iFac[i].begin(), 0.0);
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
      r_ui_est = std::inner_product(uFac[u].begin(), uFac[u].end(), 
          iFac[i].begin(), 0.0);
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
      diff = itemRat - std::inner_product(uFac[u].begin(), uFac[u].end(),
                                          iFac[item].begin(), 0.0);
      rmse += diff*diff;
    }
    uRegErr += std::inner_product(uFac[u].begin(), uFac[u].end(), 
                                  uFac[u].begin(), 0.0);
  }
  uRegErr = uRegErr*uReg;
  
  for (item = 0; item < nItems; item++) {
    iRegErr += std::inner_product(iFac[item].begin(), iFac[item].end(),
                                  iFac[item].begin(), 0.0);
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
      r_ui_est = std::inner_product(uFac[u].begin(), uFac[u].end(),
                                      iFac[item].begin(), 0.0);
      r_ui_orig = std::inner_product(data.origUFac[u].begin(), 
                                    data.origUFac[u].end(),
                                    data.origIFac[item].begin(), 0.0);
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
      r_ui_est = std::inner_product(uFac[u].begin(), uFac[u].end(),
                                    iFac[item].begin(), 0.0);
      r_ui_orig = std::inner_product(data.origUFac[u].begin(), 
                                     data.origUFac[u].end(),
                                     data.origIFac[item].begin(), 0.0);
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
      v = (double)rand() / (double) (1.0 + RAND_MAX);    
    }
  }


  //init item latent factors
  iFac.assign(nItems, std::vector<double>(facDim, 0));
  for (auto& itemf: iFac) {
    for (auto& v: itemf) {
      v = (double)rand() / (double) (1.0 + RAND_MAX);
    }
  }
  


}


