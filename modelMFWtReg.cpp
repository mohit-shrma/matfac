#include "modelMFWtReg.h"

/*
 * Collaborative filtering in a non-uniform world: learning with the weighted
 * trace norm
 */

//following computes objective of a weighted trace norm
double ModelMFWtReg::objective(const Data& data) {

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
    uRegErr += pow(uMarg[u], alpha-1)*std::inner_product(uFac[u].begin(), 
                                                      uFac[u].end(), 
                                                      uFac[u].begin(), 0.0);
  }
  uRegErr = uRegErr*uReg;
  
  for (item = 0; item < nItems; item++) {
    iRegErr += pow(iMarg[item], alpha-1)*std::inner_product(iFac[item].begin(), 
                                                    iFac[item].end(),
                                                    iFac[item].begin(), 0.0);
  }
  iRegErr = iRegErr*iReg;

  obj = rmse + uRegErr + iRegErr;
    
  //std::cout <<"\nrmse: " << std::scientific << rmse << " uReg: " << uRegErr << " iReg: " << iRegErr ; 

  return obj;
}


void ModelMFWtReg::train(const Data& data, Model& bestModel) {
  //compute no. of ratings in rows and cols
  computeMarginals(data);
  //divide regularization by nnz
  uReg = uReg/data.trainNNZ;
  iReg = iReg/data.trainNNZ;
  //std::cout<<"\nuReg: " << uReg;
  //std::cout<<"\niReg: " << iReg;
  
  ModelMF::train(data, bestModel);
}


//following compute no. of ratings in a row and col
void ModelMFWtReg::computeMarginals(const Data &data) {
  int u, ii, item, nnz;
  gk_csr_t *trainMat  = data.trainMat;
  
  nnz = 0;
  for (u = 0; u < nUsers; u++) {
    for (ii = trainMat->rowptr[u]; ii < trainMat->rowptr[u+1]; ii++) {
      item = trainMat->rowind[ii];
      uMarg[u]++;
      iMarg[item]++;
      nnz++;
    }
  }
  
  //divide no. of entries per user & item by nnz
  //std::transform(uMarg.begin(), uMarg.end(), uMarg.begin(), 
  //                [](double d){return d/nnz;});
  //std::transform(iMarg.begin(), iMarg.end(), iMarg.begin(), 
  //                [](double d){return d/nnz;});
}


void ModelMFWtReg::computeUGrad(int user, int item, float r_ui, 
        std::vector<double> &uGrad) {
  
  //estimate rating on the item
  double r_ui_est = std::inner_product(begin(uFac[user]), end(uFac[user]), 
                                        begin(iFac[item]), 0.0);
  double diff = r_ui - r_ui_est;

  //initialize gradients to 0
  std::fill(uGrad.begin(), uGrad.end(), 0);
  for (int i = 0; i < facDim; i++) {
    uGrad[i] = -2.0*diff*iFac[item][i] + 
                2.0*uReg*pow(uMarg[user], alpha-1)*uFac[user][i];
  }
}
 

void ModelMFWtReg::computeIGrad(int user, int item, float r_ui, 
        std::vector<double> &iGrad) {
  //estimate rating on the item
  double r_ui_est = std::inner_product(uFac[user].begin(), uFac[user].end(), 
                                        iFac[item].begin(), 0.0);
  double diff = r_ui - r_ui_est;

  //initialize gradients to 0
  std::fill(iGrad.begin(), iGrad.end(), 0);
  for (int i = 0; i < facDim; i++) {
    iGrad[i] = -2.0*diff*uFac[user][i] + 
                2.0*iReg*pow(iMarg[item], alpha-1)*iFac[item][i];
  } 

}




