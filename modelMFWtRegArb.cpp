#include "modelMFWtRegArb.h"

/*
 * Learning with weighted trace norm under arbitary sampling distribution
 */

double ModelMFWtRegArb::objective(const Data& data) {

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
    uRegErr += uMarg[u]*std::inner_product(uFac[u].begin(), 
                                           uFac[u].end(), 
                                           uFac[u].begin(), 0.0);
  }
  uRegErr = uRegErr*uReg;
  
  for (item = 0; item < nItems; item++) {
    iRegErr += iMarg[item]*std::inner_product(iFac[item].begin(), 
                                             iFac[item].end(),
                                             iFac[item].begin(), 0.0);
  }
  iRegErr = iRegErr*iReg;

  obj = rmse + uRegErr + iRegErr;
    
  //std::cout <<"\nrmse: " << std::scientific << rmse << " uReg: " << uRegErr << " iReg: " << iRegErr ; 

  return obj;
}


void ModelMFWtRegArb::train(const Data& data, Model& bestModel) {
  
  std::cout<<"\nModelMFWtRegArb::train";
  
  //compute empirical distribution of ratings in rows and cols
  computeMarginals(data);
  
  ModelMF::train(data, bestModel);
}


void ModelMFWtRegArb::computeMarginals(const Data &data) {
  int u, ii, item, nnz;
  gk_csr_t *trainMat  = data.trainMat;
  
  //copy for lambda expression
  int l_nUsers = nUsers;
  int l_nItems = nItems;
  double l_alpha = alpha;

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
  std::transform(uMarg.begin(), uMarg.end(), uMarg.begin(), 
                  [=](double d){return l_alpha*(d/nnz) + (1-l_alpha)*(1.0/l_nUsers);});
  std::transform(iMarg.begin(), iMarg.end(), iMarg.begin(), 
                  [=](double d){return l_alpha*(d/nnz) + (1-l_alpha)*(1.0/l_nItems);});
}


void ModelMFWtRegArb::computeUGrad(int user, int item, float r_ui, 
        std::vector<double> &uGrad) {
  
  //estimate rating on the item
  double r_ui_est = std::inner_product(begin(uFac[user]), end(uFac[user]), 
                                        begin(iFac[item]), 0.0);
  double diff = r_ui - r_ui_est;

  //initialize gradients to 0
  std::fill(uGrad.begin(), uGrad.end(), 0);
  for (int i = 0; i < facDim; i++) {
    uGrad[i] = -2.0*diff*iFac[item][i] + 
                2.0*uReg*uMarg[user]*uFac[user][i];
  }
}


void ModelMFWtRegArb::computeIGrad(int user, int item, float r_ui, 
        std::vector<double> &iGrad) {
  //estimate rating on the item
  double r_ui_est = std::inner_product(uFac[user].begin(), uFac[user].end(), 
                                        iFac[item].begin(), 0.0);
  double diff = r_ui - r_ui_est;

  //initialize gradients to 0
  std::fill(iGrad.begin(), iGrad.end(), 0);
  for (int i = 0; i < facDim; i++) {
    iGrad[i] = -2.0*diff*uFac[user][i] + 
                2.0*iReg*iMarg[item]*iFac[item][i];
  } 

}






