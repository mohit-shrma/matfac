#include "modelMF.h"


void ModelMF::updateFac(std::vector<double> &fac, std::vector<double> &grad,
    std::vector<double> &gradAcc) {
  for (int i = 0; i < facDim; i++) {
    gradAcc[i] = gradAcc[i]*rhoRMS + (1.0-rhoRMS)*grad[i]*grad[i];
    fac[i] = (learnRate/sqrt(gradAcc[i]+0.0000001)) * grad[i];
  }
}


void ModelMF::computeUGrad(int user, int item, float r_ui, 
        std::vector<double> &uGrad) {
  //estimate rating on the item
  double r_ui_est = std::inner_product(begin(uFac[user]), end(uFac[user]), 
                                        begin(iFac[item]), 0.0);
  double diff = r_ui - r_ui_est;

  //initialize gradients to 0
  std::fill(uGrad.begin(), uGrad.end(), 0);
  for (int i = 0; i < facDim; i++) {
    uGrad[i] = -2.0*diff*iFac[item][i] + 2.0*uReg*uFac[user][i];
  }

}
 

void ModelMF::computeIGrad(int user, int item, float r_ui, 
        std::vector<double> &iGrad) {
  //estimate rating on the item
  double r_ui_est = std::inner_product(uFac[user].begin(), uFac[user].end(), 
                                        iFac[item].begin(), 0.0);
  double diff = r_ui - r_ui_est;

  //initialize gradients to 0
  std::fill(iGrad.begin(), iGrad.end(), 0);
  for (int i = 0; i < facDim; i++) {
    iGrad[i] = -2.0*diff*uFac[user][i] + 2.0*iReg*iFac[item][i];
  }

}


void ModelMF::train(const Data &data, Model &bestModel) {
  
  int u, iter, subIter, bestIter;
  int item, nUserItems, itemInd;
  float itemRat;
  double bestObj, prevObj;
  int nnz = 0;


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

  //find nnz in train matrix
  for (u = 0; u < trainMat->nrows; u++) {
    nnz += trainMat->rowptr[u+1] - trainMat->rowptr[u];
  }
  
  std::cout << "\nNNZ = " << nnz;
  prevObj = objective(data);
  std::cout << "\nInit obj: " << prevObj;

  for (iter = 0; iter < maxIter; iter++) {  
    for (subIter = 0; subIter < nnz; subIter++) {
      
      //sample u
      u = rand() % nUsers;
      
      //sample item rated by user
      nUserItems =  trainMat->rowptr[u+1] - trainMat->rowptr[u];
      itemInd = rand()%nUserItems; 
      item = trainMat->rowind[trainMat->rowptr[u] + itemInd];
      itemRat = trainMat->rowval[trainMat->rowptr[u] + itemInd]; 
      
      //compute user gradient
      computeUGrad(u, item, itemRat, uGrad);

      //update user
      updateFac(uFac[u], uGrad, uGradsAcc[u]); 

      //compute item gradient
      computeIGrad(u, item, itemRat, iGrad);

      //update item
      updateFac(iFac[item], iGrad, iGradsAcc[item]);

    }

    //check objective
    //TODO: OBJ_ITER
    if (iter % OBJ_ITER == 0) {
      if (isTerminateModel(bestModel, data, iter, bestIter, bestObj, prevObj)) {
        break; 
      }
      std::cout << "\nIter: " << iter << " Objective: " << std::scientific <<prevObj ;
    }
  
  }

  std::cout << "\nNum Iter: " << iter << " Best Iter: " << bestIter
    << " Best obj: " << std::scientific << bestObj ;


}



