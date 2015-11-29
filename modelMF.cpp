#include "modelMF.h"


void ModelMF::updateFac(double *fac, double *grad) {
  for (int i = 0; i < facDim; i++) {
    fac[i] -= learnRate * grad[i];
  }
}


void ModelMF::updateAdaptiveFac(double *fac, double *grad,
    double *gradAcc) {
  for (int i = 0; i < facDim; i++) {
    gradAcc[i] = gradAcc[i]*rhoRMS + (1.0-rhoRMS)*grad[i]*grad[i];
    fac[i] -= (learnRate/sqrt(gradAcc[i]+0.0000001)) * grad[i];
  }
}


void ModelMF::computeUGrad(int user, int item, float r_ui, 
        double *uGrad) {
  //estimate rating on the item
  double r_ui_est = dotProd(uFac[user], iFac[item], facDim);
  double diff = r_ui - r_ui_est;

  //initialize gradients to 0
  for (int i = 0; i < facDim; i++) {
    uGrad[i] = -2.0*diff*iFac[item][i] + 2.0*uReg*uFac[user][i];
  }

}
 

void ModelMF::computeIGrad(int user, int item, float r_ui, 
        double *iGrad) {
  //estimate rating on the item
  double r_ui_est = dotProd(uFac[user], iFac[item], facDim);
  double diff = r_ui - r_ui_est;

  //initialize gradients to 0
  for (int i = 0; i < facDim; i++) {
    iGrad[i] = -2.0*diff*uFac[user][i] + 2.0*iReg*iFac[item][i];
  }

}


void ModelMF::gradCheck(int u, int item, float r_ui) {
  /*
  int i;
  std::vector<double> grad (facDim, 0);
  std::vector<double> tempFac (facDim, 0);
  double lossRight, lossLeft, gradE;

  double r_ui_est = std::inner_product(uFac[u].begin(), uFac[u].end(),
                                        iFac[item].begin(), 0.0);
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
  r_ui_est = std::inner_product(tempFac.begin(), tempFac.end(),
                                  iFac[item].begin(), 0.0);
  lossRight = (r_ui - r_ui_est)*(r_ui - r_ui_est);
  
  //perturb user with -E and compute loss
  tempFac = uFac[u];
  for(auto& v: tempFac) {
    v = v - 0.0001; 
  }
  r_ui_est = std::inner_product(tempFac.begin(), tempFac.end(),
                                  iFac[item].begin(), 0.0);
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
  r_ui_est = std::inner_product(tempFac.begin(), tempFac.end(),
                                  uFac[u].begin(), 0.0);
  lossRight = (r_ui - r_ui_est)*(r_ui - r_ui_est);
  
  //perturb user with -E and compute loss
  tempFac = iFac[item];
  for(auto& v: tempFac) {
    v = v - 0.0001; 
  }
  r_ui_est = std::inner_product(tempFac.begin(), tempFac.end(),
                                  uFac[u].begin(), 0.0);
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
  */
}


void ModelMF::train(const Data &data, Model &bestModel) {

  std::cout << "\nModelMF::train";

  int u, iter, subIter, bestIter;
  int item, nUserItems, itemInd;
  float itemRat;
  double bestObj, prevObj;
  int nnz = data.trainNNZ;

  gk_csr_t *trainMat = data.trainMat;

  //array to hold user and item gradients
  double *uGrad = new double[facDim];
  double *iGrad = new double[facDim];
 
  //hold user gradient accumulation
  double **uGradsAcc = new double*[nUsers];
  for (u = 0; u < nUsers; u++) {
    uGradsAcc[u] = new double[facDim];
  }

  //item gradient accumulation
  double **iGradsAcc = new double*[nItems];
  for (item = 0; item < nItems; item++) {
    iGradsAcc[item] = new double[facDim];
  }

  //std::cout << "\nNNZ = " << nnz;
  prevObj = objective(data);
  std::cout << "\nInit obj: " << prevObj;

  for (iter = 0; iter < maxIter; iter++) {  
    for (subIter = 0; subIter < nnz; subIter++) {
      
      //sample u
      u = std::rand() % nUsers;
      
      //sample item rated by user
      nUserItems =  trainMat->rowptr[u+1] - trainMat->rowptr[u];
      itemInd = std::rand()%nUserItems; 
      item = trainMat->rowind[trainMat->rowptr[u] + itemInd];
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
    if (iter % OBJ_ITER == 0) {
      if (isTerminateModel(bestModel, data, iter, bestIter, bestObj, prevObj)) {
        break; 
      }
      std::cout << "\nIter: " << iter << " Objective: " << std::scientific << prevObj 
                << " Train RMSE: " << RMSE(data.trainMat);
    }
  
  }

  //std::cout << "\nNum Iter: " << iter << " Best Iter: " << bestIter
  //  << " Best obj: " << std::scientific << bestObj ;
  
  delete[] uGrad;
  delete[] iGrad;
  for (u = 0; u < nUsers; u++) {
    delete[] uGradsAcc[u];
  }
  delete[] uGradsAcc;
  for (item = 0; item < nItems; item++) {
    delete[] iGradsAcc[item];
  }
  delete[] iGradsAcc;
}



