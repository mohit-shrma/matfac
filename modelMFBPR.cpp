#include "modelMFBPR.h"


std::vector<std::tuple<int, int, float>> ModelMFBPR::getBPRUIRatings(gk_csr_t* mat) {
  std::vector<std::tuple<int, int, float>> uiRatings;
  for (int u = 0; u < mat->nrows; u++) {
    for (int ii = mat->rowptr[u]; ii < mat->rowptr[u+1]; ii++) {
      int item = mat->rowind[ii];
      float rating = mat->rowval[ii];
      if (rating > 0) {
        uiRatings.push_back(std::make_tuple(u, item, rating));
      }
    }
  }
  return uiRatings;
}


int ModelMFBPR::sampleNegItem(int u, const gk_csr_t* trainMat, 
    std::unordered_set<int>& trainItems, 
    std::unordered_set<int>& testItems, 
    std::unordered_set<int>& valItems) const {
  
  int j = -1, jj;
  int nUserItems, start, end;
  int nTrainItems = trainMat->ncols;//trainItems.size();
  int32_t *ui_rowind = trainMat->rowind;
  ssize_t *ui_rowptr = trainMat->rowptr;
  float   *ui_rowval = trainMat->rowval;
  int nTry = 0; 
  nUserItems = ui_rowptr[u+1] - ui_rowptr[u];
  //sample neg item
  while(nTry < 100) {
    jj = std::rand()%nUserItems;
    if (ui_rowval[jj + ui_rowptr[u]] == 0.0) {
      //explicit 0
      j = ui_rowind[jj + ui_rowptr[u]];
      break;
    } else {
      //search for implicit 0
      
      if (0 == jj) {
        start = 0;
        end = ui_rowind[ui_rowptr[u]]; //first rated item by u
      } else if (nUserItems-1 == jj) {
        start = ui_rowind[ui_rowptr[u] + jj] + 1; //item next to last rated item
        end = nTrainItems;
      } else {
        start = ui_rowind[ui_rowptr[u] + jj] + 1; //item next to jjth item
        end = ui_rowind[ui_rowptr[u] + jj + 1]; //item rated after jjth item
      }

      //check for empty interval
      if (end - start > 0) {
        j = std::rand()%(end-start) + start;
      } else {
        continue;
      }

      //make sure sampled -ve item not present in testSet and valSet
      if (testItems.find(j) != testItems.end() ||
          valItems.find(j) != valItems.end()) {
        //found in either set
        continue;
      }

      if (trainItems.find(j) != trainItems.end()) {
        break;
      }
    }
    nTry++;
  } //end while
  
  if (100 == nTry) {
    j = -1;
  }

  return j;
}

void ModelMFBPR::train(const Data& data, Model& bestModel, 
    std::unordered_set<int>& invalidUsers, 
    std::unordered_set<int>& invalidItems) {

  int u, pI, nI;
  float itemRat, r_ui_est, r_uj_est;
  double bestRecall, valRecall;
  double bestHR, valHR;

  std::cout << "\nModelMFBPR::train trainSeed: " << trainSeed << std::endl;
  std::chrono::time_point<std::chrono::system_clock> start, end;
 
  //random engine
  std::mt19937 mt(trainSeed);
  
  //get non-zero ratings from training data
  const auto uiRatings = getBPRUIRatings(data.trainMat); 
  //index to above uiRatings pair
  std::vector<size_t> uiRatingInds(uiRatings.size());
  std::iota(uiRatingInds.begin(), uiRatingInds.end(), 0);

  gk_csr_t *mat = data.trainMat;
  std::unordered_set<int> trainItems, testItems, valItems;
  for (int item = 0; item < mat->ncols; item++) {
    if (mat->colptr[item+1] - mat->colptr[item] > 0) {
      trainItems.insert(item);
    }
  }

  mat = data.testMat;
  for (int item = 0; item < mat->ncols; item++) {
    if (mat->colptr[item+1] - mat->colptr[item] > 0) {
      testItems.insert(item);
    }
  }

  mat = data.valMat;
  for (int item = 0; item < mat->ncols; item++) {
    if (mat->colptr[item+1] - mat->colptr[item] > 0) {
      valItems.insert(item);
    }
  }

  for (int iter = 0; iter < maxIter; iter++) {
    
    start = std::chrono::system_clock::now();
    if (iter % 10 == 0) {
      //shuffle the user item rating indexes
      std::shuffle(uiRatingInds.begin(), uiRatingInds.end(), mt);
    } else {
      parBlockShuffle(uiRatingInds, mt);
    }

    for (const auto& ind: uiRatingInds) {
      u = std::get<0>(uiRatings[ind]);
      pI = std::get<0>(uiRatings[ind]);
      
      r_ui_est = uFac.row(u).dot(iFac.row(pI));
      
      //sample -ve item
      nI = sampleNegItem(u, data.trainMat, trainItems, testItems, valItems);
      if (-1 == nI) {
        //failed to sample negativ item
        continue;
      }
      
      r_uj_est = uFac.row(u).dot(iFac.row(nI));
       
      double r_uij = r_ui_est - r_uj_est;
      double expCoeff = -1.0 /(1.0 + exp(r_uij));
      
      //update user
      for (int i = 0; i < facDim; i++) {
        uFac(u, i) -= learnRate*(expCoeff*(iFac.row(pI, i) - iFac.row(nI, i))
                                  + 2.0*uReg*(uFac(u, i)));
      } 
            
      //update item
      for (int i = 0; i < facDim; i++) {
        iFac(pI, i) -= learnRate*(expCoeff*uFac.row(u, i) + 2.0*iReg*iFac(pI, i));
        iFac(nI, i) -= learnRate*(-expCoeff*uFac.row(u,i) + 2.0*iReg*iFac(nI, i));
      }


    }

  } 

}







