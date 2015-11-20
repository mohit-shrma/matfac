

class ModelMF : public Model {

  public:

    ModelMF(const Params& params) : Model(params) {}
    
}


ModelMF::void train(const Data &data) {
  
  int u, iter, subIter;
  int item, nUserItems, itemInd;
  float itemRat;
  int nnz = 0;

  gk_csr_t *trainMat = data.trainMat;

  //array to hold user and item gradients
  std::array<double, facDim> uGrad{};
  std::array<double, facDim> iGrad{};
 
  //TODO: verify whether below are zero init
  //array to hold user gradient accumulation
  std::array<std::array<double, facDim> , nUsers> uGradsAcc{}; 
  
  //array to hold item gradient accumulation
  std::array<std::array<double, facDim> , nItems> iGradsAcc{}; 

  //find nnz in train matrix
  for (u = 0; u < trainMat->nrows; u++) {
    nnz += trainMat->rowptr[u+1] - trainMat->rowptr[u];
  }
  
  std::cout << "\nNNZ = " << nnz;

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
      //update user
      //compute item gradient
      //update item
    }
    //check objective
    //
  }


}



