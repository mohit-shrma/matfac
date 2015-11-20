#include <iostream>
#include <cstdlib>

class Model {

  public:
    int nUsers;
    int nItems;
    int facDim;
    float learnRate;
    float rhoRMS;
    int maxIter;
    float uReg;
    float iReg;
    std::vector<std::vector<double> uFac; 
    std::vector<std::vector<double> iFac;

    //declare constructor
    Model(const Params& params);

    //declare virtual method for train
    virtual void train(const Data& data) {
      std::cerr<< "\nTraining not in base class";
    };
}

//define constructor
Model::Model(const Params& params) {
  int i;

  nUsers    = params.nUsers;
  nItems    = params.nItems;
  facDim    = params.facDim;
  uReg      = params.uReg;
  iReg      = params.iReg;
  learnRate = params.learnRate;
  rhoRMS    = params.rhoRMS;
  maxIter   = params.maxIter;

  //init user latent factors
  uFac.resize(nUsers);
  for (i = 0; i < nUsers; i++) {
    uFac[i].resize(facDim, 0);
  }

  //init item latent factors
  iFac.resize(nItems);
  for (i = 0; i < nItems; i++) {
    iFac[i].resize(facDim, 0);
  }

}


