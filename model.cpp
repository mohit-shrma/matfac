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

  nUsers    = params.nUsers;
  nItems    = params.nItems;
  facDim    = params.facDim;
  uReg      = params.uReg;
  iReg      = params.iReg;
  learnRate = params.learnRate;
  rhoRMS    = params.rhoRMS;
  maxIter   = params.maxIter;

  //init user latent factors
  uFac.reserve(nUsers);
  for (auto uf: uFac) {
    uf.reserve(facDim);
    for (double& v: uf) {
      *v = (double)rand() / (double) (1.0 + RAND_MAX);
    }
  }

  //init item latent factors
  iFac.reserve(nItems);
  for (auto itemf: iFac) {
    itemf.reserve(facDim);
    for (double& v: itemf) {
      *v = (double)rand() / (double) (1.0 + RAND_MAX);
    }
  }

}


