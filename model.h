#ifndef _MODEL_H_
#define _MODEL_H_

#include <iostream>
#include <numeric>
#include <cstdio>
#include <vector>
#include "const.h"
#include "GKlib.h"
#include "datastruct.h"

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
    std::vector<std::vector<double>> uFac; 
    std::vector<std::vector<double>> iFac;

    //declare constructor
    Model(const Params& params);

    //declare virtual method for train
    void train(const Data& data, Model& bestModel) {
      std::cerr<< "\nTraining not in base class";
    };

    double objective(const Data& data);
    bool isTerminateModel(Model& bestModel, const Data& data, int iter, 
        int& bestIter, double& bestObj, double& prevObj);
    double RMSE(gk_csr_t* mat);
};
#endif
