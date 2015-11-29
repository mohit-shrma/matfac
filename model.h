#ifndef _MODEL_H_
#define _MODEL_H_

#include <iostream>
#include <numeric>
#include <cstdio>
#include <vector>
#include "util.h"
#include "const.h"
#include "GKlib.h"
#include "datastruct.h"

class Model { 
    
  private:
    //disable copy constructor 
    Model(const Model &obj);

  public:
    int nUsers;
    int nItems;
    int facDim;
    float learnRate;
    float rhoRMS;
    int maxIter;
    float uReg;
    float iReg;
    double **uFac; 
    double **iFac;

    //declare constructor
    Model(const Params& params);
    ~Model();
    

    //assignment operator
    Model& operator=(const Model &other); 


    //declare virtual method for train
    virtual void train(const Data& data, Model& bestModel) {
      std::cerr<< "\nTraining not in base class";
    };

    virtual double objective(const Data& data);
    bool isTerminateModel(Model& bestModel, const Data& data, int iter, 
        int& bestIter, double& bestObj, double& prevObj);
    double RMSE(gk_csr_t* mat);
    double fullLowRankErr(const Data& data);
    double subMatKnownRankErr(const Data& data, int uStart, int uEnd,
      int iStart, int iEnd);
    double fullRMSE(const Data& data);
};
#endif
