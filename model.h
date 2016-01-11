#ifndef _MODEL_H_
#define _MODEL_H_

#include <iostream>
#include <numeric>
#include <cstdio>
#include <vector>
#include <chrono>
#include "util.h"
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

    Model(int nUsers, int nItems, const Params& params);

    //declare virtual method for train
    virtual void train(const Data& data, Model& bestModel) {
      std::cerr<< "\nTraining not in base class";
    };

    //virtual method for training on a part of submatrix
    virtual void subTrain(const Data& data, Model& bestModel,
                        int uStart, int uEnd, int iStart, int iEnd) {
      std::cerr<< "\nsubTrain not in base class";
    };
    
    virtual void subExTrain(const Data &data, Model &bestModel,
         int uStart, int uEnd, int iStart, int iEnd) {
      std::cerr << "\nsubExTrain not in base class";
    };

    virtual double objective(const Data& data);
    double objectiveSubMat(const Data& data, int uStart, int uEnd,
      int iStart, int iEnd);
    double objectiveExSubMat(const Data& data, int uStart, int uEnd,
      int iStart, int iEnd);
    bool isTerminateModel(Model& bestModel, const Data& data, int iter, 
        int& bestIter, double& bestObj, double& prevObj);
    bool isTerminateModelSubMat(Model& bestModel, const Data& data, int iter,
      int& bestIter, double& bestObj, double& prevObj, int uStart, int uEnd,
      int iStart, int iEnd); 
    bool isTerminateModelExSubMat(Model& bestModel, const Data& data, int iter,
      int& bestIter, double& bestObj, double& prevObj, int uStart, int uEnd,
      int iStart, int iEnd); 
    double RMSE(gk_csr_t* mat);
    double subMatRMSE(gk_csr_t *mat, int uStart, int uEnd, int iStart, 
                      int iEnd);
    double subMatExRMSE(gk_csr_t *mat, int uStart, int uEnd, 
                      int iStart, int iEnd);
    double fullLowRankErr(const Data& data);
    double subMatKnownRankErr(const Data& data, int uStart, int uEnd,
      int iStart, int iEnd);
    double subMatKnownRankNonObsErr(const Data& data, int uStart, int uEnd,
      int iStart, int iEnd);
    double fullRMSE(const Data& data);
};
#endif
