#ifndef _MODEL_H_
#define _MODEL_H_

#include <iostream>
#include <numeric>
#include <cstdio>
#include <vector>
#include <set>
#include <chrono>
#include <string>
#include <random>
#include <tuple>
#include "util.h"
#include "const.h"
#include "GKlib.h"
#include "datastruct.h"

class Model { 

  public:
    int nUsers;
    int nItems;
    int facDim;
    int trainSeed;
    float learnRate;
    float rhoRMS;
    int maxIter;
    float uReg;
    float iReg;
    std::vector<std::vector<double>> uFac; 
    std::vector<std::vector<double>> iFac;
    std::vector<double> uBias;
    std::vector<double> iBias;
    double mu; //global bias

    //declare constructor
    Model(const Params& params);
    Model(int nUsers, int nItems, const Params& params);
    Model(const Params& params, int seed);
    Model(const Params& params, const char*uFacName, const char* iFacName, 
        int seed);
    Model(const Params& params, const char*uFacName, const char* iFacName, 
        const char* uBFName, const char *iBFName, const char* gBFName, int seed);

    //declare virtual method for train
    virtual void train(const Data& data, Model& bestModel,
        std::unordered_set<int>& invalidUsers,
        std::unordered_set<int>& invalidItems ) {
      std::cerr<< "\nTraining not in base class";
    };
    virtual void partialTrain(const Data& data, Model& bestModel,
        std::unordered_set<int>& invalidUsers,
        std::unordered_set<int>& invalidItems) {
      std::cerr<< "\nPartial Training not in base class" << std::endl;
    };

    //virtual method for training on a part of submatrix
    virtual void subTrain(const Data& data, Model& bestModel,
                        int uStart, int uEnd, int iStart, int iEnd) {
      std::cerr<< "\nsubTrain not in base class";
    };
    
    virtual void fixTrain(const Data& data, Model& bestModel,
                        int uStart, int uEnd, int iStart, int iEnd) {
      std::cerr<< "\nsubTrain not in base class";
    };
    virtual void subExTrain(const Data &data, Model &bestModel,
         int uStart, int uEnd, int iStart, int iEnd) {
      std::cerr << "\nsubExTrain not in base class";
    };

    std::vector<std::tuple<int, int, float>> getUIRatings(gk_csr_t* mat);
    virtual double objective(const Data& data);
    virtual double objective(const Data& data, 
        std::unordered_set<int>& invalidUsers,
        std::unordered_set<int>& invalidItems);
    double objective(const Data& data, std::unordered_set<int>& invalidUsers,
        std::unordered_set<int>& invalidItems, 
        std::vector<std::tuple<int, int, float>>& trainRatings);
        double objectiveSubMat(const Data& data, int uStart, int uEnd,
      int iStart, int iEnd);
    double objectiveExSubMat(const Data& data, int uStart, int uEnd,
      int iStart, int iEnd);
    bool isTerminateModel(Model& bestModel, const Data& data, int iter, 
        int& bestIter, double& bestObj, double& prevObj);
    bool isTerminateModel(Model& bestModel, const Data& data, int iter, 
        int& bestIter, double& bestObj, double& prevObj, 
        std::unordered_set<int>& invalidUsers, 
        std::unordered_set<int>& invalidItems);
    bool isTerminateModel(Model& bestModel, const Data& data, int iter,
      int& bestIter, double& bestObj, double& prevObj, double& bestValRMSE,
      double& prevValRMSE, std::unordered_set<int>& invalidUsers, 
      std::unordered_set<int>& invalidItems);
    bool isTerminateModel(Model& bestModel, const Data& data, int iter,
        int& bestIter, double& bestObj, double& prevObj, 
        std::unordered_set<int>& invalidUsers, std::unordered_set<int>& invalidItems,
        std::vector<std::tuple<int, int, float>>& trainRatings);
    bool isTerminateModelSubMat(Model& bestModel, const Data& data, int iter,
      int& bestIter, double& bestObj, double& prevObj, int uStart, int uEnd,
      int iStart, int iEnd); 
    bool isTerminateModelExSubMat(Model& bestModel, const Data& data, int iter,
      int& bestIter, double& bestObj, double& prevObj, int uStart, int uEnd,
      int iStart, int iEnd); 
    double RMSE(gk_csr_t* mat);
    double RMSE(gk_csr_t* mat, std::unordered_set<int>& invalidUsers,
      std::unordered_set<int>& invalidItems);
    double RMSE(std::vector<std::tuple<int, int, float>>& trainRatings);
    double RMSE(gk_csr_t *mat, std::unordered_set<int>& invalidUsers,
        std::unordered_set<int>& invalidItems, Model& origModel);
    double subMatRMSE(gk_csr_t *mat, int uStart, int uEnd, int iStart, 
                      int iEnd);
    double subMatExRMSE(gk_csr_t *mat, int uStart, int uEnd, 
                      int iStart, int iEnd);
    double fullLowRankErr(const Data& data);
    double subMatKnownRankErr(const Data& data, int uStart, int uEnd,
      int iStart, int iEnd);
    double subMatKnownRankNonObsErr(const Data& data, int uStart, int uEnd,
      int iStart, int iEnd);
    double subMatKnownRankNonObsErrWSet(const Data& data, int uStart, int uEnd,
      int iStart, int iEnd, std::set<int> exUSet, std::set<int> exISet);
    double fullRMSE(const Data& data);
    virtual double estRating(int user, int item);
    std::string modelSignature(); 
    void display();
    void save(std::string prefix);
    void saveFacs(std::string prefix);
    void load(std::string prefix);
    void loadFacs(std::string prefix);
    void load(const char* uFacName, const char *iFacName);
    void load(const char* uFacName, const char* iFacName, const char* uBFName,
      const char* iBFName, const char*gBFName);
    void updateFac(std::vector<double> &fac, std::vector<double> &grad);
    double estAvgRating(int user, std::unordered_set<int>& invalidItems) ;
    std::vector<std::tuple<int, int, float>> getUIRatings(gk_csr_t* mat, 
        std::unordered_set<int>& invalidUsers, 
        std::unordered_set<int>& invalidItems);
};
#endif
