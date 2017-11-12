#ifndef _MODEL_INV_POP_MF_H_
#define _MODEL_INV_POP_MF_H_

#include <iostream>
#include <vector>
#include <set>
#include <cstdio>
#include <utility>
#include <unordered_set>
#include <string>
#include <map>
#include <omp.h>
#include "io.h"
#include "model.h"
#include "svdFrmsvdlib.h"

class ModelInvPopMF: public Model {

  public:

    std::map<int, double> invPopU;
    std::map<int, double> invPopI;
    std::vector<double> userFreq;
    std::vector<double> itemFreq;
    int nTrainUsers;
    int nTrainItems;

    ModelInvPopMF(int nUsers, int nItems, int facDim, 
        std::vector<double>& userFreq, std::vector<double>& itemFreq) : Model(
          nUsers, nItems, facDim), userFreq(userFreq), itemFreq(itemFreq) {}
    ModelInvPopMF(const Params& params, std::vector<double>& userFreq, 
        std::vector<double>& itemFreq) : Model(params), 
    userFreq(userFreq), itemFreq(itemFreq) {}
    ModelInvPopMF(const Params& params, int seed, std::vector<double>& userFreq,
        std::vector<double>& itemFreq) : Model(params, seed), 
        userFreq(userFreq), itemFreq(itemFreq) {}
    ModelInvPopMF(const Params& params, const char*uFacName, const char* iFacName, 
        int seed, std::vector<double>& userFreq, 
        std::vector<double>& itemFreq): Model(params, uFacName, iFacName, seed),
        userFreq(userFreq), itemFreq(itemFreq) {}
  
    void train(const Data& data, Model& bestModel,
        std::unordered_set<int>& invalidUsers,
        std::unordered_set<int>& invalidItems) override;
    void trainSGDPar(const Data &data, Model &bestModel, 
        std::unordered_set<int>& invalidUsers,
        std::unordered_set<int>& invalidItems);
    virtual double objective(const Data& data, 
        std::unordered_set<int>& invalidUsers,
        std::unordered_set<int>& invalidItems) override;
};


#endif


