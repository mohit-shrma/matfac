#ifndef _MODEL_DROPOUT_H_
#define _MODEL_DROPOUT_H_

#include <iostream>
#include <vector>
#include <set>
#include <cstdio>
#include <utility>
#include <unordered_set>
#include <string>
#include <omp.h>
#include "io.h"
#include "model.h"
#include "svdFrmsvdlib.h"

#define DISP_ITER 50 
#define SAVE_ITER 100 

class ModelDropoutMF : public Model {

  public:
    
    std::vector<int> userRankMap; 
    std::vector<int> itemRankMap;
    std::vector<int> ranks;
    
    ModelDropoutMF(const Params& params, std::vector<int>& userRankMap,
      std::vector<int>& itemRankMap,
      std::vector<int>& ranks) : Model(params), 
                                 userRankMap(userRankMap), 
                                 itemRankMap(itemRankMap), ranks(ranks) {}
    ModelDropoutMF(const Params& params, int seed, std::vector<int>& userRankMap,
      std::vector<int>& itemRankMap,
      std::vector<int>& ranks) : Model(params, seed),
                                 userRankMap(userRankMap), 
                                 itemRankMap(itemRankMap), ranks(ranks) {}
    ModelDropoutMF(const Params& params, const char*uFacName, const char* iFacName, 
        int seed, std::vector<int>& userRankMap,
        std::vector<int>& itemRankMap,
        std::vector<int>& ranks): Model(params, uFacName, iFacName, seed),
                                  userRankMap(userRankMap), 
                                  itemRankMap(itemRankMap), ranks(ranks) {}
    ModelDropoutMF(const Params& params, int seed) : Model(params, seed) {}
    void trainSGDAdapPar(const Data& data, ModelDropoutMF &bestModel, 
        std::unordered_set<int>& invalidUsers,
        std::unordered_set<int>& invalidItems);
    virtual void trainSGDProbPar(const Data &data, 
        ModelDropoutMF &bestModel, 
        std::unordered_set<int>& invalidUsers,
        std::unordered_set<int>& invalidItems);
    void trainSGDProbParBPR(const Data &data, 
        ModelDropoutMF &bestModel, 
        std::unordered_set<int>& invalidUsers,
        std::unordered_set<int>& invalidItems);
    void trainSGDProbOrderedPar(const Data &data, 
        ModelDropoutMF &bestModel, 
        std::unordered_set<int>& invalidUsers,
        std::unordered_set<int>& invalidItems);
    void trainSGDOnlyOrderedPar(const Data &data, 
        ModelDropoutMF &bestModel, 
        std::unordered_set<int>& invalidUsers,
        std::unordered_set<int>& invalidItems);
    void trainSGDProbOrderedParBPR(const Data &data, 
        ModelDropoutMF &bestModel, 
        std::unordered_set<int>& invalidUsers,
        std::unordered_set<int>& invalidItems);
    virtual double estRating(int user, int item) override;
    virtual double estRating(int user, int item, int minRank);
    bool isTerminateModel(ModelDropoutMF& bestModel, const Data& data, int iter,
      int& bestIter, double& bestObj, double& prevObj, double& bestValRMSE,
      double& prevValRMSE, std::unordered_set<int>& invalidUsers, 
      std::unordered_set<int>& invalidItems, int minRank);
    virtual double objective(const Data& data, std::unordered_set<int>& invalidUsers,
        std::unordered_set<int>& invalidItems, int minRank);
    double RMSE(gk_csr_t *mat, std::unordered_set<int>& invalidUsers,
        std::unordered_set<int>& invalidItems, int minRank);
};

#endif

