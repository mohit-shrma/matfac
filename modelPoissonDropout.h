#ifndef _MODEL_POISSON_DROPOUT_H_
#define _MODEL_POISSON_DROPOUT_H_

#include <iostream>
#include <vector>
#include <set>
#include <cstdio>
#include <utility>
#include <unordered_set>
#include <string>
#include <omp.h>
#include <cmath>
#include "io.h"
#include "model.h"
#include "svdFrmsvdlib.h"

#define DISP_ITER 50 
#define SAVE_ITER 100 

class ModelPoissonDropout : public Model {

  public:
    
    std::vector<double> userRankMap; 
    std::vector<double> itemRankMap;
    std::vector<double> userFreq;
    std::vector<double> itemFreq;
    std::vector<double> factorial;
    std::vector<double> fDimWt;

    ModelPoissonDropout(const Params& params, std::vector<double>& userRankMap,
      std::vector<double>& itemRankMap, std::vector<double>& userFreq, 
      std::vector<double>& itemFreq) : Model(params), 
                                 userRankMap(userRankMap), itemRankMap(itemRankMap), 
                                 userFreq(userFreq), itemFreq(itemFreq) {
                                    //intialize factorial table
                                    factorial.push_back(1);
                                    for (int i = 1; i <= params.facDim+1; i++) {
                                      factorial.push_back(factorial.back()*((double)i));
                                    }
                                    fDimWt = std::vector<double>(params.facDim, 0);
                                 }
    ModelPoissonDropout(const Params& params, int seed, std::vector<double>& userRankMap,
      std::vector<double>& itemRankMap, std::vector<double>& userFreq, 
      std::vector<double>& itemFreq) : Model(params, seed),
                                 userRankMap(userRankMap), itemRankMap(itemRankMap), 
                                 userFreq(userFreq), itemFreq(itemFreq) {
                                    //intialize factorial table
                                    factorial.push_back(1);
                                    for (int i = 1; i <= params.facDim+1; i++) {
                                      factorial.push_back(factorial.back()*((double)i));
                                    }
                                    fDimWt = std::vector<double>(params.facDim, 0);
                                 }
    ModelPoissonDropout(const Params& params, const char*uFacName, const char* iFacName, 
        int seed, std::vector<double>& userRankMap,
        std::vector<double>& itemRankMap, std::vector<double>& userFreq, 
      std::vector<double>& itemFreq): Model(params, uFacName, iFacName, seed),
                                 userRankMap(userRankMap), itemRankMap(itemRankMap), 
                                 userFreq(userFreq), itemFreq(itemFreq) {
                                    //intialize factorial table
                                    factorial.push_back(1);
                                    for (int i = 1; i <= params.facDim+1; i++) {
                                      factorial.push_back(factorial.back()*((double)i));
                                    }
                                    fDimWt = std::vector<double>(params.facDim, 0);
                                 }
    ModelPoissonDropout(const Params& params, int seed) : Model(params, seed) {
                                    //intialize factorial table
                                    factorial.push_back(1);
                                    for (int i = 1; i <= params.facDim+1; i++) {
                                      factorial.push_back(factorial.back()*((double)i));
                                    }
                                    fDimWt = std::vector<double>(params.facDim, 0);
    }

    virtual void train(const Data& data, Model &bestModel, 
        std::unordered_set<int>& invalidUsers,
        std::unordered_set<int>& invalidItems) override;
    void trainSigmoid(const Data& data, Model &bestModel, 
        std::unordered_set<int>& invalidUsers,
        std::unordered_set<int>& invalidItems);
    virtual double estRating(int user, int item) override;
};

#endif

