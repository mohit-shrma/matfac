#ifndef _MODEL_DROPOUT_BIAS_H_
#define _MODEL_DROPOUT_BIAS_H_

#include <iostream>
#include <vector>
#include <set>
#include <cstdio>
#include <utility>
#include <unordered_set>
#include <string>
#include <omp.h>
#include "io.h"
#include "modelDropoutMF.h"
#include "svdFrmsvdlib.h"

#define DISP_ITER 50 
#define SAVE_ITER 100 

class ModelDropoutMFBias: public ModelDropoutMF {
  public:
    
    ModelDropoutMFBias(const Params& params, std::vector<int>& userRankMap,
      std::vector<int>& itemRankMap,
      std::vector<int>& ranks) : ModelDropoutMF(params, userRankMap, 
                                  itemRankMap, ranks) {}
    ModelDropoutMFBias(const Params& params, int seed, std::vector<int>& userRankMap,
      std::vector<int>& itemRankMap,
      std::vector<int>& ranks) : ModelDropoutMF(params, seed, userRankMap, 
                                  itemRankMap, ranks) {}
    ModelDropoutMFBias(const Params& params, const char*uFacName, const char* iFacName, 
        int seed, std::vector<int>& userRankMap,
        std::vector<int>& itemRankMap,
        std::vector<int>& ranks): ModelDropoutMF(params, uFacName, iFacName, seed,
                                  userRankMap, itemRankMap, ranks) {}
    ModelDropoutMFBias(const Params& params, int seed) : ModelDropoutMF(params, seed) {}
    void trainSGDProbPar(const Data &data, 
        ModelDropoutMF &bestModel, 
        std::unordered_set<int>& invalidUsers,
        std::unordered_set<int>& invalidItems) override;
    
    double estRating(int user, int item) override;
    double estRating(int user, int item, int minRank) override;
    double objective(const Data& data, std::unordered_set<int>& invalidUsers,
        std::unordered_set<int>& invalidItems, int minRank) override;
};


#endif

