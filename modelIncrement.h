#ifndef _MODEL_INC_H_
#define _MODEL_INC_H_


#include <iostream>
#include <vector>
#include <set>
#include <cstdio>
#include <utility>
#include <unordered_set>
#include <string>
#include <omp.h>
#include <cmath>
#include <algorithm>
#include "io.h"
#include "util.h"
#include "model.h"
#include "svdFrmsvdlib.h"

#define DISP_ITER 50 
#define SAVE_ITER 100 
#define INC_ITER 5

class ModelIncrement : public Model {
  
  public:
    
    std::vector<int> currRankMapU; 
    std::vector<int> currRankMapI;

    ModelIncrement(const Params& params) : Model(params) { init(); }
    ModelIncrement(const Params& params, int seed) : Model(params, seed) { init(); }
    ModelIncrement(const Params& params, const char*uFacName, const char* iFacName, 
        int seed): Model(params, uFacName, iFacName, seed) { init(); }

    virtual void train(const Data& data, Model &bestModel, 
        std::unordered_set<int>& invalidUsers,
        std::unordered_set<int>& invalidItems) override;
    void init();
    virtual double estRating(int user, int item) override;
};


#endif



