#ifndef _MODELMF_BIAS_H_
#define _MODELMF_BIAS_H_

#include <iostream>
#include <vector>
#include <set>
#include <cstdio>
#include <utility>
#include <unordered_set>
#include <string>
#include "io.h"
#include "model.h"
#include "svdFrmsvdlib.h"

class ModelMFBias: public Model {

  public:
    
    ModelMFBias(const Params& params):Model(params) {}
    ModelMFBias(const Params& params, int seed) : Model(params, seed) {}
    ModelMFBias(const Params& params, const char*uFacName, const char* iFacName, 
        int seed):Model(params, uFacName, iFacName, seed) {}
    ModelMFBias(const Params& params, const char*uFacName, const char* iFacName, 
        const char* uBFName, const char *iBFName, const char* gBFName,
        int seed):Model (params, uFacName, iFacName, uBFName, iBFName, gBFName, 
          seed) {}
    
    void train(const Data& data, Model& bestModel,
        std::unordered_set<int>& invalidUsers,
        std::unordered_set<int>& invalidItems) override;
    double estRating(int user, int item) override;
    double objective(const Data& data) override;
    double objective(const Data& data, 
        std::unordered_set<int>& invalidUsers,
        std::unordered_set<int>& invalidItems) override;
};


#endif
