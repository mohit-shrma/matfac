#ifndef _MODELMF_H_
#define _MODELMF_H_

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

#define DISP_ITER 5 
#define SAVE_ITER 50 

class ModelMF : public Model {

  public:

    ModelMF(const Params& params) : Model(params) {}
    ModelMF(const Params& params, int seed) : Model(params, seed) {}
    ModelMF(const Params& params, const char*uFacName, const char* iFacName, 
        int seed):Model(params, uFacName, iFacName, seed) {}
    virtual void train(const Data& data, Model& bestModel,
        std::unordered_set<int>& invalidUsers,
        std::unordered_set<int>& invalidItems) ;
    void trainUShuffle(const Data& data, Model& bestModel,
        std::unordered_set<int>& invalidUsers,
        std::unordered_set<int>& invalidItems) ;
    virtual void trainALS(const Data& data, Model& bestModel,
        std::unordered_set<int>& invalidUsers,
        std::unordered_set<int>& invalidItems) ;
    void trainCCDPP(const Data& data, Model& bestModel,
        std::unordered_set<int>& invalidUsers,
        std::unordered_set<int>& invalidItems) ;
    void trainCCD(const Data& data, Model& bestModel,
        std::unordered_set<int>& invalidUsers,
        std::unordered_set<int>& invalidItems) ;
    void hogTrain(const Data& data, Model& bestModel,
        std::unordered_set<int>& invalidUsers,
        std::unordered_set<int>& invalidItems) ;
};


#endif
