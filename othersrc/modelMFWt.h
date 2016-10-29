#ifndef _MODELMFWT_H_
#define _MODELMFWT_H_

#include <iostream>
#include <vector>
#include <set>
#include <cstdio>
#include <utility>
#include <unordered_set>
#include <string>
#include "io.h"
#include "model.h"
//#include "svd.h"
//#include "svdLapack.h"
#include "svdFrmsvdlib.h"

class ModelMFWt : public Model {
  public:
    ModelMFWt(const Params& params) : Model(params) {}
    ModelMFWt(const Params& params, int seed) : Model(params, seed) {}
    ModelMFWt(const Params& params, const char*uFacName, const char* iFacName, 
        int seed):Model(params, uFacName, iFacName, seed) {}
    void hogTrain(const Data& data, Model& bestModel,
        std::unordered_set<int>& invalidUsers,
        std::unordered_set<int>& invalidItems) ;
    void hogUITrain(const Data& data, Model& bestModel,
        std::unordered_set<int>& invalidUsers,
        std::unordered_set<int>& invalidItems) ;
    virtual double objective(const Data& data, 
        std::unordered_set<int>& invalidUsers,
        std::unordered_set<int>& invalidItems);
};

#endif
