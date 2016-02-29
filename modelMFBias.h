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
//#include "svd.h"
//#include "svdLapack.h"
#include "svdFrmsvdlib.h"

class ModelMFBias: public Model {

  public:
    ModelMFBias(const Params& params):Model(params) {}
    ModelMF(const Params& params, int seed) : Model(params, seed) {}
    ModelMF(const Params& params, const char*uFacName, const char* iFacName, 
        int seed):Model(params, uFacName, iFacName, seed) {}
    virtual void train(const Data& data, Model& bestModel,
        std::unordered_set<int>& invalidUsers,
        std::unordered_set<int>& invalidItems);
    virtual double objective(const Data& data);
    virtual double estRating(int user, int item);
};


#endif
