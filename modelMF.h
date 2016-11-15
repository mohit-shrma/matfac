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
//#include "svd.h"
//#include "svdLapack.h"
#include "svdFrmsvdlib.h"


class ModelMF : public Model {

  public:

    ModelMF(const Params& params) : Model(params) {}
    ModelMF(const Params& params, int seed) : Model(params, seed) {}
    ModelMF(const Params& params, const char*uFacName, const char* iFacName, 
        int seed):Model(params, uFacName, iFacName, seed) {}
    virtual void train(const Data& data, Model& bestModel,
        std::unordered_set<int>& invalidUsers,
        std::unordered_set<int>& invalidItems) ;
    virtual void trainALS(const Data& data, Model& bestModel,
        std::unordered_set<int>& invalidUsers,
        std::unordered_set<int>& invalidItems) ;
    void trainCCDPP(const Data& data, Model& bestModel,
        std::unordered_set<int>& invalidUsers,
        std::unordered_set<int>& invalidItems) ;
    void hogTrain(const Data& data, Model& bestModel,
        std::unordered_set<int>& invalidUsers,
        std::unordered_set<int>& invalidItems) ;
    void hogAdapTrain(const Data& data, Model& bestModel,
        std::unordered_set<int>& invalidUsers,
        std::unordered_set<int>& invalidItems) ;
    void uniTrain(const Data& data, Model& bestModel,
        std::unordered_set<int>& invalidUsers,
        std::unordered_set<int>& invalidItems) ;
    virtual void partialTrain(const Data& data, Model& bestModel, 
        std::unordered_set<int>& invalidUsers,
        std::unordered_set<int>& invalidItems) ;
    virtual void subTrain(const Data& data, Model& bestModel,
                        int uStart, int uEnd, int iStart, int iEnd);
    virtual void fixTrain(const Data& data, Model& bestModel,
                        int uStart, int uEnd, int iStart, int iEnd);
    virtual void subExTrain(const Data& data, Model& bestModel,
                        int uStart, int uEnd, int iStart, int iEnd);
    virtual void computeUGrad(int user, int item, float r_ui, 
        std::vector<double> &uGrad);
    virtual void computeIGrad(int user, int item, float r_ui, 
        std::vector<double> &iGrad);
    void updateAdaptiveFac(std::vector<double> &fac, std::vector<double> &grad,
        std::vector<double> &gradAcc);
    void gradCheck(int u, int item, float r_ui);
};


#endif
