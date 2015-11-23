#ifndef _MODELMF_H_
#define _MODELMF_H_

#include <iostream>
#include <vector>
#include <cstdio>
#include "model.h"

class ModelMF : public Model {

  public:

    ModelMF(const Params& params) : Model(params) {}
    virtual void train(const Data& data, Model& bestModel) ;
    virtual void computeUGrad(int user, int item, float r_ui, 
        std::vector<double> &uGrad);
    virtual void computeIGrad(int user, int item, float r_ui, 
        std::vector<double> &iGrad);
    void updateFac(std::vector<double> &fac, std::vector<double> &grad,
        std::vector<double> &gradAcc);
    void gradCheck(int u, int item, float r_ui);
};


#endif
