#ifndef _MODELMF_H_
#define _MODELMF_H_

#include <iostream>
#include <vector>
#include <cstdio>
#include <memory>
#include "model.h"

class ModelMF : public Model {

  public:

    ModelMF(const Params& params) : Model(params) {}
    virtual void train(const Data& data, Model& bestModel) ;
    virtual void computeUGrad(int user, int item, float r_ui, 
        double *uGrad);
    virtual void computeIGrad(int user, int item, float r_ui, 
        double *iGrad);
    void updateAdaptiveFac(double *fac, double *grad,
        double *gradAcc);
    void updateFac(double *fac, double *grad);
    void gradCheck(int u, int item, float r_ui);
};


#endif
