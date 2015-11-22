#ifndef _MODELMF_H_
#define _MODELMF_H_

#include <iostream>
#include <vector>
#include <cstdio>
#include "const.h"
#include "model.h"

class ModelMF : public Model {

  public:

    ModelMF(const Params& params) : Model(params) {}
    void train(const Data &data, ModelMF &bestModel);
    void computeUGrad(int user, int item, float r_ui, 
        std::vector<double> &uGrad);
    void computeIGrad(int user, int item, float r_ui, 
        std::vector<double> &iGrad);
    void updateFac(std::vector<double> &fac, std::vector<double> &grad,
        std::vector<double> &gradAcc);
};


#endif
