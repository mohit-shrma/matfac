#ifndef _MODELMF_WT_REG_H_
#define _MODELMF_WT_REG_H_

#include <vector>
#include "modelMF.h"

class ModelMFWtReg : public ModelMF {

  public:
    float alpha;
    std::vector<double> uMarg;
    std::vector<double> iMarg;

    void computeMarginals(const Data& data);
    
    void computeUGrad(int user, int item, float r_ui, 
        std::vector<double> &uGrad);
    void computeIGrad(int user, int item, float r_ui, 
        std::vector<double> &iGrad);
    double objective(const Data& data);

    void train(const Data& data, Model& bestModel);
    ModelMFWtReg(const Params& params): ModelMF(params), alpha(params.alpha) {
      uMarg.assign(nUsers, 0);
      iMarg.assign(nItems, 0);
    }
};


#endif
