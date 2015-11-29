#ifndef _MODELMF_WT_REG_ARB_H_
#define _MODELMF_WT_REG_ARB_H_  

/*
 * Learning with weighted trace norm under arbitary sampling distribution
 */


#include <vector>
#include <algorithm>
#include "modelMF.h" 

class ModelMFWtRegArb: public ModelMF {

  public:
    float alpha;
    double *uMarg;
    double *iMarg;

    void computeMarginals(const Data& data);
    
    void computeUGrad(int user, int item, float r_ui, 
        double *uGrad);
    void computeIGrad(int user, int item, float r_ui, 
        double *iGrad);
    double objective(const Data& data);

    void train(const Data& data, Model& bestModel);
  
    ModelMFWtRegArb(const Params& params): ModelMF(params), alpha(params.alpha) {
      uMarg = new double[nUsers];
      iMarg = new double[nItems];
    }

    ~ModelMFWtRegArb() {
      delete[] uMarg;
      delete[] iMarg;
    }
};

#endif

