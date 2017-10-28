#ifndef _MODELMFBPR_H_
#define _MODELMFBPR_H_

class ModelMFBPR : public Model {
  
  public:
    ModelMFBPR(const Params& params) : Model(params) {}
    ModelMFBPR(const Params& params, int seed) : Model(params, seed) {}
    ModelMFBPR(const Params& params, const char*uFacName, const char* iFacName, 
        int seed):Model(params, uFacName, iFacName, seed) {}
    void train(const Data& data, Model& bestModel,
        std::unordered_set<int>& invalidUsers,
        std::unordered_set<int>& invalidItems) ;
};


#endif
