#ifndef _MODEL_DROPOUT_H_
#define _MODEL_DROPOUT_H_

class ModelDropoutMF : public Model {
  
  public:

    ModelDropoutMF(const Params& params) : Model(params) {}
    ModelDropoutMF(const Params& params, int seed) : Model(params, seed) {}
    ModelDropoutMF(const Params& params, const char*uFacName, const char* iFacName, 
        int seed):Model(params, uFacName, iFacName, seed) {}
    void trainSGDAdapPar(const Data &data, Model &bestModel, 
        std::unordered_set<int>& invalidUsers,
        std::unordered_set<int>& invalidItems,
        std::vector<int>& userRankMap, 
        std::vector<int>& itemRankMap,
        std::vector<int>& ranks);
    double estRating(int user, int item);
    bool isTerminateModel(Model& bestModel, const Data& data, int iter,
      int& bestIter, double& bestObj, double& prevObj, double& bestValRMSE,
      double& prevValRMSE, std::unordered_set<int>& invalidUsers, 
      std::unordered_set<int>& invalidItems);
};

#endif

