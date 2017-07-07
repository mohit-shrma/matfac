#ifndef _MODEL_DROPOUT_H_
#define _MODEL_DROPOUT_H_

class ModelDropoutMF : public Model {

  public:
    
    std::vector<int> userRankMap; 
    std::vector<int> itemRankMap;
    std::vector<int> ranks;
    
    ModelDropoutMF(
      std::vector<int> userRankMap,
      std::vector<int> itemRankMap,
      std::vector<int> ranks):userRankMap(userRankMap), 
                              itemRankMap(itemRankMap), ranks(ranks) {}
    ModelDropoutMF(const Params& params, std::vector<int> userRankMap,
      std::vector<int> itemRankMap,
      std::vector<int> ranks) : Model(params), 
                                ModelDropoutMF(userRankMap, itemRankMap, ranks) {}
    ModelDropoutMF(const Params& params, int seed, std::vector<int> userRankMap,
      std::vector<int> itemRankMap,
      std::vector<int> ranks) : Model(params, seed), 
                                ModelDropoutMF(userRankMap, itemRankMap, ranks) {}
    ModelDropoutMF(const Params& params, const char*uFacName, const char* iFacName, 
        int seed, std::vector<int> userRankMap,
        std::vector<int> itemRankMap,
        std::vector<int> ranks):Model(params, uFacName, iFacName, seed),
                                ModelDropoutMF(userRankMap, itemRankMap, ranks)  {}
    void trainSGDAdapPar(const Data &data, Model &bestModel, 
        std::unordered_set<int>& invalidUsers,
        std::unordered_set<int>& invalidItems,
        std::vector<int>& userRankMap, 
        std::vector<int>& itemRankMap,
        std::vector<int>& ranks);
    double estRating(int user, int item);
    double estRating(int user, int item, int minRank);
    bool isTerminateModel(Model& bestModel, const Data& data, int iter,
      int& bestIter, double& bestObj, double& prevObj, double& bestValRMSE,
      double& prevValRMSE, std::unordered_set<int>& invalidUsers, 
      std::unordered_set<int>& invalidItems, int minRank);
    double objective(const Data& data, std::unordered_set<int>& invalidUsers,
        std::unordered_set<int>& invalidItems, int minRank);
    double RMSE(gk_csr_t *mat, std::unordered_set<int>& invalidUsers,
        std::unordered_set<int>& invalidItems, int minRank);
};

#endif

