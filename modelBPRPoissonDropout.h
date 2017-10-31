#ifndef _MODEL_BPR_POISSON_DROPOUT_H_
#define _MODEL_BPR_POISSON_DROPOUT_H_

#include "modelMFBPR.h"

class ModelBPRPoissonDropout : public ModelMFBPR {
 
  public:
        
    std::vector<double> userRankMap; 
    std::vector<double> itemRankMap;
    std::vector<double> userFreq;
    std::vector<double> itemFreq;
    std::vector<double> factorial;
    std::vector<double> fDimWt;
    std::vector<int> cdfRanks;
    double minFreq;
    double maxFreq;
    double meanFreq;
    double stdFreq;


    ModelBPRPoissonDropout(const Params& params, std::vector<double>& userRankMap,
      std::vector<double>& itemRankMap, std::vector<double>& userFreq, 
      std::vector<double>& itemFreq) : ModelMFBPR(params), 
                                 userRankMap(userRankMap), itemRankMap(itemRankMap), 
                                 userFreq(userFreq), itemFreq(itemFreq) {
                                    //intialize factorial table
                                    factorial.push_back(1);
                                    for (int i = 1; i <= params.facDim+1; i++) {
                                      factorial.push_back(factorial.back()*((double)i));
                                    }
                                    fDimWt = std::vector<double>(params.facDim, 0);
                                    minFreq = minVec(userFreq);
                                    maxFreq = maxVec(userFreq);
                                    double temp = minVec(itemFreq);
                                    if (minFreq > temp) {
                                      minFreq = temp;
                                    }
                                    temp = maxVec(itemFreq);
                                    if (maxFreq < temp) {
                                      maxFreq = temp;
                                    }
                                    
                                    std::vector<double> concatVec(userFreq.begin(), userFreq.end());
                                    concatVec.insert(concatVec.end(), itemFreq.begin(), itemFreq.end());
                                    auto meanStd = meanStdDev(concatVec);
                                    meanFreq = meanStd.first;
                                    stdFreq = meanStd.second;
                                    initCDFRanks();
                                 }

    ModelBPRPoissonDropout(const Params& params, int seed, std::vector<double>& userRankMap,
      std::vector<double>& itemRankMap, std::vector<double>& userFreq, 
      std::vector<double>& itemFreq) : ModelMFBPR(params, seed),
                                 userRankMap(userRankMap), itemRankMap(itemRankMap), 
                                 userFreq(userFreq), itemFreq(itemFreq) {
                                    //intialize factorial table
                                    factorial.push_back(1);
                                    for (int i = 1; i <= params.facDim+1; i++) {
                                      factorial.push_back(factorial.back()*((double)i));
                                    }
                                    fDimWt = std::vector<double>(params.facDim, 0);
                                    minFreq = minVec(userFreq);
                                    maxFreq = maxVec(userFreq);
                                    double temp = minVec(itemFreq);
                                    if (minFreq > temp) {
                                      minFreq = temp;
                                    }
                                    temp = maxVec(itemFreq);
                                    if (maxFreq < temp) {
                                      maxFreq = temp;
                                    }
                                    std::vector<double> concatVec(userFreq.begin(), userFreq.end());
                                    concatVec.insert(concatVec.end(), itemFreq.begin(), itemFreq.end());
                                    auto meanStd = meanStdDev(concatVec);
                                    meanFreq = meanStd.first;
                                    stdFreq = meanStd.second;
                                    initCDFRanks();
                                 }
    
    ModelBPRPoissonDropout(const Params& params, const char*uFacName, const char* iFacName, 
        int seed, std::vector<double>& userRankMap,
        std::vector<double>& itemRankMap, std::vector<double>& userFreq, 
      std::vector<double>& itemFreq): ModelMFBPR(params, uFacName, iFacName, seed),
                                 userRankMap(userRankMap), itemRankMap(itemRankMap), 
                                 userFreq(userFreq), itemFreq(itemFreq) {
                                    //intialize factorial table
                                    factorial.push_back(1);
                                    for (int i = 1; i <= params.facDim+1; i++) {
                                      factorial.push_back(factorial.back()*((double)i));
                                    }
                                    fDimWt = std::vector<double>(params.facDim, 0);
                                    minFreq = minVec(userFreq);
                                    maxFreq = maxVec(userFreq);
                                    double temp = minVec(itemFreq);
                                    if (minFreq > temp) {
                                      minFreq = temp;
                                    }
                                    temp = maxVec(itemFreq);
                                    if (maxFreq < temp) {
                                      maxFreq = temp;
                                    }
                                    std::vector<double> concatVec(userFreq.begin(), userFreq.end());
                                    concatVec.insert(concatVec.end(), itemFreq.begin(), itemFreq.end());
                                    auto meanStd = meanStdDev(concatVec);
                                    meanFreq = meanStd.first;
                                    stdFreq = meanStd.second;
                                    initCDFRanks();
                                 }
    
    ModelBPRPoissonDropout(const Params& params, int seed) : ModelMFBPR(params, seed) {
                                    //intialize factorial table
                                    factorial.push_back(1);
                                    for (int i = 1; i <= params.facDim+1; i++) {
                                      factorial.push_back(factorial.back()*((double)i));
                                    }
                                    fDimWt = std::vector<double>(params.facDim, 0);
                                    minFreq = minVec(userFreq);
                                    maxFreq = maxVec(userFreq);
                                    double temp = minVec(itemFreq);
                                    if (minFreq > temp) {
                                      minFreq = temp;
                                    }
                                    temp = maxVec(itemFreq);
                                    if (maxFreq < temp) {
                                      maxFreq = temp;
                                    }
                                    std::vector<double> concatVec(userFreq.begin(), userFreq.end());
                                    concatVec.insert(concatVec.end(), itemFreq.begin(), itemFreq.end());
                                    auto meanStd = meanStdDev(concatVec);
                                    meanFreq = meanStd.first;
                                    stdFreq = meanStd.second;
                                    initCDFRanks();
                                    
    }

    virtual void train(const Data& data, Model &bestModel, 
        std::unordered_set<int>& invalidUsers,
        std::unordered_set<int>& invalidItems) override;
    void trainSigmoid(const Data& data, Model &bestModel, 
        std::unordered_set<int>& invalidUsers,
        std::unordered_set<int>& invalidItems);
    virtual double estRating(int user, int item) override;
    void initCDFRanks();
   
};


#endif

