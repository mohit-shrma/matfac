#ifndef _CONF_COMPUTE_H_
#define _CONF_COMPUTE_H_

#include "model.h"
#include "const.h"
#include <vector>
#include <algorithm>
#include <tuple>
#include <map>
#include <cmath>
#include <cstdlib>
#include <fstream>

void comparePPR2GPR(int nUsers, int nItems, gk_csr_t* graphMat, float lambda,
    int max_niter, const char* prFName, const char* opFName);
double confScore(int user, int item, std::vector<Model>& models);
std::vector<double> confBucketRMSEs(Model& origModel, Model& fullModel,
    std::vector<Model>& models,
    int nUsers, int nItems, int nBuckets);
std::vector<double> confBucketRMSEsWInval(Model& origModel, Model& fullModel,
    std::vector<Model>& models, int nUsers, int nItems, int nBuckets,
    std::unordered_set<int>& invalUsers, std::unordered_set<int>& invalItems);
std::vector<double> pprBucketRMSEs(Model& origModel, Model& fullModel, int nUsers, 
    int nItems, float lambda, int max_niter, gk_csr_t *graphMat, int nBuckets);
std::vector<double> gprBucketRMSEs(Model& origModel, Model& fullModel, int nUsers, 
    int nItems, float lambda, int max_niter, gk_csr_t *graphMat, int nBuckets);
std::vector<double> gprBucketRMSEsWInVal(Model& origModel, Model& fullModel, int nUsers, 
    int nItems, float lambda, int max_niter, gk_csr_t *graphMat, int nBuckets,
    std::unordered_set<int>& invalUsers, std::unordered_set<int>& invalItems);
std::vector<double> pprBucketRMSEsFrmPR(Model& origModel, Model& fullModel, int nUsers, 
    int nItems, gk_csr_t *graphMat, int nBuckets, const char* prFName);
std::vector<double> pprBucketRMSEsFrmPRWInVal(Model& origModel, Model& fullModel, int nUsers, 
    int nItems, gk_csr_t *graphMat, int nBuckets, const char* prFName,
    std::unordered_set<int>& invalUsers, std::unordered_set<int>& invalItems);
std::vector<double> pprBucketRMSEsWInVal(Model& origModel, Model& fullModel, int nUsers, 
    int nItems, float lambda, int max_niter, gk_csr_t *graphMat, int nBuckets, 
    std::unordered_set<int> invalUsers, std::unordered_set<int> invalItems);
std::vector<double> confOptBucketRMSEs(Model& origModel, Model& fullModel,
    int nUsers, int nItems, int nBuckets);
std::vector<double> confOptBucketRMSEsWInVal(Model& origModel, Model& fullModel,
    int nUsers, int nItems, int nBuckets, std::unordered_set<int>& invalUsers,
    std::unordered_set<int>& invalItems) ;
std::vector<double> confBucketRMSEsWInvalOpPerUser(Model& origModel, Model& fullModel,
    std::vector<Model>& models, int nUsers, int nItems, int nBuckets,
    std::unordered_set<int>& invalUsers, std::unordered_set<int>& invalItems,
    std::string opFileName);
std::vector<double> genConfidenceCurve(
    std::vector<std::tuple<int, int, double>>& matConfScores, Model& origModel,
    Model& fullModel, int nBuckets, float alpha);
std::vector<double> computeModConf(gk_csr_t* mat, 
    std::vector<Model>& models, std::unordered_set<int>& invalUsers,
    std::unordered_set<int>& invalItems, Model& origModel,
    Model& fullModel, int nBuckets, float alpha);
std::vector<double> computeGPRConf(gk_csr_t* mat, 
    gk_csr_t* graphMat, std::unordered_set<int>& invalUsers,
    std::unordered_set<int>& invalItems, float lambda, int max_niter, Model& origModel,
    Model& fullModel, int nBuckets, float alpha);
std::vector<double> computePPRConf(gk_csr_t* mat, 
    gk_csr_t* graphMat, std::unordered_set<int>& invalUsers,
    std::unordered_set<int>& invalItems, float lambda, int max_niter, Model& origModel,
    Model& fullModel, int nBuckets, float alpha);
std::vector<double> computeMissingGPRConf(gk_csr_t* trainMat, 
    gk_csr_t* graphMat, std::unordered_set<int>& invalUsers,
    std::unordered_set<int>& invalItems, float lambda, int max_niter, Model& origModel,
    Model& fullModel, int nBuckets, float alpha);
std::vector<double> computeMissingModConf(gk_csr_t* trainMat, 
    std::vector<Model>& models, std::unordered_set<int>& invalUsers,
    std::unordered_set<int>& invalItems, Model& origModel,
    Model& fullModel, int nBuckets, float alpha);
std::vector<double> computeMissingPPRConf(gk_csr_t* trainMat, 
    gk_csr_t* graphMat, std::unordered_set<int>& invalUsers,
    std::unordered_set<int>& invalItems, float lambda, int max_niter, Model& origModel,
    Model& fullModel, int nBuckets, float alpha);
std::vector<double> computeMissingPPRConfExt(gk_csr_t* trainMat, 
    gk_csr_t* graphMat, std::unordered_set<int>& invalUsers,
    std::unordered_set<int>& invalItems, float lambda, int max_niter, Model& origModel,
    Model& fullModel, int nBuckets, float alpha, const char* prFName);

std::vector<double> computeMissingModConfSamp(std::vector<Model>& models,
    Model& origModel, Model& fullModel, int nBuckets, float alpha, 
    std::vector<std::pair<int,int>> testPairs);
std::vector<double> computeMissingGPRConfSamp(gk_csr_t* graphMat, float lambda, 
    int max_niter, Model& origModel, Model& fullModel, int nBuckets, 
    float alpha, std::vector<std::pair<int,int>> testPairs, int nUsers);
std::vector<double> computeMissingPPRConfExtSamp(gk_csr_t* trainMat, 
    gk_csr_t* graphMat, float lambda, int max_niter, Model& origModel,
    Model& fullModel, int nBuckets, float alpha, const char* prFName, 
    std::vector<std::pair<int, int>> testPairs);

std::vector<std::pair<int, int>> getTestPairs(gk_csr_t* mat, 
    std::unordered_set<int>& invalUsers, std::unordered_set<int>& invalItems,
    int testSize, int seed);

std::vector<double> genOptConfidenceCurve(
    std::vector<std::pair<int, int>>& testPairs, Model& origModel,
    Model& fullModel, int nBuckets, float alpha);

std::vector<double> genItemConfCurve(std::vector<std::pair<int, int>>& testPairs, 
    Model& origModel, Model& fullModel, int nBuckets, float alpha, 
    std::vector<double>& itemFreq);
std::vector<double> genUserConfCurve(std::vector<std::pair<int, int>>& testPairs, 
    Model& origModel, Model& fullModel, int nBuckets, float alpha, 
    std::vector<double>& userFreq);

std::vector<double> genRMSECurve(
    std::vector<std::pair<double, double>>& confActPredDiffs,
    int nBuckets);
std::vector<double> genUserConfRMSECurve(std::vector<std::pair<int, int>>& testPairs, 
    Model& origModel, Model& fullModel, int nBuckets,  
    std::vector<double>& userFreq);
std::vector<double> genOptConfRMSECurve(
    std::vector<std::pair<int, int>>& testPairs, Model& origModel,
    Model& fullModel, int nBuckets);

std::vector<double> genGPRConfRMSECurve(std::vector<std::pair<int, int>>& testPairs, 
    Model& origModel, Model& fullModel, gk_csr_t* graphMat, float lambda,
    int max_niter, int nBuckets);

std::vector<double> genModelConfRMSECurve(std::vector<std::pair<int, int>>& testPairs, 
    Model& origModel, Model& fullModel, std::vector<Model>& models,
    int nBuckets);

std::vector<double> genItemConfRMSECurve(std::vector<std::pair<int, int>>& testPairs, 
    Model& origModel, Model& fullModel, int nBuckets,  
    std::vector<double>& itemFreq);
std::vector<double> genPPRConfRMSECurve(std::vector<std::pair<int, int>>& testPairs, 
    Model& origModel, Model& fullModel, gk_csr_t* graphMat, float lambda,
    int max_niter, const char* prFName, int nBuckets);

std::vector<double> genPPRConfRMSECurve(std::vector<std::pair<int, int>>& testPairs, 
    Model& origModel, Model& fullModel, gk_csr_t* graphMat, float lambda,
    int max_niter, const char* prFName, int nBuckets);


std::vector<double> itemFreqBucketRMSEsWInVal(Model& origModel, 
    Model& fullModel, int nUsers, int nItems, 
    std::vector<double>& itemFreq, int nBuckets, 
    std::unordered_set<int>& invalUsers, std::unordered_set<int>& invalItems);

std::vector<double> gprSampBucketRMSEsWInVal(Model& fullModel, gk_csr_t *mat,
    float lambda, int max_niter, gk_csr_t *graphMat, int nBuckets,
    std::unordered_set<int>& invalUsers, std::unordered_set<int>& invalItems, 
    int nSampUsers, int seed);
std::vector<double> itemFreqSampBucketRMSEsWInVal(gk_csr_t* mat, 
    Model& fullModel, 
    std::vector<double>& itemFreq, int nBuckets, 
    std::unordered_set<int>& invalUsers, 
    std::unordered_set<int>& invalItems, int nSampUsers, int seed);
std::vector<double> pprSampBucketRMSEsWInVal(Model& fullModel, gk_csr_t *mat, 
    float lambda, int max_niter, gk_csr_t *graphMat, int nBuckets, 
    std::unordered_set<int>& invalUsers, std::unordered_set<int>& invalItems, 
    int nSampUsers, int seed);

std::vector<double> gprSampBucketRMSEsWInVal(Model& fullModel, gk_csr_t *mat,
    float lambda, int max_niter, gk_csr_t *graphMat, int nBuckets,
    std::unordered_set<int>& invalUsers, std::unordered_set<int>& invalItems, 
     std::unordered_set<int>& filtItems, int nSampUsers, int seed);
std::vector<double> itemFreqSampBucketRMSEsWInVal(gk_csr_t* mat, 
    Model& fullModel, 
    std::vector<double>& itemFreq, int nBuckets, 
    std::unordered_set<int>& invalUsers, 
    std::unordered_set<int>& invalItems, std::unordered_set<int>& filtItems,
    int nSampUsers, int seed);

std::vector<double> pprSampBucketRMSEsWInVal(Model& fullModel, gk_csr_t *mat, 
    float lambda, int max_niter, gk_csr_t *graphMat, int nBuckets, 
    std::unordered_set<int>& invalUsers, std::unordered_set<int>& invalItems, 
    std::unordered_set<int>& filtItems, 
    int nSampUsers, int seed);


std::vector<double> itemFreqSampBucketRMSEsWInVal(Model& origModel, 
    Model& fullModel, int nUsers, int nItems, 
    std::vector<double>& itemFreq, int nBuckets, 
    std::unordered_set<int>& invalUsers, 
    std::unordered_set<int>& invalItems, int nSampUsers, int seed);
std::vector<double> gprSampBucketRMSEsWInVal(Model& origModel, 
    Model& fullModel, 
    int nUsers, int nItems,
    float lambda, int max_niter, gk_csr_t *graphMat, int nBuckets,
    std::unordered_set<int>& invalUsers, std::unordered_set<int>& invalItems, 
    int nSampUsers, int seed);
std::vector<double> pprSampBucketRMSEsWInVal(Model &origModel,
    Model& fullModel, int nUsers, int nItems, 
    float lambda, int max_niter, gk_csr_t *graphMat, int nBuckets, 
    std::unordered_set<int>& invalUsers, std::unordered_set<int>& invalItems, 
    int nSampUsers, int seed);

#endif


