#ifndef _CONF_COMPUTE_H_
#define _CONF_COMPUTE_H_

#include "model.h"
#include <vector>
#include <algorithm>
#include <tuple>
#include <cmath>

#define PROGU 100

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
    std::vector<std::tuple<int, int, double>> matConfScores, Model& origModel,
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
#endif
