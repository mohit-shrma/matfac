#ifndef _CONF_COMPUTE_H_
#define _CONF_COMPUTE_H_

#include "model.h"
#include <vector>
#include <algorithm>
#include <tuple>
#include <cmath>

double confScore(int user, int item, std::vector<Model>& models);
std::vector<double> confBucketRMSEs(Model& origModel, Model& fullModel,
    std::vector<Model>& models,
    int nUsers, int nItems, int nBuckets);
std::vector<double> pprBucketRMSEs(Model& origModel, Model& fullModel, int nUsers, 
    int nItems, float lambda, int max_niter, gk_csr_t *graphMat, int nBuckets);
std::vector<double> gprBucketRMSEs(Model& origModel, Model& fullModel, int nUsers, 
    int nItems, float lambda, int max_niter, gk_csr_t *graphMat, int nBuckets);
std::vector<double> pprBucketRMSEsFrmPR(Model& origModel, Model& fullModel, int nUsers, 
    int nItems, gk_csr_t *graphMat, int nBuckets, const char* prFName);
std::vector<double> confOptBucketRMSEs(Model& origModel, Model& fullModel,
    int nUsers, int nItems, int nBuckets);

#endif
