#ifndef _TOP_BUCKET_COMP_H_
#define _TOP_BUCKET_COMP_H_

#include "util.h"
#include "model.h"
#include "const.h"
#include "confCompute.h"
#include <vector>
#include <algorithm>
#include <map>
#include <cmath>
#include <cstdlib>
#include <fstream>

void writeTopBuckRMSEs(Model& origModel, Model& fullModel, gk_csr_t* graphMat,
    gk_csr_t* trainMat,
    float lambda, int max_niter, std::unordered_set<int>& invalUsers, 
    std::unordered_set<int>& invalItems, std::unordered_set<int>& filtItems,
    int nSampUsers, int seed, int N, std::string prefix);
void pprSampUsersRMSEProb(gk_csr_t *graphMat, gk_csr_t *trainMt, 
    int nUsers, int nItems, Model& origModel, Model& fullModel,
    float lambda, int max_niter, std::unordered_set<int>& invalUsers, 
    std::unordered_set<int>& invalItems, std::unordered_set<int>& filtItems, 
    int nSampUsers, int seed, std::string prefix);
void pprUsersRMSEProb(gk_csr_t *graphMat, 
    int nUsers, int nItems, Model& origModel, Model& fullModel,
    float lambda, std::unordered_set<int>& invalUsers, 
    std::unordered_set<int>& invalItems, std::unordered_set<int>& filtItems, 
    std::vector<int> users, std::string& prefix);
std::vector<std::pair<int, double>> itemGraphItemScores(int user, 
    gk_csr_t *graphMat, gk_csr_t *mat, float lambda, int nUsers, 
    int nItems, std::unordered_set<int>& invalItems);
void svdSampUsersRMSEProb(gk_csr_t *trainMat, int nUsers, int nItems, 
    Model& origModel, Model& fullModel, Model& svdModel,
    std::unordered_set<int>& invalUsers, std::unordered_set<int>& invalItems, 
    std::unordered_set<int>& filtItems, 
    int nSampUsers, int seed, std::string prefix);
#endif
