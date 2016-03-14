#ifndef _TOP_BUCKET_COMP_H_
#define _TOP_BUCKET_COMP_H_

#include "util.h"
#include "model.h"
#include "const.h"
#include <vector>
#include <algorithm>
#include <map>
#include <cmath>
#include <cstdlib>
#include <fstream>

void writeTopBuckRMSEs(Model& origModel, Model& fullModel, gk_csr_t* graphMat,
    float lambda, int max_niter, std::unordered_set<int>& invalUsers, 
    std::unordered_set<int>& invalItems, std::unordered_set<int>& filtItems,
    int nSampUsers, int seed, int N, std::string& prefix);

#endif
