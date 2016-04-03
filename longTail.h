#ifndef _LONG_TAIL_H_
#define _LONG_TAIL_H_

#include <vector>
#include <algorithm>
#include "model.h"
#include "topBucketComp.h"

void topNRec(Model& model, gk_csr_t *trainMat, gk_csr_t *testMat, 
    gk_csr_t *graphMat, float lambda,
    std::unordered_set<int>& invalidItems,
    std::unordered_set<int>& invalidUsers,
    std::unordered_set<int>& headItems,
    int N, int seed);
void topNRecTail(Model& model, gk_csr_t *trainMat, gk_csr_t *testMat, 
    gk_csr_t *graphMat, float lambda,
    std::unordered_set<int>& invalidItems,
    std::unordered_set<int>& invalidUsers,
    float headPc,
    int N, int seed, std::string opFileName);
void topNRecTailWSVD(Model& model, Model& svdModel, gk_csr_t *trainMat, 
    gk_csr_t *testMat, gk_csr_t *graphMat, float lambda,
    std::unordered_set<int>& invalidItems, std::unordered_set<int>& invalidUsers,
    float headPc, int N, int seed, std::string opFileName);
#endif
