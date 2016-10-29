#ifndef _MODELMFFREQ_H_
#define _MODELMFFREQ_H_

#include <iostream>
#include <vector>
#include <set>
#include <cstdio>
#include <utility>
#include <unordered_set>
#include <string>
#include "io.h"
#include "model.h"
//#include "svd.h"
//#include "svdLapack.h"
#include "svdFrmsvdlib.h"


class ModelMFFreq : public Model {
  public:
    ModelMFFreq(const Params& params) : Model(params) {}
    ModelMFFreq(const Params& params, int seed) : Model(params, seed) {}
    ModelMFFreq(const Params& params, const char*uFacName, const char* iFacName, 
        int seed):Model(params, uFacName, iFacName, seed) {}
    void train(const Data& data, Model& bestModel, 
        std::unordered_set<int>& invalidUsers, 
        std::unordered_set<int>& invalidItems);
    void subTrain(const Data& data, Model& bestModel, 
        std::unordered_set<int>& invalidUsers, 
        std::unordered_set<int>& invalidItems);
    void updateModelInval(
        const std::vector<std::tuple<int, int, float>>& uiRatings,
        std::vector<size_t>& uiRatingInds, std::unordered_set<int>& invalidUsers,
        std::unordered_set<int>& invalidItems, std::mt19937& mt);
    void updateModel(
        const std::vector<std::tuple<int, int, float>>& uiRatings,
        std::vector<size_t>& uiRatingInds, std::mt19937& mt);
    void updateModelCol(
        const std::vector<std::tuple<int, int, float>>& uiRatings,
        std::vector<size_t>& uiRatingInds, std::mt19937& mt);
    void updateModelRow(
        const std::vector<std::tuple<int, int, float>>& uiRatings,
        std::vector<size_t>& uiRatingInds, std::mt19937& mt);

};

#endif

