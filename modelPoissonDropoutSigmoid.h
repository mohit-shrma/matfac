#ifndef _MODEL_POISSON_DROPOUT_SIGMOID_H_
#define _MODEL_POISSON_DROPOUT_SIGMOID_H_

#include "io.h"
#include "model.h"
#include "svdFrmsvdlib.h"
#include "util.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <omp.h>
#include <set>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#define DISP_ITER 50
#define SAVE_ITER 100

class ModelPoissonDropoutSigmoid : public Model {

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

  ModelPoissonDropoutSigmoid(const Params &params,
                             std::vector<double> &userRankMap,
                             std::vector<double> &itemRankMap,
                             std::vector<double> &userFreq,
                             std::vector<double> &itemFreq)
      : Model(params), userRankMap(userRankMap), itemRankMap(itemRankMap),
        userFreq(userFreq), itemFreq(itemFreq) {
    // intialize factorial table
    factorial.push_back(1);
    for (int i = 1; i <= params.facDim + 1; i++) {
      factorial.push_back(factorial.back() * ((double)i));
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
  ModelPoissonDropoutSigmoid(const Params &params, int seed,
                             std::vector<double> &userRankMap,
                             std::vector<double> &itemRankMap,
                             std::vector<double> &userFreq,
                             std::vector<double> &itemFreq)
      : Model(params, seed), userRankMap(userRankMap), itemRankMap(itemRankMap),
        userFreq(userFreq), itemFreq(itemFreq) {
    // intialize factorial table
    factorial.push_back(1);
    for (int i = 1; i <= params.facDim + 1; i++) {
      factorial.push_back(factorial.back() * ((double)i));
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
  ModelPoissonDropoutSigmoid(const Params &params, const char *uFacName,
                             const char *iFacName, int seed,
                             std::vector<double> &userRankMap,
                             std::vector<double> &itemRankMap,
                             std::vector<double> &userFreq,
                             std::vector<double> &itemFreq)
      : Model(params, uFacName, iFacName, seed), userRankMap(userRankMap),
        itemRankMap(itemRankMap), userFreq(userFreq), itemFreq(itemFreq) {
    // intialize factorial table
    factorial.push_back(1);
    for (int i = 1; i <= params.facDim + 1; i++) {
      factorial.push_back(factorial.back() * ((double)i));
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
  ModelPoissonDropoutSigmoid(const Params &params, int seed)
      : Model(params, seed) {
    // intialize factorial table
    factorial.push_back(1);
    for (int i = 1; i <= params.facDim + 1; i++) {
      factorial.push_back(factorial.back() * ((double)i));
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

  virtual void train(const Data &data, Model &bestModel,
                     std::unordered_set<int> &invalidUsers,
                     std::unordered_set<int> &invalidItems) override;
  virtual double estRating(int user, int item) override;
  void initCDFRanks();
};

#endif
