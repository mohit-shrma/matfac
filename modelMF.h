#ifndef _MODELMF_H_
#define _MODELMF_H_

#include "io.h"
#include "model.h"
#include "svdFrmsvdlib.h"
#include <cstdio>
#include <iostream>
#include <omp.h>
#include <set>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#define DISP_ITER 50
#define SAVE_ITER 50

typedef Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> MatrixXb;

class ModelMF : public Model {

public:
  ModelMF(int nUsers, int nItems, int facDim) : Model(nUsers, nItems, facDim) {}
  ModelMF(const Params &params) : Model(params) {}
  ModelMF(const Params &params, int seed) : Model(params, seed) {}
  ModelMF(const Params &params, const char *uFacName, const char *iFacName,
          int seed)
      : Model(params, uFacName, iFacName, seed) {}
  void train(const Data &data, Model &bestModel,
             std::unordered_set<int> &invalidUsers,
             std::unordered_set<int> &invalidItems);
  void trainSGDPar(const Data &data, Model &bestModel,
                   std::unordered_set<int> &invalidUsers,
                   std::unordered_set<int> &invalidItems);
  void trainSGDParSVD(const Data &data, Model &bestModel,
                      std::unordered_set<int> &invalidUsers,
                      std::unordered_set<int> &invalidItems);
  void trainUShuffle(const Data &data, Model &bestModel,
                     std::unordered_set<int> &invalidUsers,
                     std::unordered_set<int> &invalidItems);
  void trainALS(const Data &data, Model &bestModel,
                std::unordered_set<int> &invalidUsers,
                std::unordered_set<int> &invalidItems);
  void trainCCDPP(const Data &data, Model &bestModel,
                  std::unordered_set<int> &invalidUsers,
                  std::unordered_set<int> &invalidItems);
  void trainCCDPPFreqAdap(const Data &data, Model &bestModel,
                          std::unordered_set<int> &invalidUsers,
                          std::unordered_set<int> &invalidItems);
  void trainCCD(const Data &data, Model &bestModel,
                std::unordered_set<int> &invalidUsers,
                std::unordered_set<int> &invalidItems);
  void hogTrain(const Data &data, Model &bestModel,
                std::unordered_set<int> &invalidUsers,
                std::unordered_set<int> &invalidItems);
};

#endif
