#ifndef _MODEL_H_
#define _MODEL_H_

#include "GKlib.h"
#include "const.h"
#include "datastruct.h"
#include "defs.h"
#include "util.h"
#include <Eigen/Cholesky>
#include <Eigen/Dense>
#include <Eigen/LU>
#include <chrono>
#include <cstdio>
#include <iostream>
#include <numeric>
#include <queue>
#include <random>
#include <set>
#include <string>
#include <tuple>
#include <vector>

class Model {

public:
  int nUsers;
  int nItems;
  int facDim;
  int trainSeed;
  float origLearnRate;
  float learnRate;
  float rhoRMS; // poissonDrop: sigmoid -> k or steepness e.g. 20
  float alpha;  // poissonDrop: alpha -> center e.g. 0.5
  int maxIter;
  float uReg;
  float iReg;
  float sing_a, sing_b; // hyper param for singular val reg
  Eigen::MatrixXf uFac;
  Eigen::MatrixXf iFac;
  Eigen::VectorXf uBias;
  Eigen::VectorXf iBias;
  Eigen::VectorXf singularVals;
  double mu; // global bias

  // declare constructor
  Model(const Params &params);
  Model(int nUsers, int nItems, int facDim);
  Model(int nUsers, int nItems, const Params &params);
  Model(const Params &params, int seed);
  Model(const Params &params, const char *uFacName, const char *iFacName,
        int seed);
  Model(const Params &params, const char *uFacName, const char *iFacName,
        const char *uBFName, const char *iBFName, const char *gBFName,
        int seed);

  // declare virtual method for train
  virtual void train(const Data &data, Model &bestModel,
                     std::unordered_set<int> &invalidUsers,
                     std::unordered_set<int> &invalidItems) {
    std::cerr << "\nTraining not in base class";
  };
  virtual void partialTrain(const Data &data, Model &bestModel,
                            std::unordered_set<int> &invalidUsers,
                            std::unordered_set<int> &invalidItems) {
    std::cerr << "\nPartial Training not in base class" << std::endl;
  };

  // virtual method for training on a part of submatrix
  virtual void subTrain(const Data &data, Model &bestModel, int uStart,
                        int uEnd, int iStart, int iEnd) {
    std::cerr << "\nsubTrain not in base class";
  };

  virtual void fixTrain(const Data &data, Model &bestModel, int uStart,
                        int uEnd, int iStart, int iEnd) {
    std::cerr << "\nsubTrain not in base class";
  };
  virtual void subExTrain(const Data &data, Model &bestModel, int uStart,
                          int uEnd, int iStart, int iEnd) {
    std::cerr << "\nsubExTrain not in base class";
  };

  virtual double objective(const Data &data);
  virtual double objective(const Data &data,
                           std::unordered_set<int> &invalidUsers,
                           std::unordered_set<int> &invalidItems);
  double objectiveSing(const Data &data, std::unordered_set<int> &invalidUsers,
                       std::unordered_set<int> &invalidItems);
  double objective(const Data &data, std::unordered_set<int> &invalidUsers,
                   std::unordered_set<int> &invalidItems,
                   std::vector<std::tuple<int, int, float>> &trainRatings);
  double objectiveSubMat(const Data &data, int uStart, int uEnd, int iStart,
                         int iEnd);
  double objectiveExSubMat(const Data &data, int uStart, int uEnd, int iStart,
                           int iEnd);
  bool isTerminateModel(Model &bestModel, const Data &data, int iter,
                        int &bestIter, double &bestObj, double &prevObj);
  bool isTerminateModel(Model &bestModel, const Data &data, int iter,
                        int &bestIter, double &bestObj, double &prevObj,
                        std::unordered_set<int> &invalidUsers,
                        std::unordered_set<int> &invalidItems);
  virtual bool isTerminateModel(Model &bestModel, const Data &data, int iter,
                                int &bestIter, double &bestObj, double &prevObj,
                                double &bestValRMSE, double &prevValRMSE,
                                std::unordered_set<int> &invalidUsers,
                                std::unordered_set<int> &invalidItems);
  bool isTerminateModelSing(Model &bestModel, const Data &data, int iter,
                            int &bestIter, double &bestObj, double &prevObj,
                            double &bestValRMSE, double &prevValRMSE,
                            std::unordered_set<int> &invalidUsers,
                            std::unordered_set<int> &invalidItems);
  bool isTerminateModel(Model &bestModel, const Data &data, int iter,
                        int &bestIter, double &bestObj, double &prevObj,
                        std::unordered_set<int> &invalidUsers,
                        std::unordered_set<int> &invalidItems,
                        std::vector<std::tuple<int, int, float>> &trainRatings);
  bool isTerminateModelSubMat(Model &bestModel, const Data &data, int iter,
                              int &bestIter, double &bestObj, double &prevObj,
                              int uStart, int uEnd, int iStart, int iEnd);
  bool isTerminateModelExSubMat(Model &bestModel, const Data &data, int iter,
                                int &bestIter, double &bestObj, double &prevObj,
                                int uStart, int uEnd, int iStart, int iEnd);
  double RMSE(gk_csr_t *mat);
  std::pair<int, double> RMSE(gk_csr_t *mat, std::unordered_set<int> &filtItems,
                              std::unordered_set<int> &invalidUsers,
                              std::unordered_set<int> &invalidItems);
  std::pair<int, double> RMSEU(gk_csr_t *mat,
                               std::unordered_set<int> &filtItems,
                               std::unordered_set<int> &invalidUsers,
                               std::unordered_set<int> &invalidItems);
  double RMSE(gk_csr_t *mat, std::unordered_set<int> &invalidUsers,
              std::unordered_set<int> &invalidItems);
  double RMSEItem(gk_csr_t *mat, std::unordered_set<int> &invalidUsers,
                  std::unordered_set<int> &invalidItems, int item);
  double RMSEUser(gk_csr_t *mat, std::unordered_set<int> &invalidUsers,
                  std::unordered_set<int> &invalidItems, int item);
  double RMSE(std::vector<std::tuple<int, int, float>> &trainRatings);
  double RMSE(gk_csr_t *mat, std::unordered_set<int> &invalidUsers,
              std::unordered_set<int> &invalidItems, Model &origModel);
  double subMatRMSE(gk_csr_t *mat, int uStart, int uEnd, int iStart, int iEnd);
  double subMatExRMSE(gk_csr_t *mat, int uStart, int uEnd, int iStart,
                      int iEnd);
  double fullLowRankErr(const Data &data);
  double subMatKnownRankErr(const Data &data, int uStart, int uEnd, int iStart,
                            int iEnd);
  double subMatKnownRankNonObsErr(const Data &data, int uStart, int uEnd,
                                  int iStart, int iEnd);
  double subMatKnownRankNonObsErrWSet(const Data &data, int uStart, int uEnd,
                                      int iStart, int iEnd,
                                      std::set<int> exUSet,
                                      std::set<int> exISet);
  double fullRMSE(const Data &data);
  virtual double estRating(int user, int item);
  std::string modelSignature();
  void display();
  void save(std::string prefix);
  void saveFacs(std::string prefix);
  void load(std::string prefix);
  void loadFacs(std::string prefix);
  void load(const char *uFacName, const char *iFacName);
  void load(const char *uFacName, const char *iFacName, const char *uBFName,
            const char *iBFName, const char *gBFName);
  void updateFac(std::vector<double> &fac, std::vector<double> &grad);
  double estAvgRating(int user, std::unordered_set<int> &invalidItems);
  double estAvgRating(std::unordered_set<int> &invalidUsers,
                      std::unordered_set<int> &invalidItems);
  void updateMatWRatings(gk_csr_t *mat);
  void updateMatWRatingsGaussianNoise(gk_csr_t *mat);
  double fullLowRankErr(const Data &data, std::unordered_set<int> &invalidUsers,
                        std::unordered_set<int> &invalidItems);
  double fullLowRankErr(const Data &data, std::unordered_set<int> &invalidUsers,
                        std::unordered_set<int> &invalidItems,
                        Model &origModel);
  std::vector<std::pair<double, double>> itemsMeanVar(gk_csr_t *mat);
  std::vector<std::pair<double, double>> usersMeanVar(gk_csr_t *mat);
  void saveBinFacs(std::string prefix);
  void loadBinFacs(std::string prefix);
  void initInfreqFactors(const Params &params, const Data &data);
  std::pair<double, double> hiLoNorms(std::unordered_set<int> &items);
  std::pair<int, double> SE(gk_csr_t *mat, std::unordered_set<int> &filtItems,
                            std::unordered_set<int> &invalidUsers,
                            std::unordered_set<int> &invalidItems);
  double hitRate(const Data &data, std::unordered_set<int> &invalidUsers,
                 std::unordered_set<int> &invalidItems, gk_csr_t *testMat,
                 const int N = 10);
  double hitRateNegatives(const Data &data,
                          std::unordered_set<int> &invalidUsers,
                          std::unordered_set<int> &invalidItems,
                          gk_csr_t *testMat, const int N = 10);
  bool isTerminateModelHR(Model &bestModel, const Data &data, int iter,
                          int &bestIter, double &bestHR, double &prevHR,
                          std::unordered_set<int> &invalidUsers,
                          std::unordered_set<int> &invalidItems);
  bool isTerminateModelNDCG(Model &bestModel, const Data &data, int iter,
                            int &bestIter, double &bestNDCG, double &prevNDCG,
                            std::unordered_set<int> &invalidUsers,
                            std::unordered_set<int> &invalidItems);
  std::pair<int, double> hitRateU(const Data &data,
                                  std::unordered_set<int> &filtUsers,
                                  std::unordered_set<int> &invalidUsers,
                                  std::unordered_set<int> &invalidItems,
                                  gk_csr_t *testMat);
  std::pair<int, double> hitRateI(const Data &data,
                                  std::unordered_set<int> &filtItems,
                                  std::unordered_set<int> &invalidUsers,
                                  std::unordered_set<int> &invalidItems,
                                  gk_csr_t *testMat);
  double arHR(const Data &data, std::unordered_set<int> &invalidUsers,
              std::unordered_set<int> &invalidItems, gk_csr_t *testMat);
  std::pair<double, double> arHRU(const Data &data,
                                  std::unordered_set<int> &filtUsers,
                                  std::unordered_set<int> &invalidUsers,
                                  std::unordered_set<int> &invalidItems,
                                  gk_csr_t *testMat);
  std::pair<double, double> arHRI(const Data &data,
                                  std::unordered_set<int> &filtItems,
                                  std::unordered_set<int> &invalidUsers,
                                  std::unordered_set<int> &invalidItems,
                                  gk_csr_t *testMat);
  double NDCG(std::unordered_set<int> &invalidUsers,
              std::unordered_set<int> &invalidItems, gk_csr_t *testMat,
              const int N = 10);
  double NDCGNegatives(const Data &data, std::unordered_set<int> &invalidUsers,
                       std::unordered_set<int> &invalidItems, gk_csr_t *testMat,
                       const int N = 10);
  std::pair<int, double> NDCGU(std::unordered_set<int> &filtUsers,
                               std::unordered_set<int> &invalidUsers,
                               std::unordered_set<int> &invalidItems,
                               gk_csr_t *testMat);
  std::pair<int, double> NDCGI(std::unordered_set<int> &filtItems,
                               std::unordered_set<int> &invalidUsers,
                               std::unordered_set<int> &invalidItems,
                               gk_csr_t *testMat);
};
#endif
