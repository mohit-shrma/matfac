#include <iostream>
#include <cstdlib>
#include <future>
#include <chrono>
#include <thread>

#define EIGEN_USE_MKL_ALL

#include "io.h"
#include "util.h"
#include "datastruct.h"
#include "modelMF.h"
#include "modelMFBias.h"
#include "confCompute.h"
#include "topBucketComp.h"
#include "longTail.h"
#include "analyzeModels.h"
#include "modelDropoutMF.h"
#include "modelPoissonDropout.h"
#include "modelIncrement.h"
#include "modelMFBPR.h"
#include "modelBPRPoissonDropout.h"
#include "modelInvPopMF.h"
#include <gflags/gflags.h>

DEFINE_uint64(maxiter, 5000, "number of iterations");
DEFINE_uint64(facdim, 5, "dimension of factors");
DEFINE_uint64(svdfacdim, 5, "dimension of factors");
DEFINE_double(ureg, 0.01, "user regularization");
DEFINE_double(ireg, 0.01, "item regularization");
DEFINE_double(learnrate, 0.005, "learn rate");
DEFINE_double(rhorms, 0.0, "rho rms");
DEFINE_double(alpha, 0.0, "alpha");
DEFINE_int32(seed, 1, "seed");

DEFINE_string(trainmat, "", "training CSR matrix");
DEFINE_string(testmat, "", "test CSR matrix");
DEFINE_string(valmat, "", "validation CSR matrix");
DEFINE_string(graphmat, "", "item-item graph csr matrix");
DEFINE_string(origufac, "", "original user factors");
DEFINE_string(origifac, "", "original item factors");
DEFINE_string(initufac, "", "initial user factors");
DEFINE_string(initifac, "", "initial item factors");
DEFINE_string(prefix, "", "prefix to prepend to logs n factors");
DEFINE_string(method, "sgd", "sgd|sgdpar|sgdu|hogsgd|als|ccd|ccd++");


Params parse_cmd_line(int argc, char *argv[]) {

  google::ParseCommandLineFlags(&argc, &argv, true);

  bool isexit = false;
  if (FLAGS_trainmat.empty() || FLAGS_testmat.empty() || FLAGS_valmat.empty()) {
    std::cerr << "Missing either: train, test or val matrix" << std::endl;
    isexit = true;
  }
  if (FLAGS_prefix.empty()) {
    std::cerr << "Missing prefix string to prepend: --prefix " << std::endl;
    isexit = true;
  }

  if (isexit) {exit(-1);}

  Params params(FLAGS_facdim, FLAGS_maxiter, FLAGS_svdfacdim, FLAGS_seed,
      FLAGS_ureg, FLAGS_ireg, FLAGS_learnrate, FLAGS_rhorms,
      FLAGS_alpha, FLAGS_trainmat, FLAGS_testmat, FLAGS_valmat, FLAGS_graphmat,
      FLAGS_origufac, FLAGS_origifac, FLAGS_initufac, FLAGS_initifac, FLAGS_prefix);

  return params;
}


void computeSampTopNFrmFullModel(Data& data, Params& params) {

  std::cout << "\nCreating full model...";
  ModelMF fullModel(params, params.seed);
  //svdFrmSvdlibCSR(data.trainMat, fullModel.facDim, fullModel.uFac,
  //    fullModel.iFac, false);
  //load previously learned factors
  fullModel.loadFacs(params.prefix);

  std::cout << "\nCreating original model...";
  ModelMF origModel(params, params.origUFacFile, params.origIFacFile,
      params.seed);

  std::cout << "\nCreating SVD model... dim: " << params.svdFacDim;
  Params svdParams(params);
  svdParams.facDim = svdParams.svdFacDim;
  ModelMF svdModel(svdParams, svdParams.seed);
  svdFrmSvdlibCSRSparsityEig(data.trainMat, svdModel.facDim, svdModel.uFac,
      svdModel.iFac, true);

  //get number of ratings per user and item, i.e. frequency
  auto rowColFreq = getRowColFreq(data.trainMat);
  auto userFreq = rowColFreq.first;
  auto itemFreq = rowColFreq.second;

  auto usersMeanStd = meanStdDev(userFreq);
  auto itemsMeanStd = meanStdDev(itemFreq);

  std::cout << "user freq mean: " << usersMeanStd.first << " std: "
    << usersMeanStd.second << std::endl;
  std::cout << "item freq mean: " << itemsMeanStd.first << " std: "
    << itemsMeanStd.second << std::endl;

  std::unordered_set<int> invalidUsers;
  std::unordered_set<int> invalidItems;
  std::string modelSign = fullModel.modelSignature();
  std::cout << "\nModel sign: " << modelSign << std::endl;

  std::string prefix = std::string(params.prefix) + "_" + modelSign + "_invalUsers.txt";
  std::vector<int> invalUsersVec = readVector(prefix.c_str());
  prefix = std::string(params.prefix) + "_" + modelSign + "_invalItems.txt";
  std::vector<int> invalItemsVec = readVector(prefix.c_str());

  for (auto v: invalUsersVec) {
    invalidUsers.insert(v);
  }

  for (auto v: invalItemsVec) {
    invalidItems.insert(v);
  }

  ModelMF bestModel(fullModel);
  //std::cout << "\nStarting model train...";
  //fullModel.train(data, bestModel, invalidUsers, invalidItems);
  std::cout << "\nTrain RMSE: " << bestModel.RMSE(data.trainMat, invalidUsers, invalidItems);
  std::cout << "\nTest RMSE: " << bestModel.RMSE(data.testMat, invalidUsers, invalidItems);
  std::cout << "\nVal RMSE: " << bestModel.RMSE(data.valMat, invalidUsers, invalidItems);
  //std::cout << "\nFull RMSE: " << bestModel.fullLowRankErr(data, invalidUsers, invalidItems);
  std::cout << std::endl;
  /*
  //write out invalid users
  std::string prefix = std::string(params.prefix) + "_" + modelSign + "_invalUsers.txt";
  writeContainer(begin(invalidUsers), end(invalidUsers), prefix.c_str());

  //write out invalid items
  prefix = std::string(params.prefix) + "_" + modelSign + "_invalItems.txt";
  writeContainer(begin(invalidItems), end(invalidItems), prefix.c_str());
  */

  std::cout << "No. of invalid users: " << invalidUsers.size() << std::endl;
  std::cout << "No. of invalid items: " << invalidItems.size() << std::endl;

  //order item in decreasing order of frequency
  std::vector<std::pair<int, double>> itemFreqPairs;
  for (int i = 0; i < itemFreq.size(); i++) {
    itemFreqPairs.push_back(std::make_pair(i, itemFreq[i]));
  }
  //comparison to sort in decreasing order
  std::sort(itemFreqPairs.begin(), itemFreqPairs.end(), descComp);

  int nSampUsers = 5000;
  int nUsers = data.trainMat->nrows;
  int nItems = data.trainMat->ncols;

  //get filtered items corresponding to head items
  //auto headItems = getHeadItems(data.trainMat, 0.2);


  std::vector<float> freqPcs = {1.0, 5.0, 10, 25, 50};
  for (auto&& filtPc: freqPcs) {
    std::unordered_set<int> filtItems, notFiltItems;
    float filtSzThresh = (filtPc/100.0)*((float)itemFreqPairs.size());
    //std::cout << "No. of top " << filtPc << " % freq items: " <<  filtSzThresh << std::endl;
    for (auto&& pair: itemFreqPairs) {
      if (filtItems.size() <= filtSzThresh) {
        filtItems.insert(pair.first);
      } else {
        notFiltItems.insert(pair.first);
      }
    }
    auto countNRMSE = bestModel.RMSE(data.testMat, filtItems, invalidUsers, invalidItems);
    auto countNRMSE2 = bestModel.RMSE(data.testMat, notFiltItems, invalidUsers, invalidItems);
    std::cout << "FiltPc: " << (int)filtPc << " " << countNRMSE.first << " " << countNRMSE.second
      << " " << countNRMSE2.first << " " <<countNRMSE2.second
      << " " << countNRMSE.first + countNRMSE2.first << std::endl;
  }


  std::vector<float> maxFreqs = {5, 10, 15, 20, 30, 50, 75, 100, 150, 200};
  for (auto&& maxFreq: maxFreqs) {
    std::unordered_set<int> freqItems, notFreqItems;
    for (auto&& pair: itemFreqPairs) {
      if (pair.second < maxFreq) {
        notFreqItems.insert(pair.first);
      } else {
        freqItems.insert(pair.first);
      }
    }
    auto countNRMSE = bestModel.RMSE(data.testMat, freqItems, invalidUsers, invalidItems);
    auto countNRMSE2 = bestModel.RMSE(data.testMat, notFreqItems, invalidUsers, invalidItems);
    std::cout << "MaxFreq: " << (int)maxFreq << " " << countNRMSE.first << " "
      << countNRMSE.second << " " << countNRMSE2.first << " " << countNRMSE2.second
      << " " << countNRMSE.first + countNRMSE2.first << std::endl;
  }

  return ;

  auto itemsMeanVar = trainItemsMeanVar(data.trainMat);
  double avgVar = 0;
  for (const auto& itemMeanVar: itemsMeanVar) {
    avgVar += itemMeanVar.second;
  }
  avgVar = avgVar/itemsMeanVar.size();

  std::cout << "Average variance: " << avgVar << std::endl;

  //std::vector<float> freqPcs = {1.0, 5.0, 10, 25, 50};
  //for (auto&& filtPc: freqPcs) {
  for (auto&& maxFreq: maxFreqs) {
    std::unordered_set<int> freqItems, inFreqItems;
    //float filtSzThresh = (filtPc/100.0)*((float)itemFreqPairs.size());
    for (auto&& pair: itemFreqPairs) {
      //if (freqItems.size() <= filtSzThresh) {
      if (pair.second >= maxFreq) {
        freqItems.insert(pair.first);
      } else {
        inFreqItems.insert(pair.first);
      }
    }

    std::vector<float> varThreshs = {0, 0.01, 0.05};

    for (const auto& varThresh: varThreshs) {

      float varianceThresh = avgVar*(1 + varThresh);
      std::unordered_set<int> highVarFreqItems, highVarNonfreqItems;

      for (auto& item: freqItems) {
        if (itemsMeanVar[item].second >= varianceThresh) {
          highVarFreqItems.insert(item);
        }
      }

      for (auto& item: inFreqItems) {
        if (itemsMeanVar[item].second >= varianceThresh) {
          highVarNonfreqItems.insert(item);
        }
      }

      auto countNRMSE = bestModel.RMSE(data.testMat, highVarFreqItems,
          invalidUsers, invalidItems);
      auto countNRMSE2 = bestModel.RMSE(data.testMat, highVarNonfreqItems,
          invalidUsers, invalidItems);
      int res =  (countNRMSE.second < countNRMSE2.second)?1:0;
      std::cout << "FreqVar: " << varThresh << " MaxFreq: " << maxFreq << " "
        << " " << countNRMSE.first << " " << countNRMSE.second
        << " " << countNRMSE2.first << " " << countNRMSE2.second
        << " " << varianceThresh  << " " << res
        << std::endl;
    }

  }

  std::unordered_set<int> filtItems;
  /*
  //add filtItems to invalItems
  for (auto&& item: filtItems) {
    invalItems.insert(item);
  }
  */

  std::cout << "\nnInvalidUsers: " << invalidUsers.size();
  std::cout << "\nnInvalidItems: " << invalidItems.size() <<std::endl;

  /*
  std::vector<int> invGraphItems = getInvalidUsers(data.graphMat);
  int found = 0;
  for (auto&& item: invGraphItems) {
    auto search = invalItems.find(item);
    if (search != invalItems.end()) {
      found++;
    }
  }

  std::cout << "\nInvGraphItems: " << invGraphItems.size()
    << " found in invalid items: " << found << std::endl;
  */

  //std::vector<int> users = {92, 43970};
  //getUserStats(users, data.trainMat, invalItems, "uStats.txt");

  //std::vector<int> users = readVector("users_300_400.txt");
  //getUserStats(users, data.trainMat, invalItems);

  /*
  prefix = std::string(params.prefix) + "_" + std::to_string(params.alpha)
    + "_users";
  pprUsersRMSEProb(data.graphMat, nUsers, nItems, origModel, fullModel,
      params.alpha, invalUsers, invalItems, users, prefix);
  */


  /*
  std::vector<float> lambdas = {0.01, 0.25, 0.5, 0.75, 0.99};
  int nThreads = lambdas.size() - 1;
  std::vector<std::thread> threads(nThreads);
  for (int thInd = 0; thInd < nThreads; thInd++) {

    prefix = std::string(params.prefix) + "_50_100_"
              + std::to_string(lambdas[thInd]);
    threads[thInd] = std::thread(pprSampUsersRMSEProb,
        data.graphMat, data.trainMat, nUsers, nItems, std::ref(origModel),
        std::ref(fullModel), lambdas[thInd], MAX_PR_ITER, std::ref(invalUsers),
        std::ref(invalItems), std::ref(filtItems), 100, params.seed, prefix);
   */
    //prefix = std::string(params.prefix) + "_" + std::to_string(lambdas[thInd])
    //  + "_" + std::to_string(N);
    /*
    threads[thInd] = std::thread(writeTopBuckRMSEs,
        std::ref(origModel), std::ref(fullModel), data.graphMat, data.trainMat,
        lambdas[thInd], MAX_PR_ITER, std::ref(invalUsers),
        std::ref(invalItems), std::ref(filtItems), nSampUsers, params.seed, N,
        prefix);
    */
  /*
  }

  //last parameter in main thread
  prefix = std::string(params.prefix) + "_50_100_"
            + std::to_string(lambdas[nThreads]);
  pprSampUsersRMSEProb(data.graphMat, data.trainMat, nUsers, nItems, origModel, fullModel,
      lambdas[nThreads], MAX_PR_ITER, invalUsers, invalItems, filtItems, 100,
      params.seed, prefix);
  */
  //prefix = std::string(params.prefix) + "_" + std::to_string(lambdas[nThreads])
  //  + "_" + std::to_string(N);
  /*
  writeTopBuckRMSEs(origModel, fullModel, svdModel, data.graphMat, data.trainMat,
      0.01, MAX_PR_ITER, invalidUsers, invalidItems, filtItems, nSampUsers,
      params.seed, 100, params.prefix);
  */
  /*
  //last parameter in main thread
  prefix = std::string(params.prefix) + "_" + std::to_string(params.alpha)
    + "_" + std::to_string(N);
  writeTopBuckRMSEs(fullModel, origModel, data.graphMat, params.alpha,
      MAX_PR_ITER, invalUsers, invalItems, filtItems, nSampUsers, params.seed,
      N, prefix);
  */

  /*
  //wait for the threads to finish
  std::cout << "\nWaiting for threads to finish..." << std::endl;
  std::for_each(threads.begin(), threads.end(), std::mem_fn(&std::thread::join));
  */

  /*
  prefix = std::string(params.prefix) + "_sampPPR_" + std::to_string(0.01);
  pprSampUsersRMSEProb(data.graphMat, data.trainMat, nUsers, nItems, origModel, fullModel,
      0.01, MAX_PR_ITER, invalidUsers, invalidItems, filtItems, 5000,
      params.seed, prefix);
  */

  /*
  prefix = std::string(params.prefix) + "_sampFreq";
  freqSampUsersRMSEProb(data.trainMat, nUsers, nItems, origModel, fullModel,
      invalidUsers, invalidItems, filtItems, 5000, params.seed, prefix);
  */

  /*
  prefix = std::string(params.prefix) + "_samp_SVD";
  svdSampUsersRMSEProb(data.trainMat, nUsers, nItems, origModel, fullModel,
      svdModel, invalidUsers, invalidItems, filtItems, 5000, params.seed, prefix);
  */

  /*
  prefix = std::string(params.prefix) + "_sampOpt";
  optSampUsersRMSEProb(data.trainMat, nUsers, nItems, origModel, fullModel,
      invalidUsers, invalidItems, filtItems, 5000, params.seed, prefix);
  */


  prefix = std::string(params.prefix) + "_top_";

  std::vector<double> alphas = {0.0001, 0.001, 0.01, 0.1, 0.5, 1, 10, 100};
  //predSampUsersRMSEProbPar(data, nUsers, nItems, origModel, fullModel,
  //  svdModel, invalidUsers, invalidItems, filtItems, 5000, params.seed,
  //  prefix);
  predSampUsersRMSEFreqPar(data, nUsers, nItems, origModel, fullModel,
    invalidUsers, invalidItems, filtItems, 5000, params.seed,
    prefix);

}


void computeFreqRMSEs(Data& data, Params& params) {

  std::cout << "\nCreating full model...";
  ModelMF fullModel(params, params.seed);
  //svdFrmSvdlibCSR(data.trainMat, fullModel.facDim, fullModel.uFac,
  //    fullModel.iFac, false);
  //load previously learned factors
  fullModel.loadFacs(params.prefix);

  //get number of ratings per user and item, i.e. frequency
  auto rowColFreq = getRowColFreq(data.trainMat);
  auto userFreq = rowColFreq.first;
  auto itemFreq = rowColFreq.second;

  auto usersMeanStd = meanStdDev(userFreq);
  auto itemsMeanStd = meanStdDev(itemFreq);

  std::cout << "user freq mean: " << usersMeanStd.first << " std: "
    << usersMeanStd.second << std::endl;
  std::cout << "item freq mean: " << itemsMeanStd.first << " std: "
    << itemsMeanStd.second << std::endl;

  std::unordered_set<int> invalidUsers;
  std::unordered_set<int> invalidItems;
  std::string modelSign = fullModel.modelSignature();
  std::cout << "\nModel sign: " << modelSign << std::endl;

  std::string prefix = std::string(params.prefix) + "_" + modelSign + "_invalUsers.txt";
  std::vector<int> invalUsersVec = readVector(prefix.c_str());
  prefix = std::string(params.prefix) + "_" + modelSign + "_invalItems.txt";
  std::vector<int> invalItemsVec = readVector(prefix.c_str());

  for (auto v: invalUsersVec) {
    invalidUsers.insert(v);
  }

  for (auto v: invalItemsVec) {
    invalidItems.insert(v);
  }

  ModelMF bestModel(fullModel);
  //std::cout << "\nStarting model train...";
  //fullModel.train(data, bestModel, invalidUsers, invalidItems);
  std::cout << "\nTrain RMSE: " << bestModel.RMSE(data.trainMat, invalidUsers, invalidItems);
  std::cout << "\nTest RMSE: " << bestModel.RMSE(data.testMat, invalidUsers, invalidItems);
  std::cout << "\nVal RMSE: " << bestModel.RMSE(data.valMat, invalidUsers, invalidItems);
  //std::cout << "\nFull RMSE: " << bestModel.fullLowRankErr(data, invalidUsers, invalidItems);
  std::cout << std::endl;

  std::cout << "No. of invalid users: " << invalidUsers.size() << std::endl;
  std::cout << "No. of invalid items: " << invalidItems.size() << std::endl;

  //order item in decreasing order of frequency
  std::vector<std::pair<int, double>> itemFreqPairs;
  for (int i = 0; i < itemFreq.size(); i++) {
    itemFreqPairs.push_back(std::make_pair(i, itemFreq[i]));
  }
  //comparison to sort in decreasing order
  std::sort(itemFreqPairs.begin(), itemFreqPairs.end(), descComp);

  int nSampUsers = 5000;
  int nUsers = data.trainMat->nrows;
  int nItems = data.trainMat->ncols;

  //get filtered items corresponding to head items
  //auto headItems = getHeadItems(data.trainMat, 0.2);

  std::vector<float> freqPcs = {1.0, 5.0, 10, 25, 50};
  for (auto&& filtPc: freqPcs) {
    std::unordered_set<int> filtItems, notFiltItems;
    float filtSzThresh = (filtPc/100.0)*((float)itemFreqPairs.size());
    //std::cout << "No. of top " << filtPc << " % freq items: " <<  filtSzThresh << std::endl;
    for (auto&& pair: itemFreqPairs) {
      if (filtItems.size() <= filtSzThresh) {
        filtItems.insert(pair.first);
      } else {
        notFiltItems.insert(pair.first);
      }
    }
    auto countNRMSE = bestModel.RMSE(data.testMat, filtItems, invalidUsers, invalidItems);
    auto countNRMSE2 = bestModel.RMSE(data.testMat, notFiltItems, invalidUsers, invalidItems);
    auto hiloNormFilt = bestModel.hiLoNorms(filtItems);
    auto hiloNormNotFilt = bestModel.hiLoNorms(notFiltItems);
    std::cout << "FiltPc: " << (int)filtPc << " " << countNRMSE.first << " " << countNRMSE.second
      << " " << countNRMSE2.first << " " <<countNRMSE2.second
      << " " << countNRMSE.first + countNRMSE2.first << std::endl;
    std::cout << "Filt norm hi: " << hiloNormFilt.first << " " << hiloNormFilt.second << std::endl;
    std::cout << "Not-Filt norm hi: " << hiloNormNotFilt.first << " " << hiloNormNotFilt.second << std::endl;

  }

  std::vector<float> maxFreqs = {5, 10, 15, 20, 30, 50, 75, 100, 150, 200};
  for (auto&& maxFreq: maxFreqs) {
    std::unordered_set<int> freqItems, notFreqItems;
    for (auto&& pair: itemFreqPairs) {
      if (pair.second < maxFreq) {
        notFreqItems.insert(pair.first);
      } else {
        freqItems.insert(pair.first);
      }
    }
    auto countNRMSE = bestModel.RMSE(data.testMat, freqItems, invalidUsers, invalidItems);
    auto countNRMSE2 = bestModel.RMSE(data.testMat, notFreqItems, invalidUsers, invalidItems);
    std::cout << "MaxFreq: " << (int)maxFreq << " " << countNRMSE.first << " "
      << countNRMSE.second << " " << countNRMSE2.first << " " << countNRMSE2.second
      << " " << countNRMSE.first + countNRMSE2.first << std::endl;
    auto hiloNormFreq = bestModel.hiLoNorms(freqItems);
    auto hiloNormNotFreq = bestModel.hiLoNorms(notFreqItems);
    std::cout << "Freq norm hi: " << hiloNormFreq.first << " " << hiloNormFreq.second << std::endl;
    std::cout << "Not-Freq norm hi: " << hiloNormNotFreq.first << " " << hiloNormNotFreq.second << std::endl;
  }

}


void diffRankRMSEs(Model& bestModel, const Data& data,
    std::vector<int>& userRankMap, std::vector<int>& itemRankMap,
    std::unordered_set<int> invalidUsers, std::unordered_set<int> invalidItems) {

  std::map<int, std::vector<int>> rankItems;
  for (int item = 0; item < itemRankMap.size(); item++) {
    if (!rankItems.count(itemRankMap[item])) {
      rankItems[itemRankMap[item]] = std::vector<int>();
    }
    rankItems[itemRankMap[item]].push_back(item);
  }

  for (auto& p_rankItems: rankItems) {
    auto rank = p_rankItems.first;
    auto items = p_rankItems.second;
    auto setItems = std::unordered_set<int>(items.begin(), items.end());
    auto countNRMSE = bestModel.RMSE(data.testMat, setItems, invalidUsers, invalidItems);
    std::cout << "Items Rank: " << rank << " " << countNRMSE.first << " "
      << countNRMSE.second << std::endl;
  }

  std::map<int, std::vector<int>> rankUsers;
  for (int user = 0; user < userRankMap.size(); user++) {
    if (!rankUsers.count(userRankMap[user])) {
      rankUsers[userRankMap[user]] = std::vector<int>();
    }
    rankUsers[userRankMap[user]].push_back(user);
  }

  for (auto& p_rankUsers: rankUsers) {
    auto rank = p_rankUsers.first;
    auto users = p_rankUsers.second;
    auto setUsers = std::unordered_set<int>(users.begin(), users.end());
    auto countNRMSE = bestModel.RMSEU(data.testMat, setUsers, invalidUsers, invalidItems);
    std::cout << "Users Rank: " << rank << " " << countNRMSE.first << " "
      << countNRMSE.second << std::endl;
  }

}


void quartileNDCG(Model& bestModel, const Data& data,
    std::vector<std::pair<int, std::vector<int>>> partItems,
    std::vector<std::pair<int, std::vector<int>>> partUsers,
    std::unordered_set<int> invalidUsers, std::unordered_set<int> invalidItems) {

  auto rowColFreq = getRowColFreq(data.trainMat);
  auto userFreq   = rowColFreq.first;
  auto itemFreq   = rowColFreq.second;
  int nItems      = itemFreq.size();
  int nUsers      = userFreq.size();

  auto testItems = getColInd(data.testMat);
  auto testUsers = getRowInd(data.testMat);

  std::cout << "Items Part: ";
  for (auto& pItems: partItems) {
    int partInd = pItems.first;
    auto filtItems = std::unordered_set<int>(pItems.second.begin(), pItems.second.end());
    std::cout << "partInd: " << partInd << " testItems: "
      << setIntersect(testItems, filtItems) << " ";
    auto countNndcg = bestModel.NDCGI(filtItems, invalidUsers, invalidItems,
                                        data.testMat);
    std::cout << countNndcg.first << " " << countNndcg.second << " " << std::endl;
  }
  std::cout << std::endl;


  std::cout << "Users Part: ";
  for (auto& pUsers: partUsers) {
    int partInd = pUsers.first;
    auto filtUsers = std::unordered_set<int>(pUsers.second.begin(), pUsers.second.end());
    std::cout << "partInd: " << partInd << " testUsers: "
      << setIntersect(testUsers, filtUsers) << " ";
    auto countNndcg = bestModel.NDCGU(filtUsers, invalidUsers,
        invalidItems, data.testMat);
    std::cout << countNndcg.first << " " << countNndcg.second << " " << std::endl;
  }
  std::cout << std::endl;

}


void quartileARHR(Model& bestModel, const Data& data,
    std::vector<std::pair<int, std::vector<int>>> partItems,
    std::vector<std::pair<int, std::vector<int>>> partUsers,
    std::unordered_set<int> invalidUsers, std::unordered_set<int> invalidItems) {

  auto rowColFreq = getRowColFreq(data.trainMat);
  auto userFreq   = rowColFreq.first;
  auto itemFreq   = rowColFreq.second;
  int nItems      = itemFreq.size();
  int nUsers      = userFreq.size();

  auto testItems = getColInd(data.testMat);
  auto testUsers = getRowInd(data.testMat);

  std::cout << "Items Part: ";
  for (auto& pItems: partItems) {
    int partInd = pItems.first;
    auto filtItems = std::unordered_set<int>(pItems.second.begin(), pItems.second.end());
    std::cout << "partInd: " << partInd << " testItems: "
      << setIntersect(testItems, filtItems) << std::endl;
    auto countNHR = bestModel.arHRI(data, filtItems, invalidUsers, invalidItems,
                                        data.testMat);
    std::cout << countNHR.first << " " << countNHR.second << " ";
  }
  std::cout << std::endl;


  std::cout << "Users Part: ";
  for (auto& pUsers: partUsers) {
    int partInd = pUsers.first;
    auto filtUsers = std::unordered_set<int>(pUsers.second.begin(), pUsers.second.end());
    std::cout << "partInd: " << partInd << " testUsers: "
      << setIntersect(testUsers, filtUsers) << std::endl;
    auto countNHR = bestModel.arHRU(data, filtUsers, invalidUsers,
        invalidItems, data.testMat);
    std::cout << countNHR.first << " " << countNHR.second << " ";
  }
  std::cout << std::endl;

}


void quartileHR(Model& bestModel, const Data& data,
    std::vector<std::pair<int, std::vector<int>>> partItems,
    std::vector<std::pair<int, std::vector<int>>> partUsers,
    std::unordered_set<int> invalidUsers, std::unordered_set<int> invalidItems) {

  auto rowColFreq = getRowColFreq(data.trainMat);
  auto userFreq   = rowColFreq.first;
  auto itemFreq   = rowColFreq.second;
  int nItems      = itemFreq.size();
  int nUsers      = userFreq.size();

  auto testItems = getColInd(data.testMat);
  auto testUsers = getRowInd(data.testMat);

  std::cout << "Items Part: ";
  for (auto& pItems: partItems) {
    int partInd = pItems.first;
    auto filtItems = std::unordered_set<int>(pItems.second.begin(), pItems.second.end());
    std::cout << "partInd: " << partInd << " testItems: "
      << setIntersect(testItems, filtItems) << std::endl;
    auto countNHR = bestModel.hitRateI(data, filtItems, invalidUsers, invalidItems,
                                        data.testMat);
    std::cout << countNHR.first << " " << countNHR.second << " ";
  }
  std::cout << std::endl;


  std::cout << "Users Part: ";
  for (auto& pUsers: partUsers) {
    int partInd = pUsers.first;
    auto filtUsers = std::unordered_set<int>(pUsers.second.begin(), pUsers.second.end());
    std::cout << "partInd: " << partInd << " testUsers: "
      << setIntersect(testUsers, filtUsers) << std::endl;
    auto countNHR = bestModel.hitRateU(data, filtUsers, invalidUsers,
        invalidItems, data.testMat);
    std::cout << countNHR.first << " " << countNHR.second << " ";
  }
  std::cout << std::endl;

}


void quartileRMSEs(Model& bestModel, const Data& data,
    std::vector<std::pair<int, std::vector<int>>> partItems,
    std::vector<std::pair<int, std::vector<int>>> partUsers,
    std::unordered_set<int> invalidUsers, std::unordered_set<int> invalidItems) {

  auto rowColFreq = getRowColFreq(data.trainMat);
  auto userFreq   = rowColFreq.first;
  auto itemFreq   = rowColFreq.second;
  int nItems      = itemFreq.size();
  int nUsers      = userFreq.size();

  std::cout << std::endl;
  std::cout << "Train RMSE: " << bestModel.RMSE(data.trainMat, invalidUsers,
      invalidItems) << std::endl;
  std::cout << "Test RMSE: " << bestModel.RMSE(data.testMat, invalidUsers,
      invalidItems) << std::endl;
  std::cout << "Val RMSE: " << bestModel.RMSE(data.valMat, invalidUsers,
      invalidItems) << std::endl;

  std::cout << "Test RMSE: " << std::endl;
  std::cout << "Items Part: ";
  for (auto& pItems: partItems) {
    int partInd = pItems.first;
    auto filtItems = std::unordered_set<int>(pItems.second.begin(), pItems.second.end());
    auto countNRMSE = bestModel.RMSE(data.testMat, filtItems, invalidUsers, invalidItems);
    std::cout << countNRMSE.first << " " << countNRMSE.second << " ";
  }
  std::cout << std::endl;

  std::cout << "Users Part: ";
  for (auto& pUsers: partUsers) {
    int partInd = pUsers.first;
    auto filtUsers = std::unordered_set<int>(pUsers.second.begin(), pUsers.second.end());
    auto countNRMSE = bestModel.RMSEU(data.testMat, filtUsers, invalidUsers, invalidItems);
    std::cout << countNRMSE.first << " " << countNRMSE.second << " ";
  }
  std::cout << std::endl;

  std::cout << "Validation RMSE: " << std::endl;
  std::cout << "Items Part: ";
  for (auto& pItems: partItems) {
    int partInd = pItems.first;
    auto filtItems = std::unordered_set<int>(pItems.second.begin(), pItems.second.end());
    auto countNRMSE = bestModel.RMSE(data.valMat, filtItems, invalidUsers, invalidItems);
    std::cout << countNRMSE.first << " " << countNRMSE.second << " ";
  }
  std::cout << std::endl;

  std::cout << "Users Part: ";
  for (auto& pUsers: partUsers) {
    int partInd = pUsers.first;
    auto filtUsers = std::unordered_set<int>(pUsers.second.begin(), pUsers.second.end());
    auto countNRMSE = bestModel.RMSEU(data.valMat, filtUsers, invalidUsers, invalidItems);
    std::cout << countNRMSE.first << " " << countNRMSE.second << " ";
  }
  std::cout << std::endl;

}


ModelMF loadModel(int nUsers, int nItems, int rank, std::string modelPref) {
  std::cout << "\nmodelPref: " << modelPref << std::endl;
  ModelMF m1(nUsers, nItems, rank);
  std::string uFacName = modelPref + ".umat";
  std::string iFacName = modelPref + ".imat";
  std::cout << "uFac: " << uFacName.c_str() << " iFac: " << iFacName.c_str() << std::endl;
  readMat(m1.uFac, nUsers, rank, uFacName.c_str());
  readMat(m1.iFac, nItems, rank, iFacName.c_str());
  return m1;
}


void diffModelRMSEs(int nUsers, int nItems, std::vector<int>& ranks,
    std::vector<std::string> modelPrefs,
    const Data& data,
    std::vector<std::pair<int, std::vector<int>>> partItems,
    std::vector<std::pair<int, std::vector<int>>> partUsers,
    std::unordered_set<int> invalidUsers, std::unordered_set<int> invalidItems) {

  std::cout << "nusers: " << nUsers << " nItems: " << nItems << std::endl;

  std::vector<ModelMF> iModels;
  ModelMF m1 = loadModel(nUsers, nItems, ranks[0], modelPrefs[0]);
  iModels.push_back(m1);
  ModelMF m2 = loadModel(nUsers, nItems, ranks[1], modelPrefs[1]);
  iModels.push_back(m2);
  ModelMF m3 = loadModel(nUsers, nItems, ranks[2], modelPrefs[2]);
  iModels.push_back(m3);
  ModelMF m4 = loadModel(nUsers, nItems, ranks[3], modelPrefs[3]);
  iModels.push_back(m4);

  std::vector<ModelMF> uModels;
  ModelMF m5 = loadModel(nUsers, nItems, ranks[4], modelPrefs[4]);
  uModels.push_back(m5);
  ModelMF m6 = loadModel(nUsers, nItems, ranks[5], modelPrefs[5]);
  uModels.push_back(m6);
  ModelMF m7 = loadModel(nUsers, nItems, ranks[6], modelPrefs[6]);
  uModels.push_back(m7);
  ModelMF m8 = loadModel(nUsers, nItems, ranks[7], modelPrefs[7]);
  uModels.push_back(m8);

  quartileRMSEs(m1, data, partItems, partUsers, invalidUsers, invalidItems);

  double se = 0;
  int nnz = 0;

  auto trainMat = data.trainMat;
  auto testMat = data.testMat;
  auto rowColFreq = getRowColFreq(data.trainMat);
  auto userFreq = rowColFreq.first;
  auto itemFreq = rowColFreq.second;

  std::map<int, int> iPartMap, uPartMap;
  for (int i = 0; i < partItems.size(); i++) {
    int partInd = partItems[i].first;
    for (auto& item: partItems[i].second) {
      iPartMap[item] = partInd;
    }
  }

  for (int i = 0; i < partUsers.size(); i++) {
    int partInd = partUsers[i].first;
    for (auto& user: partUsers[i].second) {
      uPartMap[user] = partInd;
    }
  }

  std::vector<double> itemSE(4, 0), userSE(4, 0);
  std::vector<int> itemCount(4, 0), userCount(4, 0);


  for (int u = 0; u < testMat->nrows; u++) {

    if (invalidUsers.count(u) > 0) { continue; }

    for (int ii = testMat->rowptr[u]; ii < testMat->rowptr[u+1]; ii++) {
      int item = testMat->rowind[ii];
      if (item >= trainMat->ncols || invalidItems.count(item) > 0) {
        continue;
      }
      float rating = testMat->rowval[ii];
      double diff = 0;
      if (userFreq[u] < itemFreq[item]) {
        float rat_est = uModels[3 - uPartMap[u]].estRating(u, item);
        diff = (rat_est - rating)*(rat_est - rating);
      } else {
        float rat_est = iModels[3 - iPartMap[item]].estRating(u, item);
        diff = (rat_est - rating)*(rat_est - rating);
      }
      se += diff;
      itemSE[iPartMap[item]] += diff;
      itemCount[iPartMap[item]] += 1;
      userSE[uPartMap[u]] += diff;
      userCount[uPartMap[u]] += 1;
      nnz++;
    }
  }

  std::cout << "overall RMSE: " << std::sqrt(se/nnz) << std::endl;
  std::cout << "item RMSEs: " << std::endl;
  for (int i = 0; i < 4; i++) {
    std::cout << std::sqrt(itemSE[i]/itemCount[i]) << " ";
  }
  std::cout << std::endl;

  std::cout << "item count: " << std::endl;
  for (int i = 0; i < 4; i++) {
    std::cout << itemCount[i] << " ";
  }
  std::cout << std::endl;

  std::cout << "user RMSEs: " << std::endl;
  for (int i = 0; i < 4; i++) {
    std::cout << std::sqrt(userSE[i]/userCount[i]) << " ";
  }
  std::cout << std::endl;
  for (int i = 0; i < 4; i++) {
    std::cout << userCount[i] << " ";
  }
  std::cout << std::endl;

}


void computeFreqRMSEsAdapRank(Data& data, Params& params,
    std::vector<int>& userRankMap, std::vector<int>& itemRankMap) {

  std::cout << "\nCreating full model...";
  ModelMF fullModel(params, params.seed);
  //ModelMFBias fullModel(params, params.seed);
  //ModelDropoutMFBias fullModel(params, params.seed);
  //svdFrmSvdlibCSR(data.trainMat, fullModel.facDim, fullModel.uFac,
  //    fullModel.iFac, false);

  //load previously learned factors
  fullModel.loadFacs(params.prefix);
  //fullModel.load(params.prefix);

  //get number of ratings per user and item, i.e. frequency
  auto rowColFreq = getRowColFreq(data.trainMat);
  auto userFreq = rowColFreq.first;
  auto itemFreq = rowColFreq.second;

  auto usersMeanStd = meanStdDev(userFreq);
  auto itemsMeanStd = meanStdDev(itemFreq);

  std::cout << "user freq mean: " << usersMeanStd.first << " std: "
    << usersMeanStd.second << std::endl;
  std::cout << "item freq mean: " << itemsMeanStd.first << " std: "
    << itemsMeanStd.second << std::endl;

  std::unordered_set<int> invalidUsers;
  std::unordered_set<int> invalidItems;
  std::string modelSign = fullModel.modelSignature();
  std::cout << "\nModel sign: " << modelSign << std::endl;

  std::string prefix = std::string(params.prefix) + "_" + modelSign + "_invalUsers.txt";
  std::vector<int> invalUsersVec = readVector(prefix.c_str());
  prefix = std::string(params.prefix) + "_" + modelSign + "_invalItems.txt";
  std::vector<int> invalItemsVec = readVector(prefix.c_str());

  for (auto v: invalUsersVec) {
    invalidUsers.insert(v);
  }

  for (auto v: invalItemsVec) {
    invalidItems.insert(v);
  }

  //ModelMFBias bestModel(fullModel);
  Model& bestModel = fullModel;

  //std::cout << "\nStarting model train...";
  //fullModel.train(data, bestModel, invalidUsers, invalidItems);
  std::cout << "\nTrain RMSE: " << bestModel.RMSE(data.trainMat, invalidUsers, invalidItems);
  std::cout << "\nTest RMSE: " << bestModel.RMSE(data.testMat, invalidUsers, invalidItems);
  std::cout << "\nVal RMSE: " << bestModel.RMSE(data.valMat, invalidUsers, invalidItems);
  //std::cout << "\nFull RMSE: " << bestModel.fullLowRankErr(data, invalidUsers, invalidItems);
  std::cout << std::endl;

  std::cout << "No. of invalid users: " << invalidUsers.size() << std::endl;
  std::cout << "No. of invalid items: " << invalidItems.size() << std::endl;

  //order item in decreasing order of frequency
  std::vector<std::pair<int, double>> itemFreqPairs;
  for (int i = 0; i < itemFreq.size(); i++) {
    itemFreqPairs.push_back(std::make_pair(i, itemFreq[i]));
  }
  //comparison to sort in decreasing order
  std::sort(itemFreqPairs.begin(), itemFreqPairs.end(), descComp);

  int nSampUsers = 5000;
  int nUsers = data.trainMat->nrows;
  int nItems = data.trainMat->ncols;

  //get filtered items corresponding to head items
  //auto headItems = getHeadItems(data.trainMat, 0.2);

  std::vector<float> freqPcs = {1.0, 5.0, 10, 25, 50};
  for (auto&& filtPc: freqPcs) {
    std::unordered_set<int> filtItems, notFiltItems;
    float filtSzThresh = (filtPc/100.0)*((float)itemFreqPairs.size());
    //std::cout << "No. of top " << filtPc << " % freq items: " <<  filtSzThresh << std::endl;
    for (auto&& pair: itemFreqPairs) {
      if (filtItems.size() <= filtSzThresh) {
        filtItems.insert(pair.first);
      } else {
        notFiltItems.insert(pair.first);
      }
    }
    auto countNRMSE = bestModel.RMSE(data.testMat, filtItems, invalidUsers, invalidItems);
    auto countNRMSE2 = bestModel.RMSE(data.testMat, notFiltItems, invalidUsers, invalidItems);
    auto hiloNormFilt = bestModel.hiLoNorms(filtItems);
    auto hiloNormNotFilt = bestModel.hiLoNorms(notFiltItems);
    std::cout << "FiltPc: " << (int)filtPc << " " << countNRMSE.first << " " << countNRMSE.second
      << " " << countNRMSE2.first << " " <<countNRMSE2.second
      << " " << countNRMSE.first + countNRMSE2.first << std::endl;
    std::cout << "Filt norm hi: " << hiloNormFilt.first << " " << hiloNormFilt.second << std::endl;
    std::cout << "Not-Filt norm hi: " << hiloNormNotFilt.first << " " << hiloNormNotFilt.second << std::endl;

  }

  std::vector<float> maxFreqs = {5, 10, 15, 20, 30, 50, 75, 100, 150, 200};
  for (auto&& maxFreq: maxFreqs) {
    std::unordered_set<int> freqItems, notFreqItems;
    for (auto&& pair: itemFreqPairs) {
      if (pair.second < maxFreq) {
        notFreqItems.insert(pair.first);
      } else {
        freqItems.insert(pair.first);
      }
    }
    auto countNRMSE = bestModel.RMSE(data.testMat, freqItems, invalidUsers, invalidItems);
    auto countNRMSE2 = bestModel.RMSE(data.testMat, notFreqItems, invalidUsers, invalidItems);
    std::cout << "MaxFreq: " << (int)maxFreq << " " << countNRMSE.first << " "
      << countNRMSE.second << " " << countNRMSE2.first << " " << countNRMSE2.second
      << " " << countNRMSE.first + countNRMSE2.first << std::endl;
    auto hiloNormFreq = bestModel.hiLoNorms(freqItems);
    auto hiloNormNotFreq = bestModel.hiLoNorms(notFreqItems);
    std::cout << "Freq norm hi: " << hiloNormFreq.first << " " << hiloNormFreq.second << std::endl;
    std::cout << "Not-Freq norm hi: " << hiloNormNotFreq.first << " " << hiloNormNotFreq.second << std::endl;
  }

  std::map<int, std::vector<int>> rankItems;
  for (int item = 0; item < itemRankMap.size(); item++) {
    if (!rankItems.count(itemRankMap[item])) {
      rankItems[itemRankMap[item]] = std::vector<int>();
    }
    rankItems[itemRankMap[item]].push_back(item);
  }

  for (auto& p_rankItems: rankItems) {
    auto rank = p_rankItems.first;
    auto items = p_rankItems.second;
    auto setItems = std::unordered_set<int>(items.begin(), items.end());
    auto countNRMSE = bestModel.RMSE(data.testMat, setItems, invalidUsers, invalidItems);
    std::cout << "Items Rank: " << rank << " " << countNRMSE.first << " "
      << countNRMSE.second << std::endl;
  }

  std::map<int, std::vector<int>> rankUsers;
  for (int user = 0; user < userRankMap.size(); user++) {
    if (!rankUsers.count(userRankMap[user])) {
      rankUsers[userRankMap[user]] = std::vector<int>();
    }
    rankUsers[userRankMap[user]].push_back(user);
  }

  for (auto& p_rankUsers: rankUsers) {
    auto rank = p_rankUsers.first;
    auto users = p_rankUsers.second;
    auto setUsers = std::unordered_set<int>(users.begin(), users.end());
    auto countNRMSE = bestModel.RMSEU(data.testMat, setUsers, invalidUsers, invalidItems);
    std::cout << "Users Rank: " << rank << " " << countNRMSE.first << " "
      << countNRMSE.second << std::endl;
  }

}


void transformBinData(Data& data, Params& params) {
  std::cout << "\nCreating original model to transform binary ratings..." << std::endl;
  ModelMF origModel(params, params.origUFacFile, params.origIFacFile,
      params.seed);
  origModel.updateMatWRatingsGaussianNoise(data.trainMat);
  gk_csr_CreateIndex(data.trainMat, GK_CSR_COL);
  origModel.updateMatWRatingsGaussianNoise(data.testMat);
  gk_csr_CreateIndex(data.testMat, GK_CSR_COL);
  origModel.updateMatWRatingsGaussianNoise(data.valMat);
  gk_csr_CreateIndex(data.valMat, GK_CSR_COL);
}


void setAdapRank(std::vector<int>& rankMap,
    std::vector<std::pair<int, std::vector<int>>>& partItems,
    std::vector<std::pair<int, double>>& freqPairs, int facDim) {
  int nItems = freqPairs.size();
  int currFac = facDim;
  int i = 0, partInd = 0;
  while (i < nItems) {
    int endItem = i + 0.25*((float)nItems);
    if (endItem > nItems || partInd == 3) {
      endItem = nItems;
    }
    std::cout << "start: " << i << " end: " << endItem << " currFac: "
      << currFac << std::endl;
    std::vector<int> pItems;
    for (int item = i; item < endItem; item++) {
      rankMap[freqPairs[item].first] = currFac;
      pItems.push_back(freqPairs[item].first);
    }
    partItems.push_back(make_pair(partInd, pItems));
    currFac = currFac/2;
    if (0 == currFac) {
      currFac = 1;
    }
    i = endItem;
    partInd++;
  }
}


void getUserItemRankMap(
    const Data& data, const Params& params,
    std::vector<std::pair<int, std::vector<int>>>& partItems,
    std::vector<std::pair<int, std::vector<int>>>& partUsers,
    std::vector<int>& itemRank, std::vector<int>& userRank) {

  auto rowColFreq = getRowColFreq(data.trainMat);
  auto userFreq = rowColFreq.first;
  auto itemFreq = rowColFreq.second;
  int nItems = itemFreq.size();
  int nUsers = userFreq.size();
  userRank = std::vector<int>(nUsers, 0);
  itemRank = std::vector<int>(nItems, 0);

  //order item in decreasing order of frequency
  std::vector<std::pair<int, double>> itemFreqPairs;
  for (int i = 0; i < nItems; i++) {
    itemFreqPairs.push_back(std::make_pair(i, itemFreq[i]));
  }
  //comparison to sort in decreasing order
  std::sort(itemFreqPairs.begin(), itemFreqPairs.end(), descComp);
  setAdapRank(itemRank, partItems, itemFreqPairs, params.facDim);

  //order users in decreasing order of frequency
  std::vector<std::pair<int, double>> userFreqPairs;
  for (int i = 0; i < nUsers; i++) {
    userFreqPairs.push_back(std::make_pair(i, userFreq[i]));
  }
  //comparison to sort in decreasing order
  std::sort(userFreqPairs.begin(), userFreqPairs.end(), descComp);
  setAdapRank(userRank, partUsers, userFreqPairs, params.facDim);

}


void setPc(std::vector<double>& indFreq, std::vector<double>& indRank) {

  //order item in decreasing order of frequency
  std::vector<std::pair<int, double>> indFreqPairs;
  for (int i = 0; i < indFreq.size(); i++) {
    indFreqPairs.push_back(std::make_pair(i, indFreq[i]));
  }
  //comparison to sort in decreasing order
  std::sort(indFreqPairs.begin(), indFreqPairs.end(), descComp);

  for (int i = 0; i < indFreq.size(); i++) {
    int item = indFreqPairs[i].first;
    double pc = double(indFreq.size() - i)/double(indFreq.size());
    indRank[item] = pc;
  }

}


void getUserItemRankMapPc(
    const Data& data, const Params& params,
    std::vector<double>& itemRank, std::vector<double>& userRank) {

  auto rowColFreq = getRowColFreq(data.trainMat);
  auto userFreq = rowColFreq.first;
  auto itemFreq = rowColFreq.second;
  int nItems = itemFreq.size();
  int nUsers = userFreq.size();
  userRank = std::vector<double>(nUsers, 0);
  itemRank = std::vector<double>(nItems, 0);

  setPc(userFreq, userRank);
  setPc(itemFreq, itemRank);
}


gk_csr_t** splitValMat(gk_csr_t* valMat, int seed) {

  int nnz = getNNZ(valMat);
  int* color = (int*) malloc(sizeof(int)*nnz);
  memset(color, 0, sizeof(int)*nnz);

  std::mt19937 mt(seed);
  std::uniform_int_distribution<int> nnzDist(0, nnz-1);

  int k;
  int sumColor = 0;
  while (sumColor < nnz/2) {
    k = nnzDist(mt);
    if (!color[k]) {
      color[k] = 1;
      sumColor++;
    }
  }

  //split the matrix based on color
  gk_csr_t** mats = gk_csr_Split(valMat, color);

  int sampNNZ = getNNZ(mats[1]);
  std::cout << "\nmats[0] NNZ: " << getNNZ(mats[0])
    << " mats[1] NNZ: " << getNNZ(mats[1]) << std::endl;

  free(color);
  return mats;
}


int main(int argc , char* argv[]) {
  
  /*
  gk_csr_t* mat = gk_csr_Read(argv[1], GK_CSR_FMT_CSR, 1, 0);
  writeBinarizedMat(mat, 3, argv[2]);
  return 0;
  */

  /*
  //partition the given matrix into train test val
  gk_csr_t *mat = gk_csr_Read(argv[1], GK_CSR_FMT_CSR, GK_CSR_IS_VAL, 0);
  //writeTrainTestValMat(mat, argv[2], argv[3], argv[4], 0.1, 0.1, atoi(argv[5]));
  writeBinarizedTrainValTest(mat, atoi(argv[2]), std::string(argv[3]),
      atoi(argv[4]));
  return 0;
  */

  /*
  gk_csr_t *mat1 = gk_csr_Read(argv[1], GK_CSR_FMT_CSR, 1, 0);
  gk_csr_t *mat2 = gk_csr_Read(argv[2], GK_CSR_FMT_CSR, GK_CSR_IS_VAL, 0);
  std::cout << "train test mat match: " <<  compMat(mat1, mat2) << std::endl;
  return 0;
  */

  /*
  gk_csr_t *mat = gk_csr_Read(argv[1], GK_CSR_FMT_CSR, 0, 0);
  int nnz = getNNZ(mat);

  auto rowColFreq = getRowColFreq(mat);
  auto userFreq = rowColFreq.first;
  auto itemFreq = rowColFreq.second;

  auto usersMeanStd = meanStdDev(userFreq);
  auto itemsMeanStd = meanStdDev(itemFreq);

  std::cout << "nnz: " << nnz << std::endl;
  std::cout << "user freq mean: " << usersMeanStd.first << " std: "
    << usersMeanStd.second << std::endl;
  std::cout << "item freq mean: " << itemsMeanStd.first << " std: "
    << itemsMeanStd.second << std::endl;
  return 0;
  */

  //get passed parameters
  Params params = parse_cmd_line(argc, argv);
  Data data (params);
  params.nUsers = data.nUsers;
  params.nItems = data.nItems;
  params.display();

  /*
  auto nzUsersItems = getNZUserItem(data.trainMat);
  int nnz = getNNZ(data.trainMat);
  std::cout << "nUsers with ratings: " << nzUsersItems.first.size() << std::endl;
  std::cout << "nItems with ratings: " << nzUsersItems.second.size() << std::endl;
  std::cout << "nnz: " << nnz << std::endl;
  writeFiltRandMatCSR("ranSubMat.csr", data.origUFac, data.origIFac,
      data.facDim, params.seed, nzUsersItems.first, nzUsersItems.second, nnz);
  return 0;
  */

  //initialize seed
  std::srand(params.seed);

   /*
  auto meanVar = getMeanVar(data.origUFac, data.origIFac, data.origFacDim,
      data.nUsers, data.nItems);

  std::cout << "\nmean = " << meanVar.first << " variance = "
    << meanVar.second << std::endl;
  */

  //auto headItems = getHeadItems(data.trainMat, 0.1);
  //writeTailTestMat(data.testMat, "nf_480189x17772.tail.test.5.csr", headItems);

  /*
  std::string matPre = params.prefix;//"each_61265x1623_10";
  std::string suff = ".syn.ind.csr";

  writeCSRWSparsityStructure(data.trainMat,
      (matPre + suff).c_str(),
      data.origUFac, data.origIFac, params.facDim);
  data.trainMat = gk_csr_Read((char*)(matPre + suff).c_str(), GK_CSR_FMT_CSR,
      GK_CSR_IS_VAL, 0);
  gk_csr_CreateIndex(data.trainMat, GK_CSR_COL);
  writeTrainTestValMat(data.trainMat,
      (matPre + ".train" + suff).c_str(),
      (matPre + ".test" + suff).c_str(),
      (matPre + ".val" + suff).c_str(),
      0.1, 0.1, params.seed);

  int nnz = getNNZ(data.trainMat);

#pragma omp parallel for
  for (int randSeed = 1; randSeed < 11; randSeed++) {
    std::string suff = "." + std::to_string(randSeed) + ".syn.rand.ind.csr";
    writeRandMatCSR((matPre + suff).c_str(), data.origUFac,
        data.origIFac, params.facDim, randSeed, nnz);
    gk_csr_t* randMat = gk_csr_Read((char*)(matPre + suff).c_str(),
        GK_CSR_FMT_CSR, GK_CSR_IS_VAL, 0);
    gk_csr_CreateIndex(randMat, GK_CSR_COL);
    writeTrainTestValMat(randMat,
      (matPre + ".train" + suff).c_str(),
      (matPre + ".test" + suff).c_str(),
      (matPre + ".val" + suff).c_str(),
      0.1, 0.1, params.seed);
  }
  exit(0);
  */
  //writeBlkDiagJoinedCSR("", "", "");

  //computeConfScoresFrmModel(data, params);
  //computeConf(data, params);
  //computeBucksEstFullModel(data, params);
  //computeBucksFrmFullModel(data, params);

  /*
  writeSubSampledMat(data.trainMat,
      "ratings_229060x26779_25_0.6.syn.csr2", 0.6, params.seed);
  return 0;
  */

  //writeTrainTestMat(data.trainMat,  "",
  //   "", ,  params.seed);
  //writeItemSimMat(data.trainMat, "ratings_26779x26779_25.syn.trainItems.metis");
  //writeItemSimMatNonSymm(data.trainMat,
  //    "ratings_26779x26779_25.syn.trainItems.nonsym.metis");

  //std::string graphFName = params.prefix + std::string(".train.jacSim.metis");
  //writeItemJaccSimMat(data.trainMat, graphFName.c_str());

  //writeItemJaccSimFrmCorat(data.trainMat, data.graphMat,
  //    "ratings_26779x26779_25.syn.trainItems.jacSim2.metis");
  //writeCoRatings(data.trainMat, "y_u2_i34_100Kx50K.train.coRatings");
   //ModelMF mfModel(params, params.initUFacFile,
  //    params.initIFacFile, params.seed);

  bool isUISorted = checkIfUISorted(data.trainMat);
  std::cout << "ifUISorted: " << isUISorted << std::endl;

  if (!GK_CSR_IS_VAL) {
    transformBinData(data, params);
    //writeTriplets(data.trainMat, "train.triplet");
    //writeTriplets(data.testMat, "test.triplet");
    //writeTriplets(data.valMat, "val.triplet");
    //exit(0);
  }

  /*

  std::string ans;
  std::cout << "Want to train? ";
  std::getline(std::cin, ans);
  std::cout << "you entered: " << ans << std::endl;
  if (!(ans.compare("yes") == 0 || ans.compare("y") == 0)) {
    return 0;
  }
  */

  //get number of ratings per user and item, i.e. frequency
  std::vector<std::pair<int, std::vector<int>>> partItems, partUsers;
  std::vector<int> userRankMap, itemRankMap;
  std::vector<double> userRankPc, itemRankPc;

  getUserItemRankMap(data, params, partItems, partUsers, itemRankMap,
      userRankMap);
  getUserItemRankMapPc(data, params, itemRankPc, userRankPc);
  std::vector<int> ranks = {1, params.facDim};

  auto rowColFreq = getRowColFreq(data.trainMat);
  auto userFreq = rowColFreq.first;
  auto itemFreq = rowColFreq.second;

  /*
  std::vector<std::string> modelPrefs = {
    "sgdpar_1_0.1_147612X48794_1_0.100000_0.100000_0.001000",
    "sgdpar_1_0.1_147612X48794_1_0.100000_0.100000_0.001000",
    "sgdpar_15_0.1_147612X48794_15_0.100000_0.100000_0.001000",
    "sgdpar_15_0.1_147612X48794_15_0.100000_0.100000_0.001000",

    "sgdpar_1_0.1_147612X48794_1_0.100000_0.100000_0.001000",
    "sgdpar_1_0.1_147612X48794_1_0.100000_0.100000_0.001000",
    "sgdpar_1_0.1_147612X48794_1_0.100000_0.100000_0.001000",
    "sgdpar_15_0.1_147612X48794_15_0.100000_0.100000_0.001000",
  };
  std::vector<std::string> modelPrefs2 = {
    "sgdpar_1_0.1_147612X48794_1_0.100000_0.100000_0.001000",
    "sgdpar_1_0.1_147612X48794_1_0.100000_0.100000_0.001000",
    "sgdpar_1_0.1_147612X48794_1_0.100000_0.100000_0.001000",
    "sgdpar_1_0.1_147612X48794_1_0.100000_0.100000_0.001000",

    "sgdpar_1_0.1_147612X48794_1_0.100000_0.100000_0.001000",
    "sgdpar_1_0.1_147612X48794_1_0.100000_0.100000_0.001000",
    "sgdpar_1_0.1_147612X48794_1_0.100000_0.100000_0.001000",
    "sgdpar_1_0.1_147612X48794_1_0.100000_0.100000_0.001000"
  };

  std::vector<int> ranks2 = {1, 1, 15, 15, 1, 1, 1, 15};
  //std::vector<int> ranks2 = {1, 1, 1, 1, 1, 1, 1, 1};

  auto invalUVec = readVector("sgdpar_1_0.1_147612X48794_1_0.100000_0.100000_0.001000_invalUsers.txt");
  std::unordered_set<int> invalidUsers2(invalUVec.begin(), invalUVec.end());

  auto invalIVec  = readVector("sgdpar_1_0.1_147612X48794_1_0.100000_0.100000_0.001000_invalItems.txt");
  std::unordered_set<int> invalidItems2(invalIVec.begin(), invalIVec.end());

  diffModelRMSEs(data.trainMat->nrows, data.trainMat->ncols, ranks2,
      modelPrefs, data, partItems, partUsers, invalidUsers2, invalidItems2);

  //quartileRMSEs(bestModel2, data, partItems, partUsers, invalidUsers, invalidItems);
  return 0;
  */

  //ModelMFBias mfModel(params, params.seed);
  //ModelDropoutMF mfModel(params, params.seed, userRankMap, itemRankMap, ranks);

  /*
  ModelIncrement mfModel(params, params.seed);
  gk_csr_t** mats = splitValMat(data.valMat, params.seed);
  gk_csr_Free(&data.valMat);
  data.valMat = mats[0];
  gk_csr_CreateIndex(data.valMat, GK_CSR_COL);
  std::cout << "val nusers: " << data.valMat->nrows << " nItems: " << data.valMat->ncols << std::endl;
  data.graphMat = mats[1];
  gk_csr_CreateIndex(data.graphMat, GK_CSR_COL);

  std::cout << "graph nusers: " << data.graphMat->nrows << " nItems: " << data.graphMat->ncols << std::endl;
  */

  ModelMFBPR mfModel(params, params.seed);
  //ModelPoissonDropout mfModel(params, params.seed, userRankPc, itemRankPc,
  //    userFreq, itemFreq);
  //ModelBPRPoissonDropout mfModel(params, params.seed, userRankPc, itemRankPc,
  //    userFreq, itemFreq);
  //ModelInvPopMF mfModel(params, params.seed, userFreq, itemFreq);

  //initialize model with svd
  //svdFrmSvdlibCSREig(data.trainMat, mfModel.facDim, mfModel.uFac, mfModel.iFac, false);

  //initialize MF model with last learned model if any
  //mfModel.initInfreqFactors(params, data);
  //mfModel.loadFacs(params.prefix);

  std::unordered_set<int> invalidUsers;
  std::unordered_set<int> invalidItems;

  /*
  ModelMF bestModel(mfModel);
  std::cout << "\nStarting model train...";
  if (FLAGS_method == "ccd++") {
    mfModel.trainCCDPPFreqAdap(data, bestModel, invalidUsers, invalidItems);
  } else if (FLAGS_method == "ccd") {
    if (isUISorted) {
      mfModel.trainCCD(data, bestModel, invalidUsers, invalidItems);
    } else {
      std::cerr << "Need sorted train mat for ccd." << std::endl;
    }
  } else if (FLAGS_method == "als") {
    mfModel.trainALS(data, bestModel, invalidUsers, invalidItems);
  } else if (FLAGS_method == "hogsgd")  {
    mfModel.hogTrain(data, bestModel, invalidUsers, invalidItems);
  } else if (FLAGS_method == "sgdu") {
    mfModel.trainUShuffle(data, bestModel, invalidUsers, invalidItems);
  } else if (FLAGS_method == "sgdpar") {
    mfModel.trainSGDPar(data, bestModel, invalidUsers, invalidItems);
  } else if (FLAGS_method == "sgdparsvd") {
    mfModel.trainSGDParSVD(data, bestModel, invalidUsers, invalidItems);
  } else {
    mfModel.train(data, bestModel, invalidUsers, invalidItems);
  }
  */

  //ModelInvPopMF bestModel(mfModel);
  //ModelBPRPoissonDropout bestModel(mfModel);
  ModelMFBPR bestModel(mfModel);
  //ModelDropoutMF bestModel(mfModel);
  //ModelIncrement bestModel(mfModel);
  //ModelPoissonDropout bestModel(mfModel);
  //ModelDropoutMFBias bestModel(mfModel);
  //ModelMFBias bestModel(mfModel);

  std::cout << "Begin train..." << std::endl;
  mfModel.trainHogPosNeg(data, bestModel, invalidUsers, invalidItems);
 
  //mfModel.trainSGDProbOrderedPar(data, bestModel, invalidUsers, invalidItems);
  //mfModel.trainSGDProbPar(data, bestModel, invalidUsers, invalidItems);
  //mfModel.trainSGDOnlyOrderedPar(data, bestModel, invalidUsers, invalidItems);
  //mfModel.train(data, bestModel, invalidUsers, invalidItems);

 
  std::cout << "Test NDCG: "
    << bestModel.NDCG(invalidUsers, invalidItems, data.testMat)
    << std::endl;

  quartileNDCG(bestModel, data, partItems, partUsers, invalidUsers,
      invalidItems);
  /*
  std::cout << "Test ar hit rate: "
    << bestModel.arHR(data, invalidUsers, invalidItems, data.testMat)
    << std::endl;
  
  quartileARHR(bestModel, data, partItems, partUsers, invalidUsers,
      invalidItems);
  */

  /*
  bestModel.currRankMapU = mfModel.currRankMapU;
  bestModel.currRankMapI = mfModel.currRankMapI;
  */

  /*
  std::cout << "\nTrain RMSE: " << bestModel.RMSE(data.trainMat, invalidUsers,
      invalidItems);
  std::cout << "\nTest RMSE: " << bestModel.RMSE(data.testMat, invalidUsers,
      invalidItems);
  std::cout << "\nValidation RMSE: " << bestModel.RMSE(data.valMat, invalidUsers,
      invalidItems);

  std::string modelSign = bestModel.modelSignature();

  //write out invalid users
  std::string prefix = std::string(params.prefix) + "_" + modelSign + "_invalUsers.txt";
  //writeContainer(begin(invalidUsers), end(invalidUsers), prefix.c_str());

  //write out invalid items
  prefix = std::string(params.prefix) + "_" + modelSign + "_invalItems.txt";
  //writeContainer(begin(invalidItems), end(invalidItems), prefix.c_str());
  std::cout << std::endl << "**** Model parameters ****" << std::endl;
  mfModel.display();

  if (!FLAGS_origufac.empty()) {
    std::cout << "\nFull RMSE: " <<
      bestModel.fullLowRankErr(data, invalidUsers, invalidItems) << std::endl;
  }

  std::cout << std::endl;
  Model& bestModel2 = bestModel;
  //diffRankRMSEs(bestModel, data, userRankMap, itemRankMap, invalidUsers, invalidItems);
  */
  //quartileRMSEs(bestModel, data, partItems, partUsers, invalidUsers, invalidItems);

  //computeSampTopNFrmFullModel(data, params);
  //computeFreqRMSEsAdapRank(data, params, userRankMap, itemRankMap);

  //testTailLocRec(data, params);
  //testTailRec(data, params);
  //testRec(data, params);
  //computeHeadTailRMSE(data, params);

  //analyzeAccuracy(data, params);
  //analyzeAccuracySingleOrigModel(data, params);
  //compJaccSimAccuMeth(data, params);
  //compJaccSimAccuSingleOrigModel(data, params);
  //averageModels(data, params);
  //meanAndVarSameGroundAllUsers(data, params);
  //convertToBin(data, params);

  return 0;
}
