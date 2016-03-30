#include <iostream>
#include <cstdlib>
#include <future>
#include <chrono>
#include <thread>
#include "io.h"
#include "util.h"
#include "datastruct.h"
#include "modelMF.h"
#include "modelMFBias.h"
#include "confCompute.h"
#include "topBucketComp.h"
#include "longTail.h"

Params parse_cmd_line(int argc, char *argv[]) {
  
  if (argc < 21) {
    std::cout << "\nNot enough arguments";
    exit(0);
  }  
  
  Params params(atoi(argv[1]), atoi(argv[2]), atoi(argv[3]), atoi(argv[4]), 
      atoi(argv[5]), atoi(argv[6]),
      atof(argv[7]), atof(argv[8]), atof(argv[9]), atof(argv[10]), atof(argv[11]),
      argv[12], argv[13], argv[14], argv[15], argv[16], argv[17], argv[18], argv[19], 
      argv[20]);

  return params;
}


void knownLowRankEval(Data& data, Model& bestModel, Params& params) {

  //compute metrics
  double trainRMSE  = bestModel.RMSE(data.trainMat);
  double loRankRMSE = bestModel.fullLowRankErr(data);

  int D00_rowStart = 0;
  int D00_rowEnd   = 499;
  int D00_colStart = 0;
  int D00_colEnd   = 449;

  int D11_rowStart = 500;
  int D11_rowEnd = 999;
  int D11_colStart = 450;
  int D11_colEnd = 899;

  int S01_rowStart = 0;
  int S01_rowEnd = 499;
  int S01_colStart = 450;
  int S01_colEnd = 899;

  int S10_rowStart = 500;
  int S10_rowEnd = 999;
  int S10_colStart = 0;
  int S10_colEnd = 449;

  int Glu_rowStart = 0;
  int Glu_rowEnd = 999;
  int Glu_colStart = 900;
  int Glu_colEnd = 999;


  double D00RMSE    = bestModel.subMatKnownRankErr(data, D00_rowStart, D00_rowEnd, 
                                                    D00_colStart, D00_colEnd);
  double D11RMSE    = bestModel.subMatKnownRankErr(data, D11_rowStart, D11_rowEnd, 
                                                    D11_colStart, D11_colEnd);
  double S01RMSE    = bestModel.subMatKnownRankErr(data, S01_rowStart, S01_rowEnd, 
                                                    S01_colStart, S01_colEnd);
  double S10RMSE    = bestModel.subMatKnownRankErr(data, S10_rowStart, S10_rowEnd, 
                                                    S10_colStart, S10_colEnd);
  double GluRMSE    = bestModel.subMatKnownRankErr(data, Glu_rowStart, Glu_rowEnd, 
                                                    Glu_colStart, Glu_colEnd);

  std::cout << "\nD00RMSE: " << D00RMSE;
  std::cout << "\nD11RMSE: " << D11RMSE;
  std::cout << "\nS01RMSE: " << S01RMSE;
  std::cout << "\nS10RMSE: " << S10RMSE;
  std::cout << "\nGluRMSE: " << GluRMSE;


  std::cout.precision(5);
  std::cout<<"\nRE: " << std::fixed << params.facDim << " " << params.uReg << " " 
            << params.iReg << " " << params.rhoRMS << " " << params.alpha << " " 
            << trainRMSE << " " << loRankRMSE << " "
            << D00RMSE << " " << D11RMSE << " " << " " 
            << S01RMSE << " " << S10RMSE << " " << GluRMSE << " " 
            << meanRating(data.trainMat) << " " 
            << data.meanKnownSubMatRat(0, 999, 0, 999) << std::endl;
}


void knownLowRankGluEval(Data& data, Model& bestModel, Params& params) {

  //compute metrics
  double trainRMSE  = bestModel.RMSE(data.trainMat);
  double loRankRMSE = bestModel.fullLowRankErr(data);
  
  //TODO: pass in command line
  int gluSz = params.gluSz;

  int D00_rowStart = 0;
  int D00_rowEnd   = params.nUsers/2 - 1;
  int D00_colStart = 0;
  int D00_colEnd   = (params.nItems - gluSz)/2 - 1;

  std::cout << "\nD00: " << D00_rowStart << "," << D00_rowEnd << "," 
            << D00_colStart << "," << D00_colEnd;

  int D11_rowStart = D00_rowEnd + 1;
  int D11_rowEnd   = params.nUsers - 1;
  int D11_colStart = D00_colEnd + 1;
  int D11_colEnd   = params.nItems - gluSz -1;
  
  std::cout << "\nD11: " << D11_rowStart << "," << D11_rowEnd << "," 
            << D11_colStart << "," << D11_colEnd;

  int S01_rowStart = D00_rowStart;
  int S01_rowEnd   = D00_rowEnd;
  int S01_colStart = D11_colStart;
  int S01_colEnd   = D11_colEnd;

  std::cout << "\nS01: " << S01_rowStart << "," << S01_rowEnd << "," 
            << S01_colStart << "," << S01_colEnd;
  
  int S10_rowStart = D11_rowStart;
  int S10_rowEnd   = D11_rowEnd;
  int S10_colStart = D00_colStart;
  int S10_colEnd   = D00_colEnd;

  std::cout << "\nS10: " << S10_rowStart << "," << S10_rowEnd << "," 
            << S10_colStart << "," << S10_colEnd;
  
  int Glu_rowStart = D00_rowStart;
  int Glu_rowEnd   = D11_rowEnd;
  int Glu_colStart = D11_colEnd + 1;
  int Glu_colEnd   = params.nItems - 1;

  std::cout << "\nGlu: " << Glu_rowStart << "," << Glu_rowEnd << "," 
            << Glu_colStart << "," << Glu_colEnd;

  double D00RMSE  = bestModel.subMatKnownRankErr(data, D00_rowStart, D00_rowEnd, 
                                                    D00_colStart, D00_colEnd);
  double D11RMSE  = bestModel.subMatKnownRankErr(data, D11_rowStart, D11_rowEnd, 
                                                    D11_colStart, D11_colEnd);
  double S01RMSE  = bestModel.subMatKnownRankErr(data, S01_rowStart, S01_rowEnd, 
                                                    S01_colStart, S01_colEnd);
  double S10RMSE  = bestModel.subMatKnownRankErr(data, S10_rowStart, S10_rowEnd, 
                                                    S10_colStart, S10_colEnd);
  double GluRMSE  = bestModel.subMatKnownRankErr(data, Glu_rowStart, Glu_rowEnd, 
                                                    Glu_colStart, Glu_colEnd);

  std::cout << "\nD00RMSE: " << D00RMSE;
  std::cout << "\nD11RMSE: " << D11RMSE;
  std::cout << "\nS01RMSE: " << S01RMSE;
  std::cout << "\nS10RMSE: " << S10RMSE;
  std::cout << "\nGluRMSE: " << GluRMSE;


  std::cout.precision(5);
  std::cout<<"\nRE: " << std::fixed << params.facDim << " " << params.uReg << " " 
            << params.iReg << " " << params.rhoRMS << " " << params.alpha << " " 
            << trainRMSE << " " << loRankRMSE << " "
            << D00RMSE << " " << D11RMSE << " " << " " 
            << S01RMSE << " " << S10RMSE << " " << GluRMSE << " " 
            << meanRating(data.trainMat) << " " 
            << data.meanKnownSubMatRat(0, 999, 0, 999) << std::endl;
}


void knownLowRankEval2(Data& data, Model& bestModel, Params& params) {

  //compute metrics
  double trainRMSE  = bestModel.RMSE(data.trainMat);
  double loRankRMSE = bestModel.fullLowRankErr(data);

  std::cout.precision(5);
  std::cout<<"\nRE: " << std::fixed << params.facDim << " " << params.uReg << " " 
            << params.iReg << " " << params.rhoRMS << " " << params.alpha << " " 
            << trainRMSE << " " << loRankRMSE << " "
            << meanRating(data.trainMat) << " " 
            << data.meanKnownSubMatRat(0, 199, 0, 199) << std::endl;
}


void computeConf(Data& data, Params& params) {

  ModelMF fullModel(params, params.seed);
  svdFrmSvdlibCSR(data.trainMat, fullModel.facDim, fullModel.uFac, 
      fullModel.iFac); 
  
  int nThreads = 5;
  
  std::vector<std::thread> threads(nThreads);
  
  std::vector<std::shared_ptr<ModelMF>> trainModels;
  std::vector<std::shared_ptr<Model>> pbestModels;
  std::vector<std::unordered_set<int>> mInvalUsers (nThreads);
  std::vector<std::unordered_set<int>> mInvalItems (nThreads);

  for (int thInd = 0; thInd < nThreads; thInd++) {
    int seed = params.seed + thInd + 1;
    std::shared_ptr<ModelMF> p(new ModelMF(params, seed));
    p->uFac = fullModel.uFac;
    p->iFac = fullModel.iFac;
    trainModels.push_back(p);
    std::shared_ptr<Model> p2(new Model(params, seed));
    pbestModels.push_back(p2);
    //invoke training on thread
    std::cout << "\nInvoking thread: " << thInd << std::endl;  
    threads[thInd] = std::thread(&ModelMF::partialTrain, 
        trainModels[thInd], std::ref(data), 
        std::ref(*pbestModels[thInd]), std::ref(mInvalUsers[thInd]),
        std::ref(mInvalItems[thInd]));
  }

  //train full model in main thread
  ModelMF fullBestModel(fullModel);
  std::cout << "\nStarting full model train...";
  std::unordered_set<int> invalidUsers;
  std::unordered_set<int> invalidItems;
  fullModel.train(data, fullBestModel, invalidUsers, invalidItems);

  std::cout << "\nWaiting for threads to finish...";
  //wait for threads to finish
  std::for_each(threads.begin(), threads.end(), 
      std::mem_fn(&std::thread::join));

  std::cout << "\nFinished all threads.";

  std::cout << "\nModels save...";
  std::vector<Model> bestModels;
  //save the best models
  for (int thInd = 0; thInd < nThreads; thInd++) {
    int seed = params.seed + thInd + 1;
    std::string prefix(params.prefix);
    prefix = prefix + "_partial_" + std::to_string(seed); 
    pbestModels[thInd]->save(prefix);
    bestModels.push_back(*pbestModels[thInd]);
  }
  //save the best full model
  std::string prefix(params.prefix);
  prefix = prefix + "_full";
  fullBestModel.save(prefix);

  ModelMF origModel(params, params.seed);
  origModel.load(params.origUFacFile, params.origIFacFile);
  
  //find all invalid users
  for (int thInd = 0; thInd < nThreads; thInd++) {
    for (const auto& elem: mInvalUsers[thInd]) {
      invalidUsers.insert(elem);
    }
  }
  
  std::cout << "\nTotal invalid users: " << invalidUsers.size();
  //write out invalid users
  prefix = std::string(params.prefix) + "_invalUsers.txt";
  writeContainer(begin(invalidUsers), end(invalidUsers), prefix.c_str());
  
  //find all invalid items
  for (int thInd = 0; thInd < nThreads; thInd++) {
    for (const auto& elem: mInvalItems[thInd]) {
      invalidItems.insert(elem);
    }
  }

  std::cout << "\nTotal invalid items: " << invalidItems.size();
  //write out invalid items
  prefix = std::string(params.prefix) + "_invalItems.txt";
  writeContainer(begin(invalidItems), end(invalidItems), prefix.c_str());

  //get number of ratings per user and item, i.e. frequency
  auto rowColFreq = getRowColFreq(data.trainMat);
  auto userFreq = rowColFreq.first;
  auto itemFreq = rowColFreq.second;
  
  //compute confidence using the best models for 10 buckets
  std::vector<double> confRMSEs = confBucketRMSEsWInval(origModel, fullBestModel, 
      bestModels, params.nUsers, params.nItems, 10, invalidUsers, invalidItems);
  std::cout << "\nModel Confidence bucket RMSEs: ";
  dispVector(confRMSEs);
  prefix = std::string(params.prefix) + "_conf_bucket.txt";
  writeVector(confRMSEs, prefix.c_str());

  std::cout << "\nItem Freq confidence: ";
  std::vector<double> itemRMSEs = itemFreqBucketRMSEsWInVal(origModel, fullModel,
      params.nUsers, params.nItems, itemFreq, 
      10, invalidUsers, invalidItems);
  dispVector(itemRMSEs);
  prefix = std::string(params.prefix) + "_iFreq_bucket.txt";
  writeVector(itemRMSEs, prefix.c_str());
 
  //compute optimal confidence
  std::vector<double> optRMSEs = confOptBucketRMSEsWInVal(origModel, fullBestModel, 
    params.nUsers, params.nItems, 10, invalidUsers, invalidItems);
  prefix = std::string(params.prefix) + "_optconf_bucket.txt";
  writeVector(optRMSEs, prefix.c_str());
  std::cout << "\nOptimal confidence RMSEs:";
  dispVector(optRMSEs);

  //compute global page rank confidence
  //NOTE: using params.alpha as (1 - restartProb)
  std::vector<double> gprRMSEs = gprBucketRMSEsWInVal(origModel, fullBestModel, 
    params.nUsers, params.nItems, params.alpha, MAX_PR_ITER, 
    data.graphMat, 10, invalidUsers, invalidItems);
  prefix = std::string(params.prefix) + "_gprconf_" + std::to_string(params.alpha ) + "_bucket.txt";
  writeVector(gprRMSEs, prefix.c_str());
  std::cout << "\nGPR confidence RMSEs:";
  dispVector(gprRMSEs);

   
  //compute ppr confidence
  std::vector<double> pprRMSEs = pprBucketRMSEs(origModel, fullBestModel,
      params.nUsers, params.nItems, params.alpha, MAX_PR_ITER, data.graphMat, 10);
  /*
  std::vector<double> pprRMSEs = pprBucketRMSEsFrmPRWInVal(origModel, fullBestModel,
      params.nUsers, params.nItems, data.graphMat, 10, 
      "", invalidUsers, invalidItems);
   std::vector<double> pprRMSEs = pprBucketRMSEsWInVal(origModel, fullBestModel, 
    params.nUsers, params.nItems, params.alpha, MAX_PR_ITER, 
    data.graphMat, 10, invalidUsers, invalidItems);
  */
  
  prefix = std::string(params.prefix) + "_pprconf_"+ std::to_string(params.alpha ) + "_bucket.txt";
  writeVector(pprRMSEs, prefix.c_str());
  std::cout << "\nPPR confidence RMSEs:";
  dispVector(pprRMSEs);
  
}


void computeBucksFrmFullModel(Data& data, Params& params) {

  Model fullModel(params, params.seed);
  fullModel.load(params.initUFacFile, params.initIFacFile);
  Model origModel(params, params.seed);
  origModel.load(params.origUFacFile, params.origIFacFile);

  std::unordered_set<int> invalUsers;
  std::unordered_set<int> invalItems;

  //get number of ratings per user and item, i.e. frequency
  auto rowColFreq = getRowColFreq(data.trainMat);
  auto userFreq = rowColFreq.first;
  auto itemFreq = rowColFreq.second;
  
  std::cout << "\nOptimal confidence: ";
  std::vector<double> optRMSEs = confOptBucketRMSEsWInVal(origModel, fullModel,
      params.nUsers, params.nItems, 10, invalUsers, invalItems);
  dispVector(optRMSEs);
  std::string prefix = std::string(params.prefix) + "_opt_bucket.txt";
  writeVector(optRMSEs, prefix.c_str());

  std::cout << "\nGPR confidence: ";
  std::vector<double> gprRMSEs = gprBucketRMSEsWInVal(origModel, fullModel,
      params.nUsers, params.nItems, params.alpha, MAX_PR_ITER, data.graphMat, 
      10, invalUsers, invalItems);
  dispVector(gprRMSEs);
  prefix = std::string(params.prefix) + "_gpr_" + std::to_string(params.alpha) + "_bucket.txt";
  writeVector(gprRMSEs, prefix.c_str());

  std::cout << "\nItem Freq confidence: ";
  std::vector<double> itemRMSEs = itemFreqBucketRMSEsWInVal(origModel, fullModel,
      params.nUsers, params.nItems, itemFreq, 
      10, invalUsers, invalItems);
  dispVector(itemRMSEs);
  prefix = std::string(params.prefix) + "_iFreq_bucket.txt";
  writeVector(itemRMSEs, prefix.c_str());
   
  std::cout << "\nPPR confidence: ";
  std::vector<double> pprRMSEs = pprBucketRMSEsFrmPRWInVal(origModel, fullModel,
      params.nUsers, params.nItems, data.graphMat, 10, 
      ".ppr", invalUsers, invalItems);
  dispVector(pprRMSEs);
  prefix = std::string(params.prefix) + "_ppr_" + std::to_string(params.alpha) + "_bucket.txt";
  writeVector(pprRMSEs, prefix.c_str());
}


void computeSampPPRGPRBucksFrmFullModel(ModelMF& fullModel, ModelMF& origModel, 
    Data& data, Params& params, float lambda, int nSampUsers,
    std::unordered_set<int>& invalUsers, std::unordered_set<int>& invalItems) {
  
  std::cout << "\nGPR confidence... " << lambda ;
  std::vector<double> gprRMSEs = gprSampBucketRMSEsWInVal(origModel, fullModel, 
      params.nUsers, params.nItems,
      lambda, MAX_PR_ITER, data.graphMat, 10, invalUsers, invalItems, 
      nSampUsers, params.seed);
  std::string prefix = std::string(params.prefix) + "_gpr_" + std::to_string(lambda) + "_bucket.txt";
  writeVector(gprRMSEs, prefix.c_str());
    
  std::cout << "\nPPR confidence... " << lambda;
  std::vector<double> pprRMSEs = pprSampBucketRMSEsWInVal(origModel, fullModel, 
      params.nUsers, params.nItems, lambda, MAX_PR_ITER, data.graphMat, 10, 
      invalUsers, invalItems, nSampUsers, params.seed);
  prefix = std::string(params.prefix) + "_ppr_" + std::to_string(lambda) + "_bucket.txt";
  writeVector(pprRMSEs, prefix.c_str());
}


void computeSampBucksFrmFullModel(Data& data, Params& params) {
  std::cout << "\nCreating full model...";
  ModelMF fullModel(params, params.seed);
  fullModel.loadFacs(params.prefix);

  std::cout << "\nCreating original model...";
  ModelMF origModel(params, params.origUFacFile, params.origIFacFile, 
      params.seed);

  //get number of ratings per user and item, i.e. frequency
  auto rowColFreq = getRowColFreq(data.trainMat);
  auto userFreq = rowColFreq.first;
  auto itemFreq = rowColFreq.second;

  std::string prefix = std::string(params.prefix) + "_invalUsers.txt";
  std::vector<int> invalUsersVec = readVector(prefix.c_str());
  prefix = std::string(params.prefix) + "_invalItems.txt";
  std::vector<int> invalItemsVec = readVector(prefix.c_str());
  
  std::unordered_set<int> invalUsers;
  for (auto v: invalUsersVec) {
    invalUsers.insert(v);
  }
  
  std::unordered_set<int> invalItems;
  for (auto v: invalItemsVec) {
    invalItems.insert(v);
  }

  std::cout << "\nnInvalidUsers: " << invalUsers.size();
  std::cout << "\nnInvalidItems: " << invalItems.size() <<std::endl;
 
  std::cout << "\nTrain RMSE: " << fullModel.RMSE(data.trainMat, invalUsers, invalItems);
  std::cout << "\nTest RMSE: " << fullModel.RMSE(data.testMat, invalUsers, invalItems);
  std::cout << "\nVal RMSE: " << fullModel.RMSE(data.valMat, invalUsers, invalItems);
  
  //order item in decreasing order of frequency 
  std::vector<std::pair<int, double>> itemFreqPairs;
  for (int i = 0; i < itemFreq.size(); i++) {
    itemFreqPairs.push_back(std::make_pair(i, itemFreq[i]));
  }
  //TODO: put it in util
  //comparison to sort in decreasing order
  auto comparePair = [](std::pair<int, double> a, std::pair<int, double> b) { 
    return a.second > b.second; 
  };
  std::sort(itemFreqPairs.begin(), itemFreqPairs.end(), comparePair);

  int nSampUsers = 5000;
  int nUsers = data.trainMat->nrows;
  int nItems = data.trainMat->ncols;

  //compute item frequency buckets
  std::cout << "\nItem Freq confidence: ";
  std::vector<double> itemRMSEs = itemFreqSampBucketRMSEsWInVal(origModel,
      fullModel, nUsers, nItems,
      itemFreq, 10, invalUsers, invalItems, nSampUsers, params.seed);
  dispVector(itemRMSEs);
  prefix = std::string(params.prefix) + "_iFreq_bucket.txt";
  writeVector(itemRMSEs, prefix.c_str());
  
  std::vector<float> lambdas = {0.2, 0.4, 0.6, 0.8};
  int nThreads = lambdas.size() - 1;
  std::vector<std::thread> threads(nThreads);
  for (int thInd = 0; thInd < nThreads; thInd++) {
    threads[thInd] = std::thread(computeSampPPRGPRBucksFrmFullModel,
        std::ref(fullModel), std::ref(origModel), std::ref(data), 
        std::ref(params), lambdas[thInd], nSampUsers, 
        std::ref(invalUsers), std::ref(invalItems));
  }
  
  //last parameter in main thread
  computeSampPPRGPRBucksFrmFullModel(fullModel, origModel, data, params,
      lambdas[nThreads], nSampUsers, invalUsers, invalItems);
  
  //computeSampPPRGPRBucksFrmFullModel(fullModel, origModel, data, params,
  //    params.alpha, nSampUsers, invalUsers, invalItems);
  
  
  //wait for the threads to finish
  std::cout << "\nWaiting for threads to finish..." << std::endl;
  std::for_each(threads.begin(), threads.end(), std::mem_fn(&std::thread::join));
    
}


void computeSampTopNFrmFullModel(Data& data, Params& params) {
  
  std::cout << "\nCreating full model...";
  ModelMF fullModel(params, params.seed);
  fullModel.loadFacs(params.prefix);

  std::cout << "\nCreating original model...";
  ModelMF origModel(params, params.origUFacFile, params.origIFacFile, 
      params.seed);

  //get number of ratings per user and item, i.e. frequency
  auto rowColFreq = getRowColFreq(data.trainMat);
  auto userFreq = rowColFreq.first;
  auto itemFreq = rowColFreq.second;

  std::string prefix = std::string(params.prefix) + "_invalUsers.txt";
  std::vector<int> invalUsersVec = readVector(prefix.c_str());
  prefix = std::string(params.prefix) + "_invalItems.txt";
  std::vector<int> invalItemsVec = readVector(prefix.c_str());
  
  std::unordered_set<int> invalUsers;
  for (auto v: invalUsersVec) {
    invalUsers.insert(v);
  }
  
  std::unordered_set<int> invalItems;
  for (auto v: invalItemsVec) {
    invalItems.insert(v);
  }

  std::cout << "\nTrain RMSE: " << fullModel.RMSE(data.trainMat, invalUsers, invalItems);
  std::cout << "\nTest RMSE: " << fullModel.RMSE(data.testMat, invalUsers, invalItems);
  std::cout << "\nVal RMSE: " << fullModel.RMSE(data.valMat, invalUsers, invalItems);
  
  //order item in decreasing order of frequency 
  std::vector<std::pair<int, double>> itemFreqPairs;
  for (int i = 0; i < itemFreq.size(); i++) {
    itemFreqPairs.push_back(std::make_pair(i, itemFreq[i]));
  }
  //comparison to sort in decreasing order
  std::sort(itemFreqPairs.begin(), itemFreqPairs.end(), descComp);
  /*
  std::vector<int> users = {179907, 99641, 101732, 180172, 51668, 165001, 
                            68374, 21257, 122129};
  */

  int N = 1000;
  int nSampUsers = 5000;
  int nUsers = data.trainMat->nrows;
  int nItems = data.trainMat->ncols;

  //add top-N frequent items to filtered items
  std::unordered_set<int> filtItems;
  for (int i = 0; i < N; i++) {
    filtItems.insert(itemFreqPairs[i].first);
  }
 
  //add filtItems to invalItems
  //for (auto&& item: filtItems) {
  //  invalItems.insert(item);
  //}
   

  std::cout << "\nnInvalidUsers: " << invalUsers.size();
  std::cout << "\nnInvalidItems: " << invalItems.size() <<std::endl;
  
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

  
  
  std::vector<float> lambdas = {0.01, 0.25, 0.5, 0.75, 0.99};
  int nThreads = lambdas.size() - 1;
  std::vector<std::thread> threads(nThreads);
  for (int thInd = 0; thInd < nThreads; thInd++) {
    //prefix = std::string(params.prefix) + "_50_100_" 
    //          + std::to_string(lambdas[thInd]);
    prefix = std::string(params.prefix) + "_" + std::to_string(lambdas[thInd]) 
      + "_" + std::to_string(N);
    
    /*
    threads[thInd] = std::thread(pprSampUsersRMSEProb,
        data.graphMat, data.trainMat, nUsers, nItems, std::ref(origModel), 
        std::ref(fullModel), lambdas[thInd], MAX_PR_ITER, std::ref(invalUsers), 
        std::ref(invalItems), std::ref(filtItems), 100, params.seed, prefix);
    */
    
    threads[thInd] = std::thread(writeTopBuckRMSEs,
        std::ref(origModel), std::ref(fullModel), data.graphMat, data.trainMat, 
        lambdas[thInd], MAX_PR_ITER, std::ref(invalUsers), 
        std::ref(invalItems), std::ref(filtItems), nSampUsers, params.seed, N, 
        prefix);
  }
  
  //last parameter in main thread
  //prefix = std::string(params.prefix) + "_50_100_" 
  //          + std::to_string(lambdas[nThreads]);
  prefix = std::string(params.prefix) + "_" + std::to_string(lambdas[nThreads])
    + "_" + std::to_string(N);
  //pprSampUsersRMSEProb(data.graphMat, data.trainMat, nUsers, nItems, origModel, fullModel,
  //    lambdas[nThreads], MAX_PR_ITER, invalUsers, invalItems, filtItems, 100, 
  //    params.seed, prefix);
  writeTopBuckRMSEs(origModel, fullModel, data.graphMat, data.trainMat, 
      lambdas[nThreads], MAX_PR_ITER, invalUsers, invalItems, filtItems, nSampUsers, 
      params.seed, N, prefix);
  
  
  /*
  //last parameter in main thread
  prefix = std::string(params.prefix) + "_" + std::to_string(params.alpha)
    + "_" + std::to_string(N);
  writeTopBuckRMSEs(fullModel, origModel, data.graphMat, params.alpha,
      MAX_PR_ITER, invalUsers, invalItems, filtItems, nSampUsers, params.seed,
      N, prefix);
  */
   
  //wait for the threads to finish
  std::cout << "\nWaiting for threads to finish..." << std::endl;
  std::for_each(threads.begin(), threads.end(), std::mem_fn(&std::thread::join));
   
  /* 
  prefix = std::string(params.prefix) + "_sampFreq_" + std::to_string(params.alpha);
  pprSampUsersRMSEProb(data.graphMat, data.trainMat, nUsers, nItems, origModel, fullModel,
      params.alpha, MAX_PR_ITER, invalUsers, invalItems, filtItems, 5000, 
      params.seed, prefix);
  */
}


void computeBucksEstFullModel(Data& data, Params& params) {

  ModelMF fullModel(params, "fullEst1_invalItems.txt_full_uFac_50000_5_0.010000_0.001000.mat", 
      "fullEst1_invalItems.txt_full_iFac_33027_5_0.010000_0.001000.mat", params.seed);
  ModelMF fullBestModel(fullModel);
  
  Model origModel(params, params.seed);
  origModel.load(params.origUFacFile, params.origIFacFile);

  std::unordered_set<int> invalUsers;
  std::unordered_set<int> invalItems;
  
  std::cout << "\nStarting full model train...";
  fullModel.train(data, fullBestModel, invalUsers, invalItems);
 
  //save invalid users
  std::cout << "\nTotal invalid users: " << invalUsers.size();
  //write out invalid users
  std::string prefix = std::string(params.prefix) + "_invalUsers.txt";
  writeContainer(begin(invalUsers), end(invalUsers), prefix.c_str());
  
  //save invalid items
  std::cout << "\nTotal invalid items: " << invalItems.size();
  //write out invalid users
  prefix = std::string(params.prefix) + "_invalItems.txt";
  writeContainer(begin(invalItems), end(invalItems), prefix.c_str());

  //save best model
  prefix = std::string(params.prefix) + "_full";
  fullBestModel.save(prefix);
  
  //get number of ratings per user and item, i.e. frequency
  auto rowColFreq = getRowColFreq(data.trainMat);
  auto userFreq = rowColFreq.first;
  auto itemFreq = rowColFreq.second;
  
  std::cout << "\nOptimal confidence: ";
  std::vector<double> optRMSEs = confOptBucketRMSEsWInVal(origModel, fullModel,
      params.nUsers, params.nItems, 10, invalUsers, invalItems);
  dispVector(optRMSEs);
  prefix = std::string(params.prefix) + "_opt_bucket.txt";
  writeVector(optRMSEs, prefix.c_str());

  std::cout << "\nGPR confidence: ";
  std::vector<double> gprRMSEs = gprBucketRMSEsWInVal(origModel, fullModel,
      params.nUsers, params.nItems, params.alpha, MAX_PR_ITER, data.graphMat, 
      10, invalUsers, invalItems);
  dispVector(gprRMSEs);
  prefix = std::string(params.prefix) + "_gpr_" + std::to_string(params.alpha) + "_bucket.txt";
  writeVector(gprRMSEs, prefix.c_str());

  std::cout << "\nItem Freq confidence: ";
  std::vector<double> itemRMSEs = itemFreqBucketRMSEsWInVal(origModel, fullModel,
      params.nUsers, params.nItems, itemFreq, 
      10, invalUsers, invalItems);
  dispVector(itemRMSEs);
  prefix = std::string(params.prefix) + "_iFreq_bucket.txt";
  writeVector(itemRMSEs, prefix.c_str());
   
  std::cout << "\nPPR confidence: ";
  std::vector<double> pprRMSEs = pprBucketRMSEsFrmPRWInVal(origModel, fullModel,
      params.nUsers, params.nItems, data.graphMat, 10, 
      "flix_u1_i1_50Kx33027_0.8.ppr", invalUsers, invalItems);
  dispVector(pprRMSEs);
  prefix = std::string(params.prefix) + "_ppr_" + std::to_string(params.alpha) + "_bucket.txt";
  writeVector(pprRMSEs, prefix.c_str());
  
}


void computeConfScoresFrmModel(Data& data, Params& params) {
  std::vector<Model> bestModels;
  bestModels.push_back(Model(params, "",
        "", params.seed));
  
  std::cout << "\nnBestModels: " << bestModels.size();

  Model fullModel(params, params.seed);
  fullModel.load(params.initUFacFile, params.initIFacFile);
  Model origModel(params, params.seed);
  origModel.load(params.origUFacFile, params.origIFacFile);

  std::vector<int> invalUsersVec = readVector("multiconf_invalUsers.txt");
  std::vector<int> invalItemsVec = readVector("multiconf_invalItems.txt");


  std::unordered_set<int> invalUsers;
  for (auto v: invalUsersVec) {
    invalUsers.insert(v);
  }

  std::unordered_set<int> invalItems;
  for (auto v: invalItemsVec) {
    invalItems.insert(v);
  }

  std::cout << "\nnInvalidUsers: " << invalUsers.size();
  std::cout << "\nnInvalidItems: " << invalItems.size() <<std::endl;
  
  //get number of ratings per user and item, i.e. frequency
  auto rowColFreq = getRowColFreq(data.trainMat);
  auto userFreq = rowColFreq.first;
  auto itemFreq = rowColFreq.second;

  /*
  comparePPR2GPR(params.nUsers, params.nItems, data.graphMat,
      params.alpha, MAX_PR_ITER,
      "flix_y_10000x15905_0.8.ppr", "ppr2GPR_0_2.txt");  
  */
  
  std::cout << "\nModel confidence: ";
  std::vector<double> confRMSEs = confBucketRMSEsWInval(origModel, fullModel, 
      bestModels, params.nUsers, params.nItems, 10, invalUsers, invalItems);
  dispVector(confRMSEs);
  std::string prefix = std::string(params.prefix) + "_conf_bucket.txt";
  writeVector(confRMSEs, prefix.c_str());

  std::cout << "\nOptimal confidence: ";
  std::vector<double> optRMSEs = confOptBucketRMSEsWInVal(origModel, fullModel,
      params.nUsers, params.nItems, 10, invalUsers, invalItems);
  dispVector(optRMSEs);
  prefix = std::string(params.prefix) + "_opt_bucket.txt";
  writeVector(optRMSEs, prefix.c_str());

  std::cout << "\nGPR confidence: ";
  std::vector<double> gprRMSEs = gprBucketRMSEsWInVal(origModel, fullModel,
      params.nUsers, params.nItems, params.alpha, MAX_PR_ITER, data.graphMat, 
      10, invalUsers, invalItems);
  dispVector(gprRMSEs);
  prefix = std::string(params.prefix) + "_gpr_" + std::to_string(params.alpha) + "_bucket.txt";
  writeVector(gprRMSEs, prefix.c_str());

  std::cout << "\nItem Freq confidence: ";
  std::vector<double> itemRMSEs = itemFreqBucketRMSEsWInVal(origModel, fullModel,
      params.nUsers, params.nItems, itemFreq, 
      10, invalUsers, invalItems);
  dispVector(itemRMSEs);
  prefix = std::string(params.prefix) + "_iFreq_bucket.txt";
  writeVector(itemRMSEs, prefix.c_str());
 
 
  std::cout << "\nPPR confidence: ";
  std::vector<double> pprRMSEs = pprBucketRMSEsFrmPRWInVal(origModel, fullModel,
      params.nUsers, params.nItems, data.graphMat, 10, 
      ".ppr", invalUsers, invalItems);
  dispVector(pprRMSEs);
  prefix = std::string(params.prefix) + "_ppr_" + std::to_string(params.alpha) + "_bucket.txt";
  writeVector(pprRMSEs, prefix.c_str());
  
}


void computePRScores(Data& data, Params& params) {
  Model fullModel(params, params.seed);
  fullModel.load(params.initUFacFile, params.initIFacFile);
  Model origModel(params, params.seed);
  origModel.load(params.origUFacFile, params.origIFacFile);
  //NOTE: using params.alpha as (1 - restartProb)
  std::vector<double> pprRMSEs = pprBucketRMSEs(origModel, fullModel, 
    params.nUsers, params.nItems, params.alpha, MAX_PR_ITER, 
    data.graphMat, 10);
  std::string prefix = std::string(params.prefix) + "_pprconf_bucket.txt";
  writeVector(pprRMSEs, prefix.c_str());
  dispVector(pprRMSEs);
}


void computeGPRScores(Data& data, Params& params) {
  Model fullModel(params, params.seed);
  fullModel.load(params.initUFacFile, params.initIFacFile);
  Model origModel(params, params.seed);
  origModel.load(params.origUFacFile, params.origIFacFile);
  //NOTE: using params.alpha as (1 - restartProb)
  std::vector<double> pprRMSEs = gprBucketRMSEs(origModel, fullModel, 
    params.nUsers, params.nItems, params.alpha, MAX_PR_ITER, 
    data.graphMat, 10);
  std::string prefix = std::string(params.prefix) + "_gprconf_bucket.txt";
  writeVector(pprRMSEs, prefix.c_str());
  dispVector(pprRMSEs);
}


void computePRScores2(Data& data, Params& params) {
  Model fullModel(params, params.seed);
  fullModel.load(params.initUFacFile, params.initIFacFile);
  Model origModel(params, params.seed);
  origModel.load(params.origUFacFile, params.origIFacFile);
  //NOTE: using params.alpha as (1 - restartProb)
  std::vector<double> pprRMSEs = pprBucketRMSEsFrmPR(origModel, fullModel, 
    params.nUsers, params.nItems, 
    data.graphMat, 10, "prItems_0.8.txt");
  std::string prefix = std::string(params.prefix) + "_ppr2conf_bucket.txt";
  writeVector(pprRMSEs, prefix.c_str());
  dispVector(pprRMSEs);
}


void computeOptScores(Data& data, Params& params) {
  
  Model fullModel(params, params.seed);
  fullModel.load(params.initUFacFile, params.initIFacFile);
  Model origModel(params, params.seed);
  origModel.load(params.origUFacFile, params.origIFacFile);
  
  std::vector<double> confRMSEs = confOptBucketRMSEs(origModel, fullModel, 
    params.nUsers, params.nItems, 10);

  std::string prefix = std::string(params.prefix) + "_optconf_bucket.txt";
  writeVector(confRMSEs, prefix.c_str());
  dispVector(confRMSEs);
}


void computeConfCurve(Data& data, Params& params) {
  
  int nUsers = data.trainMat->nrows;
  int nItems = data.trainMat->ncols;

  ModelMF fullModel(params, params.seed);
  svdFrmSvdlibCSR(data.trainMat, fullModel.facDim, fullModel.uFac, 
      fullModel.iFac); 
  
  int nThreads = 5;
  
  std::vector<std::thread> threads(nThreads);
  
  std::vector<std::shared_ptr<ModelMF>> trainModels;
  std::vector<std::shared_ptr<Model>> pbestModels;
  std::vector<std::unordered_set<int>> mInvalUsers (nThreads);
  std::vector<std::unordered_set<int>> mInvalItems (nThreads);

  for (int thInd = 0; thInd < nThreads; thInd++) {
    int seed = params.seed + thInd + 1;
    std::shared_ptr<ModelMF> p(new ModelMF(params, seed));
    p->uFac = fullModel.uFac;
    p->iFac = fullModel.iFac;
    trainModels.push_back(p);
    std::shared_ptr<Model> p2(new Model(params, seed));
    pbestModels.push_back(p2);
    //invoke training on thread
    std::cout << "\nInvoking thread: " << thInd << std::endl;  
    threads[thInd] = std::thread(&ModelMF::partialTrain, 
        trainModels[thInd], std::ref(data), 
        std::ref(*pbestModels[thInd]), std::ref(mInvalUsers[thInd]),
        std::ref(mInvalItems[thInd]));
  }

  //train full model in main thread
  ModelMF fullBestModel(fullModel);
  std::unordered_set<int> invalidUsers;
  std::unordered_set<int> invalidItems;
  std::cout << "\nStarting full model train...";
  fullModel.train(data, fullBestModel, invalidUsers, invalidItems);

  std::cout << "\nWaiting for threads to finish...";
  //wait for threads to finish
  std::for_each(threads.begin(), threads.end(), 
      std::mem_fn(&std::thread::join));

  std::cout << "\nFinished all threads.";

  std::cout << "\nModels save...";
  std::vector<Model> bestModels;
  //save the best models
  for (int thInd = 0; thInd < nThreads; thInd++) {
    int seed = params.seed + thInd + 1;
    std::string prefix(params.prefix);
    prefix = prefix + "_partial_" + std::to_string(seed); 
    pbestModels[thInd]->save(prefix);
    bestModels.push_back(*pbestModels[thInd]);
  }
  //save the best full model
  std::string prefix(params.prefix);
  prefix = prefix + "_full";
  fullBestModel.save(prefix);

  ModelMF origModel(params, params.seed);
  origModel.load(params.origUFacFile, params.origIFacFile);
  
  //find all invalid users
  for (int thInd = 0; thInd < nThreads; thInd++) {
    for (const auto& elem: mInvalUsers[thInd]) {
      invalidUsers.insert(elem);
    }
  }
  
  std::cout << "\nTotal invalid users: " << invalidUsers.size();
  //write out invalid users
  prefix = std::string(params.prefix) + "_invalUsers.txt";
  writeContainer(begin(invalidUsers), end(invalidUsers), prefix.c_str());
  
  //find all invalid items
  for (int thInd = 0; thInd < nThreads; thInd++) {
    for (const auto& elem: mInvalItems[thInd]) {
      invalidItems.insert(elem);
    }
  }

  std::cout << "\nTotal invalid items: " << invalidItems.size();
  //write out invalid items
  prefix = std::string(params.prefix) + "_invalItems.txt";
  writeContainer(begin(invalidItems), end(invalidItems), prefix.c_str());

  double halfRatCount = ((double)nUsers*(double)nItems)/2.0;
  int testSize = std::min((double)MAX_MISS_RATS, halfRatCount);
  std::vector<std::pair<int, int>> testPairs = getTestPairs(data.trainMat, invalidUsers,
      invalidItems, testSize, params.seed);
  
  //compute confidence using the best models for 10 buckets
  std::vector<double> confCurve = computeMissingModConfSamp(bestModels, 
      origModel, fullModel, 10, 0.05, testPairs);
  std::cout << "\nConfidence bucket Curve: ";
  dispVector(confCurve);
  prefix = std::string(params.prefix) + "_mconf_curve.txt";
  writeVector(confCurve, prefix.c_str());

  std::vector<double> optConfCurve = genOptConfidenceCurve(testPairs, origModel,
      fullModel, 10, 0.05);
  std::cout << "\nOpt model conf curve: ";
  dispVector(optConfCurve);
  prefix = std::string(params.prefix) + "_optconf_curve_miss.txt";
  writeVector(optConfCurve, prefix.c_str());

  //get number of ratings per user and item, i.e. frequency
  auto rowColFreq = getRowColFreq(data.trainMat);
  auto userFreq = rowColFreq.first;
  auto itemFreq = rowColFreq.second;

  std::vector<double> userConfCurve = genUserConfCurve(testPairs, origModel,
      fullModel, 10, 0.05, userFreq);
  std::cout << "\nuser conf curve: ";
  dispVector(userConfCurve);
  prefix = std::string(params.prefix) + "_userconf_curve_miss.txt";
  writeVector(userConfCurve, prefix.c_str());

  std::vector<double> itemConfCurve = genItemConfCurve(testPairs, origModel,
      fullModel, 10, 0.05, itemFreq);
  std::cout << "\nitem conf curve: ";
  dispVector(itemConfCurve);
  prefix = std::string(params.prefix) + "_itemconf_curve_miss.txt";
  writeVector(itemConfCurve, prefix.c_str());

  //compute global page rank confidence
  //NOTE: using params.alpha as (1 - restartProb)
  std::vector<double> gprCurve = computeMissingGPRConfSamp(data.graphMat,
      params.alpha, MAX_PR_ITER, origModel, 
      fullModel, 10, 0.05, testPairs, nUsers);
  prefix = std::string(params.prefix) + "_gprconf_curve.txt";
  writeVector(gprCurve, prefix.c_str());
  std::cout << "\nGPR confidence Curve:";
  dispVector(gprCurve);

  //compute ppr confidence
  std::vector<double> pprCurve = computeMissingPPRConfExtSamp(data.trainMat, 
      data.graphMat, params.alpha, MAX_PR_ITER, origModel, 
      fullModel, 10, 0.05, ".ppr", testPairs);
  prefix = std::string(params.prefix) + "_pprconf_curve.txt";
  writeVector(pprCurve, prefix.c_str());
  std::cout << "\nPPR confidence Curve:";
  dispVector(pprCurve);
  
}


void computeConfCurvesFrmModel(Data& data, Params& params) {
  
  int nUsers = data.trainMat->nrows;
  int nItems = data.trainMat->ncols;

  std::vector<Model> bestModels;
  bestModels.push_back(Model(params, "",
        "", params.seed));

  std::cout << "\nnBestModels: " << bestModels.size();

  Model fullModel(params, params.seed);
  fullModel.load(params.initUFacFile, params.initIFacFile);
  Model origModel(params, params.seed);
  origModel.load(params.origUFacFile, params.origIFacFile);

  std::vector<int> invalUsersVec = readVector("multiconf_invalUsers.txt");
  std::vector<int> invalItemsVec = readVector("multiconf_invalItems.txt");

  std::cout << "\nnInvalidUsers: " << invalUsersVec.size();
  std::cout << "\nnInvalidItems: " << invalItemsVec.size() << std::endl;

  std::unordered_set<int> invalUsers;
  for (auto v: invalUsersVec) {
    invalUsers.insert(v);
  }

  std::unordered_set<int> invalItems;
  for (auto v: invalItemsVec) {
    invalItems.insert(v);
  }


  double halfRatCount = ((double)nUsers*(double)nItems)/2.0;
  int testSize = std::min((double)MAX_MISS_RATS, halfRatCount);
  std::vector<std::pair<int, int>> testPairs = getTestPairs(data.trainMat, invalUsers,
      invalItems, testSize, params.seed);

  
  std::vector<double> confCurve = computeMissingModConfSamp(bestModels, 
      origModel, fullModel, 10, 0.05, testPairs);
  std::cout << "\nModels confidence Curve: ";
  dispVector(confCurve);
  std::string prefix = std::string(params.prefix) + "_mconf_curve_miss.txt";
  writeVector(confCurve, prefix.c_str());

  std::vector<double> optConfCurve = genOptConfidenceCurve(testPairs, origModel,
      fullModel, 10, 0.05);
  std::cout << "\nOpt model conf curve: ";
  dispVector(optConfCurve);
  prefix = std::string(params.prefix) + "_optconf_curve_miss.txt";
  writeVector(optConfCurve, prefix.c_str());

  //get number of ratings per user and item, i.e. frequency
  auto rowColFreq = getRowColFreq(data.trainMat);
  auto userFreq = rowColFreq.first;
  auto itemFreq = rowColFreq.second;

  std::vector<double> userConfCurve = genUserConfCurve(testPairs, origModel,
      fullModel, 10, 0.05, userFreq);
  std::cout << "\nuser conf curve: ";
  dispVector(userConfCurve);
  prefix = std::string(params.prefix) + "_userconf_curve_miss.txt";
  writeVector(userConfCurve, prefix.c_str());

  std::vector<double> itemConfCurve = genItemConfCurve(testPairs, origModel,
      fullModel, 10, 0.05, itemFreq);
  std::cout << "\nitem conf curve: ";
  dispVector(itemConfCurve);
  prefix = std::string(params.prefix) + "_itemconf_curve_miss.txt";
  writeVector(itemConfCurve, prefix.c_str());

  //compute global page rank confidence
  //NOTE: using params.alpha as (1 - restartProb)
  std::vector<double> gprCurve = computeMissingGPRConfSamp(data.graphMat,
      params.alpha, MAX_PR_ITER, origModel, 
      fullModel, 10, 0.05, testPairs, nUsers);
  prefix = std::string(params.prefix) + "_gprconf_curve_miss.txt";
  writeVector(gprCurve, prefix.c_str());
  std::cout << "\nGPR confidence Curve:";
  dispVector(gprCurve);
  /*
  std::vector<double> pprCurve = computeMissingPPRConfExtSamp(data.trainMat, 
      data.graphMat, params.alpha, MAX_PR_ITER, origModel, 
      fullModel, 10, 0.05, ".ppr", testPairs);
  prefix = std::string(params.prefix) + "_pprconf_curve_miss.txt";
  writeVector(pprCurve, prefix.c_str());
  std::cout << "\nPPR confidence Curve:";
  dispVector(pprCurve);
  */
}


void computeConfRMSECurvesFrmModel(Data& data, Params& params) {
  
  int nUsers = data.trainMat->nrows;
  int nItems = data.trainMat->ncols;

  std::vector<Model> bestModels;
  bestModels.push_back(Model(params, "",
        "", params.seed));

  std::cout << "\nnBestModels: " << bestModels.size();

  Model fullModel(params, params.seed);
  fullModel.load(params.initUFacFile, params.initIFacFile);
  Model origModel(params, params.seed);
  origModel.load(params.origUFacFile, params.origIFacFile);

  std::vector<int> invalUsersVec = readVector("multiconf_invalUsers.txt");
  std::vector<int> invalItemsVec = readVector("multiconf_invalItems.txt");

  std::cout << "\nnInvalidUsers: " << invalUsersVec.size();
  std::cout << "\nnInvalidItems: " << invalItemsVec.size() << std::endl;

  std::unordered_set<int> invalUsers;
  for (auto v: invalUsersVec) {
    invalUsers.insert(v);
  }

  std::unordered_set<int> invalItems;
  for (auto v: invalItemsVec) {
    invalItems.insert(v);
  }


  double halfRatCount = ((double)nUsers*(double)nItems)/2.0;
  int testSize = std::min((double)MAX_MISS_RATS, halfRatCount);
  std::vector<std::pair<int, int>> testPairs = getTestPairs(data.trainMat, invalUsers,
      invalItems, testSize, params.seed);

  std::string prefix = std::string(params.prefix) + "_opt_rmse_curve_miss.txt";
  
  std::vector<double> optConfCurve = genOptConfRMSECurve(testPairs, origModel,
      fullModel, 10);
  std::cout << "\nOpt RMSE curve: ";
  dispVector(optConfCurve);
  writeVector(optConfCurve, prefix.c_str());
  
  //get number of ratings per user and item, i.e. frequency
  auto rowColFreq = getRowColFreq(data.trainMat);
  auto userFreq = rowColFreq.first;
  auto itemFreq = rowColFreq.second;

  std::vector<double> userConfCurve = genUserConfRMSECurve(testPairs, origModel,
      fullModel, 10, userFreq);
  std::cout << "\nuser RMSE curve: ";
  dispVector(userConfCurve);
  prefix = std::string(params.prefix) + "_user_rmse_curve_miss.txt";
  writeVector(userConfCurve, prefix.c_str());

  std::vector<double> itemConfCurve = genItemConfRMSECurve(testPairs, origModel,
      fullModel, 10, itemFreq);
  std::cout << "\nItem RMSE curve:";
  dispVector(itemConfCurve);
  prefix = std::string(params.prefix) + "_item_rmse_curve_miss.txt";
  writeVector(itemConfCurve, prefix.c_str());

  std::vector<double> modelConfCurve = genModelConfRMSECurve(testPairs, origModel,
      fullModel, bestModels, 10);
  std::cout << "\nModel conf RMSE curve:";
  dispVector(modelConfCurve);
  prefix = std::string(params.prefix) + "_model_rmse_curve_miss.txt";
  writeVector(modelConfCurve, prefix.c_str());

  std::vector<double> gprConfCurve = genGPRConfRMSECurve(testPairs, origModel,
      fullModel, data.graphMat, params.alpha, MAX_PR_ITER, 10);
  std::cout << "\nGpr conf RMSE curve:";
  dispVector(gprConfCurve);
  prefix = std::string(params.prefix) + "_gpr_rmse_curve_miss.txt";
  writeVector(gprConfCurve, prefix.c_str());

  std::vector<double> pprConfCurve = genPPRConfRMSECurve(testPairs, origModel,
      fullModel, data.graphMat, params.alpha, MAX_PR_ITER, 
      ".ppr", 10);
  std::cout << "\nppr conf RMSE curve:";
  dispVector(pprConfCurve);
  prefix = std::string(params.prefix) + "_ppr_rmse_curve_miss.txt";
  writeVector(pprConfCurve, prefix.c_str());  
}


void computeConfCurveTest(Data& data, Params& params) {
  
  int nUsers = data.trainMat->nrows;
  int nItems = data.trainMat->ncols;

  ModelMF fullModel(params, params.seed);
  svdFrmSvdlibCSR(data.trainMat, fullModel.facDim, fullModel.uFac, 
      fullModel.iFac); 
  
  int nThreads = 5;
  
  std::vector<std::thread> threads(nThreads);
  
  std::vector<std::shared_ptr<ModelMF>> trainModels;
  std::vector<std::shared_ptr<Model>> pbestModels;
  std::vector<std::unordered_set<int>> mInvalUsers (nThreads);
  std::vector<std::unordered_set<int>> mInvalItems (nThreads);

  for (int thInd = 0; thInd < nThreads; thInd++) {
    int seed = params.seed + thInd + 1;
    std::shared_ptr<ModelMF> p(new ModelMF(params, seed));
    p->uFac = fullModel.uFac;
    p->iFac = fullModel.iFac;
    trainModels.push_back(p);
    std::shared_ptr<Model> p2(new Model(params, seed));
    pbestModels.push_back(p2);
    //invoke training on thread
    std::cout << "\nInvoking thread: " << thInd << std::endl;  
    threads[thInd] = std::thread(&ModelMF::partialTrain, 
        trainModels[thInd], std::ref(data), 
        std::ref(*pbestModels[thInd]), std::ref(mInvalUsers[thInd]),
        std::ref(mInvalItems[thInd]));
  }

  //train full model in main thread
  ModelMF fullBestModel(fullModel);
  std::unordered_set<int> invalidUsers;
  std::unordered_set<int> invalidItems;
  std::cout << "\nStarting full model train...";
  fullModel.train(data, fullBestModel, invalidUsers, invalidItems);

  std::cout << "\nWaiting for threads to finish...";
  //wait for threads to finish
  std::for_each(threads.begin(), threads.end(), 
      std::mem_fn(&std::thread::join));

  std::cout << "\nFinished all threads.";

  std::cout << "\nModels save...";
  std::vector<Model> bestModels;
  //save the best models
  for (int thInd = 0; thInd < nThreads; thInd++) {
    int seed = params.seed + thInd + 1;
    std::string prefix(params.prefix);
    prefix = prefix + "_partial_" + std::to_string(seed); 
    pbestModels[thInd]->save(prefix);
    bestModels.push_back(*pbestModels[thInd]);
  }
  //save the best full model
  std::string prefix(params.prefix);
  prefix = prefix + "_full";
  fullBestModel.save(prefix);

  ModelMF origModel(params, params.seed);
  origModel.load(params.origUFacFile, params.origIFacFile);
  
  //find all invalid users
  for (int thInd = 0; thInd < nThreads; thInd++) {
    for (const auto& elem: mInvalUsers[thInd]) {
      invalidUsers.insert(elem);
    }
  }
  
  std::cout << "\nTotal invalid users: " << invalidUsers.size();
  //write out invalid users
  prefix = std::string(params.prefix) + "_invalUsers.txt";
  writeContainer(begin(invalidUsers), end(invalidUsers), prefix.c_str());
  
  //find all invalid items
  for (int thInd = 0; thInd < nThreads; thInd++) {
    for (const auto& elem: mInvalItems[thInd]) {
      invalidItems.insert(elem);
    }
  }

  std::cout << "\nTotal invalid items: " << invalidItems.size();
  //write out invalid items
  prefix = std::string(params.prefix) + "_invalItems.txt";
  writeContainer(begin(invalidItems), end(invalidItems), prefix.c_str());

  auto testPairs = getUIPairs(data.testMat, invalidUsers, invalidItems);
  
  //compute confidence using the best models for 10 buckets
  std::vector<double> confCurve = computeMissingModConfSamp(bestModels, 
      origModel, fullModel, 10, 0.05, testPairs);
  std::cout << "\nConfidence bucket Curve: ";
  dispVector(confCurve);
  prefix = std::string(params.prefix) + "_mconf_curve.txt";
  writeVector(confCurve, prefix.c_str());

  std::vector<double> optConfCurve = genOptConfidenceCurve(testPairs, origModel,
      fullModel, 10, 0.05);
  std::cout << "\nOpt model conf curve: ";
  dispVector(optConfCurve);
  prefix = std::string(params.prefix) + "_optconf_curve_miss.txt";
  writeVector(optConfCurve, prefix.c_str());

  //get number of ratings per user and item, i.e. frequency
  auto rowColFreq = getRowColFreq(data.trainMat);
  auto userFreq = rowColFreq.first;
  auto itemFreq = rowColFreq.second;

  std::vector<double> userConfCurve = genUserConfCurve(testPairs, origModel,
      fullModel, 10, 0.05, userFreq);
  std::cout << "\nuser conf curve: ";
  dispVector(userConfCurve);
  prefix = std::string(params.prefix) + "_userconf_curve_miss.txt";
  writeVector(userConfCurve, prefix.c_str());

  std::vector<double> itemConfCurve = genItemConfCurve(testPairs, origModel,
      fullModel, 10, 0.05, itemFreq);
  std::cout << "\nitem conf curve: ";
  dispVector(itemConfCurve);
  prefix = std::string(params.prefix) + "_itemconf_curve_miss.txt";
  writeVector(itemConfCurve, prefix.c_str());

  //compute global page rank confidence
  //NOTE: using params.alpha as (1 - restartProb)
  std::vector<double> gprCurve = computeMissingGPRConfSamp(data.graphMat,
      params.alpha, MAX_PR_ITER, origModel, 
      fullModel, 10, 0.05, testPairs, nUsers);
  prefix = std::string(params.prefix) + "_gprconf_curve.txt";
  writeVector(gprCurve, prefix.c_str());
  std::cout << "\nGPR confidence Curve:";
  dispVector(gprCurve);

  //compute ppr confidence
  std::vector<double> pprCurve = computeMissingPPRConfExtSamp(data.trainMat, 
      data.graphMat, params.alpha, MAX_PR_ITER, origModel, 
      fullModel, 10, 0.05, ".ppr", testPairs);
  prefix = std::string(params.prefix) + "_pprconf_curve.txt";
  writeVector(pprCurve, prefix.c_str());
  std::cout << "\nPPR confidence Curve:";
  dispVector(pprCurve);
  
}


void testTailRec(Data& data, Params& params) {
  
  ModelMF mfModel(params, params.seed);
  //load previously learned factors
  mfModel.loadFacs(params.prefix);
  
  //replace model with svd
  //svdFrmSvdlibCSR(data.trainMat, mfModel.facDim, mfModel.uFac, mfModel.iFac);

  std::unordered_set<int> invalidUsers;
  std::unordered_set<int> invalidItems;
  
  /*
  std::string prefix = std::string(params.prefix) + "_invalUsers.txt";
  std::vector<int> invalUsersVec = readVector(prefix.c_str());
  prefix = std::string(params.prefix) + "_invalItems.txt";
  std::vector<int> invalItemsVec = readVector(prefix.c_str());
  
  for (auto v: invalUsersVec) {
    invalidUsers.insert(v);
  }
  
  for (auto v: invalItemsVec) {
    invalidItems.insert(v);
  }
  */

  std::unordered_set<int> headItems;

  ModelMF bestModel(mfModel);
  std::cout << "\nStarting model train...";
  //mfModel.train(data, bestModel, invalidUsers, invalidItems);
  std::cout << "\nTest RMSE: " << bestModel.RMSE(data.testMat, invalidUsers, 
      invalidItems);
  
  //write out invalid users
  std::string prefix = std::string(params.prefix) + "_invalUsers.txt";
  writeContainer(begin(invalidUsers), end(invalidUsers), prefix.c_str());

  //write out invalid users
  prefix = std::string(params.prefix) + "_invalItems.txt";
  writeContainer(begin(invalidItems), end(invalidItems), prefix.c_str());
  
  //get headdItems
  headItems = getHeadItems(data.trainMat);
  std::cout << "\nNo. of head items: " << headItems.size() << " head items pc: " 
    << ((float)headItems.size()/(data.trainMat->ncols)) << std::endl; 

  //write out head items
  //prefix = std::string(params.prefix) + "_headItems.txt";
  //writeContainer(begin(headItems), end(headItems), prefix.c_str());
  
  int N = 10;
  
  std::vector<float> lambdas = {0.01, 0.25};
  int nThreads = lambdas.size();
  std::vector<std::thread> threads(nThreads);
  std::cout << "\nStarting threads...." << std::endl;
  for (int thInd = 0; thInd < nThreads; thInd++) {
    prefix = std::string(params.prefix) + "_" + std::to_string(lambdas[thInd]) 
      + "_" + std::to_string(N);
    threads[thInd] = std::thread(topNRecTail, std::ref(bestModel), 
        data.trainMat, data.testMat, data.graphMat, lambdas[thInd],
        std::ref(invalidItems), std::ref(invalidUsers), std::ref(headItems),
        N, params.seed, prefix);    
  }

  /*
  prefix = std::string(params.prefix) + "_" + std::to_string(lambdas[nThreads])
    + "_" + std::to_string(N);
  topNRecTail(bestModel, data.trainMat, data.testMat, data.graphMat, 
      lambdas[nThreads], invalidItems, invalidUsers, headItems, N, 
      params.seed, prefix);   
  */

  //run puresvd in main thread
  svdFrmSvdlibCSR(data.trainMat, mfModel.facDim, mfModel.uFac, mfModel.iFac);
  prefix = std::string(params.prefix) + "_puresvd_" + std::to_string(lambdas[0])
    + "_" + std::to_string(N);
  topNRecTail(mfModel, data.trainMat, data.testMat, data.graphMat, 
      lambdas[0], invalidItems, invalidUsers, headItems, N, 
      params.seed, prefix);

  //prefix = std::string(params.prefix) + "_" + std::to_string(params.alpha)
  //  + "_" + std::to_string(N);
  //topNRecTail(bestModel, data.trainMat, data.testMat, data.graphMat, 
  //    params.alpha, invalidItems, invalidUsers, headItems, N, 
  //    params.seed, prefix);   
  
  //wait for threads to finish
  std::cout << "\nWaiting for threads to finish..." << std::endl;
  std::for_each(threads.begin(), threads.end(), 
      std::mem_fn(&std::thread::join));
  
}


int main(int argc , char* argv[]) {
  
  //get passed parameters
  Params params = parse_cmd_line(argc, argv);

  //initialize seed
  std::srand(params.seed);

  Data data (params);

  /*
  auto meanVar = getMeanVar(data.origUFac, data.origIFac, data.origFacDim, 
      data.nUsers, data.nItems);

  std::cout << "\nmean = " << meanVar.first << " variance = " 
    << meanVar.second << std::endl;
  */

  //writeCSRWSparsityStructure(data.trainMat, "y_u2_i34_100Kx50K_50.syn.csr",
  //    data.origUFac, data.origIFac, params.facDim);
  //writeBlkDiagJoinedCSR("", "", "");

  //computeConfScoresFrmModel(data, params);
  //computeConf(data, params);  
  //computeBucksEstFullModel(data, params);
  //computeBucksFrmFullModel(data, params);

  //writeSubSampledMat(data.trainMat, 
  //    "ratings_229060x26779_25_0.6.syn.csr2", 0.6, params.seed);
  //writeTrainTestMat(data.trainMat,  "", 
  //   "", ,  params.seed);
  //writeItemSimMat(data.trainMat, "ratings_26779x26779_25.syn.trainItems.metis");
  //writeItemSimMatNonSymm(data.trainMat, 
  //    "ratings_26779x26779_25.syn.trainItems.nonsym.metis");
  //writeItemJaccSimMat(data.trainMat, "y_u2_i34_100Kx50K.train.jacSim.metis");
  //writeItemJaccSimFrmCorat(data.trainMat, data.graphMat, 
  //    "ratings_26779x26779_25.syn.trainItems.jacSim2.metis");
  //writeCoRatings(data.trainMat, "y_u2_i34_100Kx50K.train.coRatings");
  //std::cout << "ifUISorted: " << checkIfUISorted(data.trainMat) << std::endl ;

  /*  
  writeTrainTestValMat(data.trainMat,  
      "y_u2_i34_100Kx50K.train.csr",
      "y_u2_i34_100Kx50K.test.csr",
      "y_u2_i34_100Kx50K.val.csr",
      0.1, 0.1, params.seed); 
  */ 

  //ModelMF mfModel(params, params.initUFacFile, 
  //    params.initIFacFile, params.seed);
  
  /*
  ModelMF mfModel(params, params.seed);
  //initialize model with svd
  svdFrmSvdlibCSR(data.trainMat, mfModel.facDim, mfModel.uFac, mfModel.iFac);
  
  std::unordered_set<int> invalidUsers;
  std::unordered_set<int> invalidItems;

  ModelMF bestModel(mfModel);
  std::cout << "\nStarting model train...";
  mfModel.train(data, bestModel, invalidUsers, invalidItems);
  std::cout << "\nTest RMSE: " << bestModel.RMSE(data.testMat, invalidUsers, 
      invalidItems);
  
  //write out invalid users
  std::string prefix = std::string(params.prefix) + "_invalUsers.txt";
  writeContainer(begin(invalidUsers), end(invalidUsers), prefix.c_str());

  //write out invalid users
  prefix = std::string(params.prefix) + "_invalItems.txt";
  writeContainer(begin(invalidItems), end(invalidItems), prefix.c_str());
  */
  //computeSampTopNFrmFullModel(data, params);  
  testTailRec(data, params);

  return 0;
}


