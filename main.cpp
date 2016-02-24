#include <iostream>
#include <cstdlib>
#include <future>
#include <chrono>
#include <thread>
#include "io.h"
#include "util.h"
#include "datastruct.h"
#include "modelMF.h"
#include "confCompute.h"


Params parse_cmd_line(int argc, char *argv[]) {
  
  if (argc < 20) {
    std::cout << "\nNot enough arguments";
    exit(0);
  }  
  
  Params params(atoi(argv[1]), atoi(argv[2]), atoi(argv[3]), atoi(argv[4]), 
      atoi(argv[5]), atoi(argv[6]),
      atof(argv[7]), atof(argv[8]), atof(argv[9]), atof(argv[10]), atof(argv[11]),
      argv[12], argv[13], argv[14], argv[15], argv[16], argv[17], argv[18], argv[19]);

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
  std::vector<double> pprRMSEs = pprBucketRMSEsFrmPRWInVal(origModel, fullBestModel,
      params.nUsers, params.nItems, data.graphMat, 10, 
      ".ppr", invalidUsers, invalidItems);
  /*
   std::vector<double> pprRMSEs = pprBucketRMSEsWInVal(origModel, fullBestModel, 
    params.nUsers, params.nItems, params.alpha, MAX_PR_ITER, 
    data.graphMat, 10, invalidUsers, invalidItems);
  */
  prefix = std::string(params.prefix) + "_pprconf_"+ std::to_string(params.alpha ) + "_bucket.txt";
  writeVector(pprRMSEs, prefix.c_str());
  std::cout << "\nPPR confidence RMSEs:";
  dispVector(pprRMSEs);
  
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
  for (auto v: invalItems) {
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
      "ppr", invalUsers, invalItems);
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
  for (auto v: invalItems) {
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
  for (auto v: invalItems) {
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


int main(int argc , char* argv[]) {

  
  //get passed parameters
  Params params = parse_cmd_line(argc, argv);

  //initialize seed
  std::srand(params.seed);

  Data data (params);

  //writeCSRWSparsityStructure(data.trainMat, "",
  //    data.origUFac, data.origIFac, params.facDim);
  
  //writeBlkDiagJoinedCSR("", "", "");

  //computeConfScoresFrmModel(data, params);
  computeConf(data, params);  
  return 0;
}


