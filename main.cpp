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
  fullModel.train(data, fullBestModel);

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

  std::unordered_set<int> invalidUsers;
  std::unordered_set<int> invalidItems;
  
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

  //compute confidence using the best models for 10 buckets
  std::vector<double> confRMSEs = confBucketRMSEsWInval(origModel, fullBestModel, 
      bestModels, params.nUsers, params.nItems, 10, invalidUsers, invalidItems);
  std::cout << "\nConfidence bucket RMSEs: ";
  dispVector(confRMSEs);
  prefix = std::string(params.prefix) + "_mconf_bucket.txt";
  writeVector(confRMSEs, prefix.c_str());

  //compute global page rank confidence
  //NOTE: using params.alpha as (1 - restartProb)
  std::vector<double> gprRMSEs = gprBucketRMSEsWInVal(origModel, fullBestModel, 
    params.nUsers, params.nItems, params.alpha, MAX_PR_ITER, 
    data.graphMat, 10, invalidUsers, invalidItems);
  prefix = std::string(params.prefix) + "_gprconf_bucket.txt";
  writeVector(gprRMSEs, prefix.c_str());
  std::cout << "\nGPR confidence RMSEs:";
  dispVector(gprRMSEs);

  //compute optimal confidence
  std::vector<double> optRMSEs = confOptBucketRMSEsWInVal(origModel, fullBestModel, 
    params.nUsers, params.nItems, 10, invalidUsers, invalidItems);
  prefix = std::string(params.prefix) + "_optconf_bucket.txt";
  writeVector(optRMSEs, prefix.c_str());
  std::cout << "\nOptimal confidence RMSEs:";
  dispVector(optRMSEs);

  //compute ppr confidence
  std::vector<double> pprRMSEs = pprBucketRMSEsWInVal(origModel, fullBestModel, 
    params.nUsers, params.nItems, params.alpha, MAX_PR_ITER, 
    data.graphMat, 10, invalidUsers, invalidItems);
  prefix = std::string(params.prefix) + "_pprconf_bucket.txt";
  writeVector(pprRMSEs, prefix.c_str());
  std::cout << "\nPPR confidence RMSEs:";
  dispVector(pprRMSEs);
}


void computeConfCurve(Data& data, Params& params) {

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
  fullModel.train(data, fullBestModel);

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

  std::unordered_set<int> invalidUsers;
  std::unordered_set<int> invalidItems;
  
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

  //compute confidence using the best models for 10 buckets
  std::vector<double> confCurve = computeModConf(data.testMat, bestModels, 
      invalidUsers, invalidItems, origModel, 
      fullBestModel, 10, 0.05);
  std::cout << "\nConfidence bucket Curve: ";
  dispVector(confCurve);
  prefix = std::string(params.prefix) + "_mconf_bucket.txt";
  writeVector(confCurve, prefix.c_str());

  //compute global page rank confidence
  //NOTE: using params.alpha as (1 - restartProb)
  std::vector<double> gprCurve = computeGPRConf(data.testMat, data.graphMat,
      invalidUsers, invalidItems, params.alpha, MAX_PR_ITER, origModel, 
      fullBestModel, 10, 0.05);
  prefix = std::string(params.prefix) + "_gprconf_bucket.txt";
  writeVector(gprCurve, prefix.c_str());
  std::cout << "\nGPR confidence Curve:";
  dispVector(gprCurve);

  //compute ppr confidence
  /*
  std::vector<double> pprCurve = computePPRConf(data.testMat, data.graphMat,
      invalidUsers, invalidItems, params.alpha, MAX_PR_ITER, origModel, 
      fullBestModel, 10, 0.05);
  prefix = std::string(params.prefix) + "_pprconf_bucket.txt";
  writeVector(pprCurve, prefix.c_str());
  std::cout << "\nPPR confidence Curve:";
  dispVector(pprCurve);
  */
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


void computeConfScores(Data& data, Params& params) {
 
  std::vector<Model> bestModels;
  bestModels.push_back(Model(params, "multiconf_partial_2_uFac_50000_5_0.001000.mat",
        "multiconf_partial_2_iFac_19964_5_0.001000.mat", params.seed));
  bestModels.push_back(Model(params, "multiconf_partial_3_uFac_50000_5_0.001000.mat",
        "multiconf_partial_3_iFac_19964_5_0.001000.mat", params.seed));
  bestModels.push_back(Model(params, "multiconf_partial_4_uFac_50000_5_0.001000.mat",
        "multiconf_partial_4_iFac_19964_5_0.001000.mat", params.seed));
  bestModels.push_back(Model(params, "multiconf_partial_5_uFac_50000_5_0.001000.mat",
        "multiconf_partial_5_iFac_19964_5_0.001000.mat", params.seed));
  bestModels.push_back(Model(params, "multiconf_partial_6_uFac_50000_5_0.001000.mat",
        "multiconf_partial_6_iFac_19964_5_0.001000.mat", params.seed));
  
  std::cout << "\nnBestModels: " << bestModels.size();

  Model fullModel(params, params.seed);
  fullModel.load(params.initUFacFile, params.initIFacFile);
  Model origModel(params, params.seed);
  origModel.load(params.origUFacFile, params.origIFacFile);

  std::vector<int> invalUsersVec = readVector("tempInval_invalUsers.txt");
  std::vector<int> invalItemsVec = readVector("tempInval_invalItems.txt");


  std::cout << "\nnInvalidUsers: " << invalUsersVec.size();
  std::cout << "\nnInvalidItems: " << invalItemsVec.size() <<std::endl;

  std::unordered_set<int> invalUsers;
  for (auto v: invalUsersVec) {
    invalUsers.insert(v);
  }

  std::unordered_set<int> invalItems;
  for (auto v: invalItems) {
    invalItems.insert(v);
  }

  std::string prefix = std::string(params.prefix) + "_mConfs.txt";
  std::vector<double> confRMSEs = confBucketRMSEsWInvalOpPerUser(origModel, fullModel, 
      bestModels, params.nUsers, params.nItems, 10, invalUsers, invalItems, prefix);
  dispVector(confRMSEs);
  prefix = std::string(params.prefix) + "_conf_bucket.txt";
  writeVector(confRMSEs, prefix.c_str());
}


int main(int argc , char* argv[]) {

  //get passed parameters
  Params params = parse_cmd_line(argc, argv);

  //initialize seed
  std::srand(params.seed);

  Data data (params);

  //computeConf(data, params);
  computeConfCurve(data, params);
  //computeConfScores(data, params);
  //computePRScores2(data, params);
  //computeGPRScores(data, params);
  //computeOptScores(data, params);

  //writeTrainTestMat(data.trainMat, "ratings_u20_i20_706X1248.syn.train.csr", 
  //    "ratings_u20_i20_706X1248.syn.test.csr", 0.1, params.seed);
  //writeCSRWSparsityStructure(data.trainMat, "ratings_u20_i20_706X1248.syn.csr", 
  //    data.origUFac, data.origIFac, 5);
  //writeCSRWHalfSparsity(data.trainMat, "mat.csr", 0, 10000, 0, 10000);

  /*
  int uStart = 0, uEnd = 10000;
  int iStart = 0, iEnd = 10000;

  std::cout << "\nnnz: " << nnzSubMat(data.trainMat, 0, data.trainMat->nrows, 
                                      0, data.trainMat->ncols)
    << " nnz submat: " << nnzSubMat(data.trainMat, uStart, uEnd, iStart, iEnd);


    //create mf model instance
    ModelMF trainModel(params, params.seed);
    //trainModel.load(params.initUFacFile, params.initIFacFile);

    //create mf model instance to store the best model
    ModelMF bestModel(trainModel);
    //bestModel.load(params.initUFacFile, params.initIFacFile);
    
    trainModel.train(data, bestModel);
    //trainModel.subTrain(data, bestModel, uStart, uEnd, iStart, iEnd);
    //trainModel.fixTrain(data, bestModel, uStart, uEnd, iStart, iEnd);

    //std::string prefix(params.prefix);
    //bestModel.save(prefix);
  */  
  /*
    std::cout << "\n subMat Non-Obs RMSE("<<uStart<<","<<uEnd<<","<<iStart<<","<<iEnd<<"): "
              << bestModel.subMatKnownRankNonObsErr(data, uStart, uEnd, iStart, iEnd) 
              << "\n subMat Non-Obs RMSE("<<uEnd<<","<<data.nUsers<<","<<iStart<<","<<iEnd<<"): "
              << bestModel.subMatKnownRankNonObsErr(data, uEnd, data.nUsers, iStart, iEnd) 
              << "\n subMat Non-Obs RMSE("<<uStart<<","<<uEnd<<","<<iEnd<<","<<data.nItems<<"): "
              << bestModel.subMatKnownRankNonObsErr(data, uStart, uEnd, iEnd, data.nItems)
              << "\n subMat Non-Obs RMSE("<<uEnd<<","<<data.nUsers<<","<<iEnd<<","<<data.nItems<<"): "
              << bestModel.subMatKnownRankNonObsErr(data, uEnd, data.nUsers, iEnd, data.nItems)
              << std::endl;
    */
    /*
    double subMatNonObsRMSE = bestModel.subMatKnownRankNonObsErr(data, uStart, uEnd, 
                                                                    iStart, iEnd);
    double matNonObsRMSE = bestModel.subMatKnownRankNonObsErr(data, 0, params.nUsers, 
                                                                    0, params.nItems);

    std::cout << "\nsubMatNonObs RMSE: " << subMatNonObsRMSE;
    std::cout << "\nmatNonObs RMSE: " << matNonObsRMSE;
    std::cout << "\nResult: " << " " << subMatNonObsRMSE << " " 
      << matNonObsRMSE <<  std::endl;
    //knownLowRankEval2(data, bestModel, params); 
  */
  return 0;
}


