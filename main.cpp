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
#include "topBucketComp.h"
#include "longTail.h"
#include "analyzeModels.h"

Params parse_cmd_line(int argc, char *argv[]) {
  
  if (argc < 22) {
    std::cout << "\nNot enough arguments";
    exit(0);
  }  
  
  Params params(atoi(argv[1]), atoi(argv[2]), atoi(argv[3]), atoi(argv[4]), 
      atoi(argv[5]), atoi(argv[6]), atoi(argv[7]),
      atof(argv[8]), atof(argv[9]), atof(argv[10]), atof(argv[11]), atof(argv[12]),
      argv[13], argv[14], argv[15], argv[16], argv[17], argv[18], argv[19], argv[20], 
      argv[21]);

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
  std::cout << "\nFull RMSE: " << bestModel.fullLowRankErr(data, invalidUsers, invalidItems);
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
  std::unordered_set<int> filtItems;
  //auto headItems = getHeadItems(data.trainMat, 0.2); 
  //filtItems = headItems;

  //add top 100 frequent items to filtItems
  for (auto&& pair: itemFreqPairs) {
    filtItems.insert(pair.first);
    if (filtItems.size() == 100) {
      break;
    }
  }


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


void transformBinData(Data& data, Params& params) {
  std::cout << "\nCreating original model to transform binary ratings..." << std::endl;
  ModelMF origModel(params, params.origUFacFile, params.origIFacFile, 
      params.seed);
  origModel.updateMatWRatings(data.trainMat);
  gk_csr_CreateIndex(data.trainMat, GK_CSR_COL);
  origModel.updateMatWRatings(data.testMat);
  gk_csr_CreateIndex(data.testMat, GK_CSR_COL);
  origModel.updateMatWRatings(data.valMat);
  gk_csr_CreateIndex(data.valMat, GK_CSR_COL);
}


int main(int argc , char* argv[]) {
 
  /* 
  gk_csr_t *mat1 = gk_csr_Read(argv[1], GK_CSR_FMT_CSR, 1, 0);
  gk_csr_t *mat2 = gk_csr_Read(argv[2], GK_CSR_FMT_CSR, GK_CSR_IS_VAL, 0);
  std::cout << "train test mat match: " <<  compMat(mat1, mat2) << std::endl;
  return 0;
  */

  //get passed parameters
  Params params = parse_cmd_line(argc, argv);
  params.display();
  
  
  //initialize seed
  std::srand(params.seed);

  Data data (params);
  
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
  */

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
 
  //std::string graphFName = params.prefix + std::string(".train.jacSim.metis");
  //writeItemJaccSimMat(data.trainMat, graphFName.c_str());
  
  //writeItemJaccSimFrmCorat(data.trainMat, data.graphMat, 
  //    "ratings_26779x26779_25.syn.trainItems.jacSim2.metis");
  //writeCoRatings(data.trainMat, "y_u2_i34_100Kx50K.train.coRatings");
   //ModelMF mfModel(params, params.initUFacFile, 
  //    params.initIFacFile, params.seed);
 
  /* 
  std::string ans;
  std::cout << "Want to train? ";
  std::getline(std::cin, ans);
  std::cout << "you entered: " << ans << std::endl;
  if (!(ans.compare("yes") == 0 || ans.compare("y") == 0)) {
    return 0;
  }
  */
  
    
  std::cout << "ifUISorted: " << checkIfUISorted(data.trainMat) << std::endl ;
  
  if (!GK_CSR_IS_VAL) {
    transformBinData(data, params);
  }

  ModelMF mfModel(params, params.seed);
  //initialize model with svd
  svdFrmSvdlibCSREig(data.trainMat, mfModel.facDim, mfModel.uFac, mfModel.iFac, false);
  //initialize MF model with last learned model if any
  //mfModel.loadFacs(params.prefix);

  std::unordered_set<int> invalidUsers;
  std::unordered_set<int> invalidItems;

  ModelMF bestModel(mfModel);
  std::cout << "\nStarting model train...";
  mfModel.train(data, bestModel, invalidUsers, invalidItems);
  std::cout << "\nTest RMSE: " << bestModel.RMSE(data.testMat, invalidUsers, 
      invalidItems);
  std::cout << "\nValidation RMSE: " << bestModel.RMSE(data.valMat, invalidUsers, 
      invalidItems);
  
  std::string modelSign = bestModel.modelSignature();

  //write out invalid users
  std::string prefix = std::string(params.prefix) + "_" + modelSign + "_invalUsers.txt";
  writeContainer(begin(invalidUsers), end(invalidUsers), prefix.c_str());

  //write out invalid items
  prefix = std::string(params.prefix) + "_" + modelSign + "_invalItems.txt";
  writeContainer(begin(invalidItems), end(invalidItems), prefix.c_str());
  std::cout << std::endl << "**** Model parameters ****" << std::endl;
  mfModel.display();
   

  //computeSampTopNFrmFullModel(data, params);  
  
  //testTailLocRec(data, params);
  //testTailRec(data, params);
  //testRec(data, params);
  //computeHeadTailRMSE(data, params);
  
  //analyzeAccuracy(data, params);
  //compJaccSimAccu(data, params);
  //meanAndVarSameGroundAllUsers(data, params);
  //convertToBin(data, params);

  return 0;
}


