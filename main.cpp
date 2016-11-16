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
#include "modelMFLoc.h"
#include "confCompute.h"
#include "topBucketComp.h"
#include "longTail.h"
#include "analyzeModels.h"

#include <gflags/gflags.h>
DEFINE_uint64(maxiter, 100, "number of iterations");
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
DEFINE_string(method, "sgd", "sgd|hogsgd|als|ccd++");

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
  //svdFrmSvdlibCSRSparsity(data.trainMat, svdModel.facDim, svdModel.uFac, 
  //    svdModel.iFac, true);

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
  

  std::vector<float> maxFreqs = {5, 10 , 15, 20, 30, 50, 75, 100, 150, 200};
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

  //partition the given matrix into train test val
  /*
  gk_csr_t *mat = gk_csr_Read(argv[1], GK_CSR_FMT_CSR, 1, 0);
  writeTrainTestValMat(mat, argv[2], argv[3], argv[4], 0.2, 0.2, atoi(argv[5]));  
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
 
  
  std::cout << "ifUISorted: " << checkIfUISorted(data.trainMat) << std::endl ;
  
  if (!GK_CSR_IS_VAL) {
    transformBinData(data, params);
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
  
  ModelMF mfModel(params, params.seed);
  //initialize model with svd
  //svdFrmSvdlibCSR(data.trainMat, mfModel.facDim, mfModel.uFac, mfModel.iFac, false);
  //initialize MF model with last learned model if any
  mfModel.loadFacs(params.prefix);

  std::unordered_set<int> invalidUsers;
  std::unordered_set<int> invalidItems;
  
  ModelMF bestModel(mfModel);
  std::cout << "\nStarting model train...";
  if (FLAGS_method == "ccd++") { 
    mfModel.trainCCDPP(data, bestModel, invalidUsers, invalidItems);
  } else if (FLAGS_method == "als") {
    mfModel.trainALS(data, bestModel, invalidUsers, invalidItems);
  } else if (FLAGS_method== "hogsgd")  {
    mfModel.hogTrain(data, bestModel, invalidUsers, invalidItems);
  } else {
    mfModel.train(data, bestModel, invalidUsers, invalidItems);
  }

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
  if (!FLAGS_origufac.empty()) {
    std::cout << "\nFull RMSE: " << 
      bestModel.fullLowRankErr(data, invalidUsers, invalidItems) << std::endl;
  }
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


