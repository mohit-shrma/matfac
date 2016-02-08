#include <iostream>
#include <cstdlib>
#include <future>
#include <chrono>
#include <thread>
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
        std::ref(*pbestModels[thInd]));
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

  //compute confidence using the best models for 10 buckets
  std::vector<double> confRMSEs = confBucketRMSEs(origModel, fullModel, 
      bestModels, params.nUsers, params.nItems, 10);
  std::cout << "\nConfidence bucket RMSEs: ";
  dispVector(confRMSEs);
  std::cout << "\nwriting confidence bucket RMSEs" << std::endl;
  prefix = std::string(params.prefix) + "_conf_bucket.txt";
  writeVector(confRMSEs, prefix.c_str());
}


void computePRScores(Data& data, Params& params) {
  Model fullModel(params, "nf_fix_1.0_mat_uFac_20000_5_0.001000.mat",
      "nf_fix_1.0_mat_iFac_17764_5_0.001000.mat", params.seed);
  Model origModel(params, params.seed);
  origModel.load(params.origUFacFile, params.origIFacFile);
  //NOTE: using params.alpha as (1 - restartProb)
  std::vector<double> pprRMSEs = pprBucketRMSEs(origModel, fullModel, 
    params.nUsers, params.nItems, params.alpha, params.maxIter, 
    data.graphMat, 10);
  dispVector(pprRMSEs);
}


void computeConfScores(Data& data, Params& params) {
 
  std::vector<Model> bestModels;
  bestModels.push_back(Model(params, "conf_0_uFac_20000_5_0.001000.mat",
        "conf_0_iFac_20000_5_0.001000.mat", params.seed));
  bestModels.push_back(Model(params, "conf_1_uFac_20000_5_0.001000.mat",
        "conf_1_iFac_20000_5_0.001000.mat", params.seed));
  bestModels.push_back(Model(params, "conf_2_uFac_20000_5_0.001000.mat",
        "conf_2_iFac_20000_5_0.001000.mat", params.seed));
  bestModels.push_back(Model(params, "conf6_uFac_20000_5_0.001000.mat",
        "conf6_iFac_20000_5_0.001000.mat", params.seed));
  bestModels.push_back(Model(params, "conf7_uFac_20000_5_0.001000.mat",
        "conf7_iFac_20000_5_0.001000.mat", params.seed));
  bestModels.push_back(Model(params, "conf8_uFac_20000_5_0.001000.mat",
        "conf8_iFac_20000_5_0.001000.mat", params.seed));

  std::cout << "\nnBestModels: " << bestModels.size();

  Model fullModel(params, "y_fix_1.0_mat_uFac_20000_5_0.001000.mat",
      "y_fix_1.0_mat_iFac_20000_5_0.001000.mat", params.seed);
  Model origModel(params, params.seed);
  origModel.load(params.origUFacFile, params.origIFacFile);

  std::vector<double> confRMSEs = confBucketRMSEs(origModel, fullModel, 
      bestModels, params.nUsers, params.nItems, 10);
  dispVector(confRMSEs);
}



int main(int argc , char* argv[]) {

  //get passed parameters
  Params params = parse_cmd_line(argc, argv);

  //initialize seed
  std::srand(params.seed);

  Data data (params);

  computeConf(data, params);
  //computeConfScores(data, params);
  //computePRScores(data, params);

  //writeCSRWSparsityStructure(data.trainMat, "mlrand_20kX8324_syn.csr", data.origUFac,
  //    data.origIFac, 5);
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


