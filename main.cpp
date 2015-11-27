#include <iostream>
#include <cstdlib>
#include <future>
#include <chrono>
#include <thread>
#include "util.h"
#include "datastruct.h"
#include "modelMF.h"
#include "modelMFWtReg.h"
#include "modelMFWtRegArb.h"

Params parse_cmd_line(int argc, char *argv[]) {
  
  if (argc < 15) {
    std::cout << "\nNot enough arguments";
    exit(0);
  } 
  
  Params params(atoi(argv[1]), atoi(argv[2]), atoi(argv[3]), atoi(argv[4]), 
      atoi(argv[5]),
      atof(argv[6]), atof(argv[7]), atof(argv[8]), atof(argv[9]), atof(argv[10]),
      argv[11], argv[12], argv[13], argv[14]);

  return params;
}


void knownLowRankEval(Data& data, Model& bestModel, Params& params) {

  //compute metrics
  double trainRMSE  = bestModel.RMSE(data.trainMat);
  double loRankRMSE = bestModel.fullLowRankErr(data);
  double D00RMSE    = bestModel.subMatKnownRankErr(data, 0, 499, 0, 449);
  double D11RMSE    = bestModel.subMatKnownRankErr(data, 500, 999, 450, 899);
  double S01RMSE    = bestModel.subMatKnownRankErr(data, 0, 499, 450, 899);
  double S10RMSE    = bestModel.subMatKnownRankErr(data, 500, 999, 0, 449);
  double GluRMSE    = bestModel.subMatKnownRankErr(data, 0, 999, 900, 999);

  std::cout.precision(5);
  std::cout<<"\nRE: " << std::fixed << params.facDim << " " << params.uReg << " " 
            << params.iReg << " " << params.rhoRMS << " " << params.alpha << " " 
            << trainRMSE << " " << loRankRMSE << " "
            << D00RMSE << " " << D11RMSE << " " << " " 
            << S01RMSE << " " << S10RMSE << " " << GluRMSE << " " 
            << meanRating(data.trainMat) << " " 
            << data.meanKnownSubMatRat(0, 999, 0, 999) << std::endl;
}


int main(int argc , char* argv[]) {

  //get passed parameters
  Params params = parse_cmd_line(argc, argv);

  Data data (params);

  //create mf model instance
  //ModelMFWtRegArb trainModel(params);
  //ModelMFWtReg trainModel(params);
  ModelMF trainModel(params);


  //create mf model instance to store the best model
  Model bestModel(trainModel);

  trainModel.train(data, bestModel);

  knownLowRankEval(data, bestModel, params); 

  return 0;
}


