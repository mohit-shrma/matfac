#include <iostream>
#include <cstdlib>
#include <future>
#include <chrono>
#include <thread>
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

  //compute metric async
  std::future<double> trainRMSEFut(std::async(std::launch::async,
        [&bestModel, &data](){return bestModel.RMSE(data.trainMat);}));
  std::future<double> LowRankRMSEFut(std::async(std::launch::async,
        [&bestModel, &data](){return bestModel.fullLowRankErr(data);}));
  std::future<double> D00RMSEFut(std::async(std::launch::async,
        [&bestModel, &data](){return bestModel.subMatKnownRankErr(
          data, 0, 499, 0, 449);}));
  std::future<double> D11RMSEFut(std::async(std::launch::async,
        [&bestModel, &data](){return bestModel.subMatKnownRankErr(
          data, 500, 1000, 450, 900);}));
  std::future<double> S01RMSEFut(std::async(std::launch::async,
        [&bestModel, &data](){return bestModel.subMatKnownRankErr(
          data, 0, 499, 450, 900);}));
  std::future<double> S10RMSEFut(std::async(std::launch::async,
        [&bestModel, &data](){return bestModel.subMatKnownRankErr(
          data, 500, 1000, 0, 449);}));
  std::future<double> GluRMSEFut(std::async(std::launch::async,
        [&bestModel, &data](){return bestModel.subMatKnownRankErr(
          data, 0, 1000, 901, 1000);}));
  


  std::cout<<"\nRE: " << params.facDim << " " << params.uReg << " " 
            << params.iReg << " " << params.rhoRMS << " " << params.alpha << " " 
            << trainRMSEFut.get() << " " << LowRankRMSEFut.get() << " "
            << D00RMSEFut.get() << " " << D11RMSEFut.get() << " " << " " 
            << S01RMSEFut.get() << " " << S10RMSEFut.get() << std::endl;
}


int main(int argc , char* argv[]) {

  //get passed parameters
  Params params = parse_cmd_line(argc, argv);

  Data data (params);

  //create mf model instance
  ModelMFWtRegArb trainModel(params);
  //ModelMFWtReg trainModel(params);
  //ModelMF trainModel(params);


  //create mf model instance to store the best model
  Model bestModel(trainModel);

  trainModel.train(data, bestModel);

  knownLowRankEval(data, bestModel, params); 

  return 0;
}


