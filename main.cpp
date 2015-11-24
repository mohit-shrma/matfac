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


int main(int argc , char* argv[]) {

  //get passed parameters
  Params params = parse_cmd_line(argc, argv);

  Data data (params);

  /*
  //create mf model instance
  ModelMFWtRegArb trainModel(params);
  //ModelMFWtReg trainModel(params);
  //ModelMF trainModel(params);


  //create mf model instance to store the best model
  Model bestModel(trainModel);

  //run training asynchronously
  trainModel.train(data, bestModel);
 
  std::cout<<"\nRE: " << params.facDim << " " << params.uReg << " " 
            << params.iReg << " " << params.rhoRMS << " " << params.alpha << " " 
            << bestModel.RMSE(data.trainMat)  << " "
            << bestModel.RMSE(data.testMat)   << " "
            << bestModel.fullRMSE(data) << std::endl;
  */

  return 0;
}


