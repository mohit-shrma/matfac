#include <iostream>
#include <cstdlib>
#include <future>
#include "datastruct.h"
#include "modelMF.h"
#include "modelMFWtReg.h"
#include "modelMFWtRegArb.h"

Params parse_cmd_line(int argc, char *argv[]) {
  
  if (argc < 12) {
    std::cout << "\nNot enough arguments";
    exit(0);
  } 
  
  Params params(atoi(argv[1]), atoi(argv[2]), atoi(argv[3]), atoi(argv[4]),
      atof(argv[5]), atof(argv[6]), atof(argv[7]), atof(argv[8]), atof(argv[9]),
      argv[10], argv[11]);

  return params;
}


int main(int argc , char* argv[]) {

  //get passed parameters
  Params params = parse_cmd_line(argc, argv);

  Data data (params.trainMatFile, params.testMatFile);


  //create mf model instance
  ModelMFWtRegArb trainWtArbModel(params);
  ModelMFWtReg trainWtModel(params);
  ModelMF trainModel(params);


  //create mf model instance to store the best model
  Model bestWtModel(trainWtModel);
  Model bestWtArbModel(trainWtArbModel); //NOTE: for this alpha is [0,1] TC.
  Model bestModel(trainWtModel);

  //run training asynchronously
  
  std::future<void> futModel(std::async(std::launch::async,
        [&trainModel, &data, &bestModel] (){
          trainModel.train(data, bestModel);}));
  
  std::future<void> futWtModel(std::async(std::launch::async,
        [&trainWtModel, &data, &bestWtModel] (){
          trainWtModel.train(data, bestWtModel);}));
  
  std::future<void> futWtArbModel(std::async(std::launch::async,
        [&trainWtArbModel, &data, &bestWtArbModel] (){
          trainWtArbModel.train(data, bestWtArbModel);}));
  
  //wait for these to finish can use .wait()
  futModel.wait();
  futWtModel.wait();
  futWtArbModel.wait();
  

  std::cout<<"MF model train RMSE: " << bestModel.RMSE(data.trainMat);
  std::cout<<"MF model test RMSE: " << bestModel.RMSE(data.testMat);
  std::cout<<"MF model full RMSE: " << bestModel.fullRMSE(data);
  
  std::cout<<"MF wt model train RMSE: " << bestWtModel.RMSE(data.trainMat);
  std::cout<<"MF wt model test RMSE: " << bestWtModel.RMSE(data.testMat);
  std::cout<<"MF wt model full RMSE: " << bestWtModel.fullRMSE(data);
  
  std::cout<<"MF wt arb model train RMSE: " << bestWtArbModel.RMSE(data.trainMat);
  std::cout<<"MF wt arb model test RMSE: " << bestWtArbModel.RMSE(data.testMat);
  std::cout<<"MF wt arb model full RMSE: " << bestWtArbModel.fullRMSE(data);

  return 0;
}


