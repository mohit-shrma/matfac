#include <iostream>
#include <cstdlib>
#include <future>
#include "datastruct.h"
#include "modelMF.h"
#include "modelMFWtReg.h"

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
  ModelMFWtReg trainWtModel(params);
  ModelMF trainModel(params);


  //create mf model instance to store the best model
  Model bestWtModel(trainWtModel);
  Model bestModel(trainWtModel);

  //run training asynchronously
  std::future<void> futModel(std::async(
        [&trainModel, &data, &bestModel] (){
          trainModel.train(data, bestModel);}));
  std::future<void> futWtModel(std::async(
        [&trainWtModel, &data, &bestWtModel] (){
          trainWtModel.train(data, bestWtModel);}));
  
  //wait for these to finish can use .wait()
  futModel.get();
  futWtModel.get();

  std::cout<<"MF model full RMSE: " << bestModel.fullRMSE(data);
  std::cout<<"MF wt model full RMSE: " << bestWtModel.fullRMSE(data);

  return 0;
}


