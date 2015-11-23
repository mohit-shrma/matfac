#include <iostream>
#include <cstdlib>
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


int main(int argc, char* argv[]) {

  //get passed parameters
  Params params = parse_cmd_line(argc, argv);

  Data data (params.trainMatFile, params.testMatFile);


  //create mf model instance
  ModelMFWtReg trainModel(params);

  //create mf model instance to store the best model
  Model bestModel(trainModel);

  //run training
  trainModel.train(data, bestModel);

  double trainErr = bestModel.RMSE(data.trainMat);
  double testErr  = bestModel.RMSE(data.testMat);

  std::cout << "\nTrain Err: " << trainErr << "\nTest Err: " << testErr;

  return 0;
}


