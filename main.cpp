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
  
  if (argc < 16) {
    std::cout << "\nNot enough arguments";
    exit(0);
  }  
  
  Params params(atoi(argv[1]), atoi(argv[2]), atoi(argv[3]), atoi(argv[4]), 
      atoi(argv[5]), atoi(argv[6]),
      atof(argv[7]), atof(argv[8]), atof(argv[9]), atof(argv[10]), atof(argv[11]),
      argv[12], argv[13], NULL, NULL);
      //argv[12], argv[13], argv[14], argv[15]);

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


int main(int argc , char* argv[]) {

  //get passed parameters
  Params params = parse_cmd_line(argc, argv);

  //initialize
  std::srand(params.seed);

  Data data (params);

  //create mf model instance
  //ModelMFWtRegArb trainModel(params);
  //ModelMFWtReg trainModel(params);
  ModelMF trainModel(params);
  
  //create mf model instance to store the best model
  Model bestModel(trainModel);

  trainModel.train(data, bestModel);
  //trainModel.subTrain(data, bestModel, 0, 10000, 0, 10000);

  double subMatRMSE   = bestModel.subMatRMSE(data.trainMat, 0, 10000, 0, 10000);
  double subMatExRMSE = bestModel.subMatExRMSE(data.trainMat, 0, 10000, 0, 10000);
  double matRMSE      = bestModel.RMSE(data.trainMat);

  std::cout << "\nSubmat RMSE: " << subMatRMSE << std::endl;
  std::cout << "\nSubmatEx RMSE: " << subMatExRMSE << std::endl;
  std::cout << "\nMat RMSE: " << matRMSE << std::endl;
  std::cout << "\nRE: " << subMatRMSE << " " << subMatExRMSE << " " << matRMSE << std::endl;
  //knownLowRankEval2(data, bestModel, params); 

  return 0;
}


