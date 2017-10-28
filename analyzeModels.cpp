#include "analyzeModels.h"


void compJaccSimAccu(Data& data, Params& params) {

  std::vector<ModelMF> mfModels;
  std::vector<ModelMF> origModels;
  for (int i = 1; i < 3; i++) {
    std::string prefix = "mf_0_0_10_" + std::to_string(i);
    ModelMF fullModel(params, params.seed);
    fullModel.loadFacs(prefix.c_str());
    mfModels.push_back(fullModel);

    std::string uFName = "uFac_" + std::to_string(fullModel.nUsers) + "_10_0.txt";
    std::string iFName = "iFac_" + std::to_string(fullModel.nItems+1) + "_10_0.txt";
    ModelMF origModel(params, uFName.c_str(), iFName.c_str(), params.seed);
    origModels.push_back(origModel);
  }
  
  /*
  for (int i = 0; i < 2; i++) {
    std::string prefix = "mfrand_" + std::to_string(i) + "_0_20";
    ModelMF fullModel(params, params.seed);
    fullModel.loadFacs(prefix.c_str());
    mfModels.push_back(fullModel);

    std::string uFName = "uFac_" + std::to_string(fullModel.nUsers) + "_20_" 
      + std::to_string(i) + ".txt";
    std::string iFName = "iFac_" + std::to_string(fullModel.nItems) + "_20_" 
      + std::to_string(i) + ".txt";
    ModelMF origModel(params, uFName.c_str(), iFName.c_str(), params.seed);
    origModels.push_back(origModel);
  }
  */

  int nModels    = mfModels.size();

  std::unordered_set<int> invalidUsers;
  std::unordered_set<int> invalidItems;
  std::string modelSign = mfModels[0].modelSignature();
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

  for (int i = 0; i < nModels; i++) {
    auto& fullModel = mfModels[i];
    auto& origModel = origModels[i];
    std::cout << "Model: " << i << std::endl;
    std::cout << "Train RMSE: " << fullModel.RMSE(data.trainMat, invalidUsers, 
        invalidItems, origModel) << std::endl;
    std::cout << "Test RMSE: " << fullModel.RMSE(data.testMat, invalidUsers, 
        invalidItems, origModel) << std::endl;
    std::cout << "Val RMSE: " << fullModel.RMSE(data.valMat, invalidUsers, 
        invalidItems, origModel) << std::endl;
    //std::cout << "Full RMSE: " << fullModel.fullLowRankErr(data, invalidUsers, 
    //    invalidItems, origModel) << std::endl;
  }

  std::vector<double> epsilons = {0.025, 0.05, 0.1, 0.25, 0.5, 1.0};
  //std::vector<double> epsilons = {0.5};
  int nUsers     = data.trainMat->nrows;
  int nItems     = data.trainMat->ncols;

  std::cout << "nModels: " << nModels << std::endl;

  for (auto&& epsilon: epsilons) {
    std::cout << "epsilon: " << epsilon << std::endl;
    std::vector<std::vector<float>> itemsJacSims(nItems);
    std::vector<std::vector<float>> itemAccuCount(nItems);
    std::vector<std::vector<float>> itemPearsonCorr(nItems);

#pragma omp parallel for
    for (int item = 0; item < nItems; item++) {
      if (invalidItems.count(item) > 0) {
        continue;
      }
      
      std::vector<std::unordered_set<int>> modelAccuItems(nModels);
      std::vector<std::vector<float>> itemErr(nModels);
      std::vector<float> itemErrMean(nModels, 0);
      std::vector<bool> ratedUsers(nUsers, false);
      int count = 0;

      for (int uu = data.trainMat->colptr[item]; 
          uu < data.trainMat->colptr[item+1]; uu++) {
        ratedUsers[data.trainMat->colind[uu]] = true;
      }

      for (int user = 0; user < nUsers; user++) {
        
        if (invalidUsers.count(user) > 0) {
          continue;
        }

        if (ratedUsers[user]) {
          continue;
        }
        
        for (int i = 0; i < nModels; i++) {
          auto& mfModel = mfModels[i];
          auto& origModel = origModels[i];
          float r_ui_est = mfModel.estRating(user, item);
          float r_ui = origModel.estRating(user, item);
          float diff = fabs(r_ui - r_ui_est);
          itemErr[i].push_back(diff);
          itemErrMean[i] += diff;
          if (diff <= epsilon) {
             modelAccuItems[i].insert(user);
          }
        }
        count++;
      }
      
      for (int i = 0; i < nModels; i++) {
        itemErrMean[i] /= count;
      }

      for (int i = 0; i < nModels; i++) {
        itemAccuCount[item].push_back(modelAccuItems[i].size());
        for (int j = i+1; j < nModels; j++) {
          //compute overlap between model i and model j
          float intersectCount = (float) setIntersect(modelAccuItems[i], modelAccuItems[j]);
          float unionCount = (float) setUnion(modelAccuItems[i], modelAccuItems[j]);
          float jacSim = 0;
          if (unionCount > 0) {
            jacSim = intersectCount/unionCount;
          }
          itemsJacSims[item].push_back(jacSim);
          //compute correlation between model i & j for the item
          itemPearsonCorr[item].push_back(pearsonCorr(itemErr[i], itemErr[j], 
                itemErrMean[i], itemErrMean[j])); 
        }
      }

    }
    

    std::string opFName = std::string(params.prefix) + "_" + 
      std::to_string(epsilon) + "_modelsJacSim.txt";
    
    std::string opFName2 = std::string(params.prefix) + "_" + 
      std::to_string(epsilon) + "_modelsPearson.txt";

    std::cout << "Writing... " << opFName << std::endl;

    std::ofstream opFile(opFName.c_str());
    std::ofstream opFile2(opFName2.c_str());

    for (int item  = 0; item < nItems; item++) {
      opFile << item << " ";
      for (auto&& sim: itemsJacSims[item]) {
        opFile << sim << " ";
      }
      for (auto&& accu: itemAccuCount[item]) {
        opFile << accu << " ";
      }
      opFile << std::endl;

      opFile2 << item << " ";
      for (auto&& corr: itemPearsonCorr[item]) {
        opFile2 << corr << " ";
      }
      opFile2 << std::endl;
    }

    opFile.close();
    opFile2.close();
  }

}


void compJaccSimAccuSingleOrigModel(Data& data, Params& params) {

  std::vector<ModelMF> mfModels;
  ModelMF origModel(params, params.origUFacFile, params.origIFacFile,
      params.seed);

  for (int i = 1; i <= 10; i++) {
    std::string prefix = std::string(params.prefix) + "_" + std::to_string(i);
    ModelMF fullModel(params, params.seed);
    fullModel.loadFacs(prefix.c_str());
    mfModels.push_back(fullModel);
  }
  
  int nModels    = mfModels.size();

  std::unordered_set<int> invalidUsers;
  std::unordered_set<int> invalidItems;
  std::string modelSign = mfModels[0].modelSignature();
  std::cout << "\nModel sign: " << modelSign << std::endl;    
  std::string prefix = std::string(params.prefix) + "_" + 
    std::to_string(params.facDim) + "_" + modelSign + "_invalUsers.txt";
  std::vector<int> invalUsersVec = readVector(prefix.c_str());
  prefix = std::string(params.prefix) + "_" + 
    std::to_string(params.facDim) + "_" + modelSign + "_invalItems.txt";
  std::vector<int> invalItemsVec = readVector(prefix.c_str());
  for (auto v: invalUsersVec) {
    invalidUsers.insert(v);
  }
  for (auto v: invalItemsVec) {
    invalidItems.insert(v);
  }

  for (int i = 0; i < nModels; i++) {
    auto& fullModel = mfModels[i];
    std::cout << "Model: " << i << std::endl;
    std::cout << "Train RMSE: " << fullModel.RMSE(data.trainMat, invalidUsers, 
        invalidItems, origModel) << std::endl;
    std::cout << "Test RMSE: " << fullModel.RMSE(data.testMat, invalidUsers, 
        invalidItems, origModel) << std::endl;
    std::cout << "Val RMSE: " << fullModel.RMSE(data.valMat, invalidUsers, 
        invalidItems, origModel) << std::endl;
    //std::cout << "Full RMSE: " << fullModel.fullLowRankErr(data, invalidUsers, 
    //    invalidItems, origModel) << std::endl;
  }

  std::vector<double> epsilons = {0.1, 0.25, 0.5, 1.0};
  //std::vector<double> epsilons = {0.5};
  int nUsers     = data.trainMat->nrows;
  int nItems     = data.trainMat->ncols;

  std::cout << "nModels: " << nModels << std::endl;

  for (auto&& epsilon: epsilons) {
    std::cout << "epsilon: " << epsilon << std::endl;
    std::vector<std::vector<float>> itemsJacSims(nItems);
    std::vector<std::vector<float>> itemAccuCount(nItems);
    std::vector<std::vector<float>> itemPearsonCorr(nItems);

#pragma omp parallel for
    for (int item = 0; item < nItems; item++) {
      if (invalidItems.count(item) > 0) {
        continue;
      }
      
      std::vector<std::unordered_set<int>> modelAccuItems(nModels);
      std::vector<std::vector<float>> itemErr(nModels);
      std::vector<float> itemErrMean(nModels, 0);
      std::vector<bool> ratedUsers(nUsers, false);
      int count = 0;

      for (int uu = data.trainMat->colptr[item]; 
          uu < data.trainMat->colptr[item+1]; uu++) {
        ratedUsers[data.trainMat->colind[uu]] = true;
      }

      for (int user = 0; user < nUsers; user++) {
        
        if (invalidUsers.count(user) > 0) {
          continue;
        }

        if (ratedUsers[user]) {
          continue;
        }
        
        for (int i = 0; i < nModels; i++) {
          auto& mfModel  = mfModels[i];
          float r_ui_est = mfModel.estRating(user, item);
          float r_ui     = origModel.estRating(user, item);
          float diff     = fabs(r_ui - r_ui_est);
          itemErrMean[i] += diff;
          itemErr[i].push_back(diff);
          if (diff <= epsilon) {
             modelAccuItems[i].insert(user);
          }
        }
        count++;
      }
      
      for (int i = 0; i < nModels; i++) {
        itemErrMean[i] /= count;
      }

      for (int i = 0; i < nModels; i++) {
        itemAccuCount[item].push_back(modelAccuItems[i].size());
        for (int j = i+1; j < nModels; j++) {
          //compute overlap between model i and model j
          float intersectCount = (float) setIntersect(modelAccuItems[i], modelAccuItems[j]);
          float unionCount = (float) setUnion(modelAccuItems[i], modelAccuItems[j]);
          float jacSim = 0;
          if (unionCount > 0) {
            jacSim = intersectCount/unionCount;
          }
          itemsJacSims[item].push_back(jacSim);
          //compute correlation between model i & j for the item
          itemPearsonCorr[item].push_back(pearsonCorr(itemErr[i], itemErr[j], 
                itemErrMean[i], itemErrMean[j])); 
        }
      }

    }
    
    std::string opFName = std::string(params.prefix) + "_" + 
      std::to_string(epsilon) + "_modelsJacSim.txt";
    
    std::string opFName2 = std::string(params.prefix) + "_" + 
      std::to_string(epsilon) + "_modelsPearson.txt";

    std::cout << "Writing... " << opFName << std::endl;

    std::ofstream opFile(opFName.c_str());
    std::ofstream opFile2(opFName2.c_str());

    for (int item  = 0; item < nItems; item++) {
      opFile << item << " ";
      for (auto&& sim: itemsJacSims[item]) {
        opFile << sim << " ";
      }
      for (auto&& accu: itemAccuCount[item]) {
        opFile << accu << " ";
      }
      opFile << std::endl;

      opFile2 << item << " ";
      for (auto&& corr: itemPearsonCorr[item]) {
        opFile2 << corr << " ";
      }
      opFile2 << std::endl;
    }

    opFile.close();
    opFile2.close();
  }

}


void averageModels(Data& data, Params& params) {

  ModelMF origModel(params, params.origUFacFile, params.origIFacFile,
      params.seed);

  ModelMF alsModel(params, 
      "als_0_1_uFac_229060X26779_20_0.010000_0.010000_0.001000.mat", 
      "als_0_1_iFac_229060X26779_20_0.010000_0.010000_0.001000.mat",
      params.seed
      );

  ModelMF ccdppModel(params, 
      "ccd++_0_1_uFac_229060X26779_20_0.010000_0.010000_0.001000.mat",
      "ccd++_0_1_iFac_229060X26779_20_0.010000_0.010000_0.001000.mat",
      params.seed
      );

  ModelMF sgdModel(params, 
      "sgdpar_0_1_uFac_229060X26779_20_0.010000_0.010000_0.001000.mat",
      "sgdpar_0_1_iFac_229060X26779_20_0.010000_0.010000_0.001000.mat",
      params.seed
      );


  std::unordered_set<int> invalidUsers;
  std::unordered_set<int> invalidItems;
  std::string modelSign = alsModel.modelSignature();
  std::cout << "\nModel sign: " << modelSign << std::endl;    
  std::string prefix = "sgdpar_20_1_" + modelSign + "_invalUsers.txt";
  std::vector<int> invalUsersVec = readVector(prefix.c_str());
  prefix = "sgdpar_20_1_" + modelSign + "_invalItems.txt";
  std::vector<int> invalItemsVec = readVector(prefix.c_str());
  for (auto v: invalUsersVec) {
    invalidUsers.insert(v);
  }
  for (auto v: invalItemsVec) {
    invalidItems.insert(v);
  }
 
  std::cout << "No. of invalid users : " << invalidUsers.size() << std::endl;
  std::cout << "No. of invalid items : " << invalidItems.size() << std::endl;

  std::cout << "SGD model: " << std::endl;
  std::cout << "Train RMSE: " << sgdModel.RMSE(data.trainMat, invalidUsers,
      invalidItems, origModel) << std::endl;
  std::cout << "Test RMSE: " << sgdModel.RMSE(data.testMat, invalidUsers, 
      invalidItems, origModel) << std::endl;
  std::cout << "Val RMSE: " << sgdModel.RMSE(data.valMat, invalidUsers,
      invalidItems, origModel) << std::endl;
  std::cout << "Full RMSE: " << sgdModel.fullLowRankErr(data, invalidUsers, 
      invalidItems, origModel) << std::endl;
  std::cout << "Objective: " << sgdModel.objective(data, invalidUsers, invalidItems)
    << std::endl;

  std::cout << "ALS model: " << std::endl;
  std::cout << "Train RMSE: " << alsModel.RMSE(data.trainMat, invalidUsers,
      invalidItems, origModel) << std::endl;
  std::cout << "Test RMSE: " << alsModel.RMSE(data.testMat, invalidUsers, 
      invalidItems, origModel) << std::endl;
  std::cout << "Val RMSE: " << alsModel.RMSE(data.valMat, invalidUsers,
      invalidItems, origModel) << std::endl;
  std::cout << "Full RMSE: " << alsModel.fullLowRankErr(data, invalidUsers, 
      invalidItems, origModel) << std::endl;
  std::cout << "Objective: " << alsModel.objective(data, invalidUsers, invalidItems)
    << std::endl;

  std::cout << "CCD++ model: " << std::endl;
  std::cout << "Train RMSE: " << ccdppModel.RMSE(data.trainMat, invalidUsers,
      invalidItems, origModel) << std::endl;
  std::cout << "Test RMSE: " << ccdppModel.RMSE(data.testMat, invalidUsers, 
      invalidItems, origModel) << std::endl;
  std::cout << "Val RMSE: " << ccdppModel.RMSE(data.valMat, invalidUsers,
      invalidItems, origModel) << std::endl;
  std::cout << "Full RMSE: " << ccdppModel.fullLowRankErr(data, invalidUsers, 
      invalidItems, origModel) << std::endl;
  std::cout << "Objective: " << ccdppModel.objective(data, invalidUsers, invalidItems)
    << std::endl;
  

  float epsilon = 0.25;
  int nUsers     = data.trainMat->nrows;
  int nItems     = data.trainMat->ncols;

  std::vector<int> itemFreq(nItems, 0);
  std::vector<int> userFreq(nUsers, 0);

#pragma omp parallel for
  for (int item = 0; item < nItems; item++) {
    itemFreq[item] = (data.trainMat->colptr[item+1] - 
        data.trainMat->colptr[item]);
  }

#pragma omp parallel for
  for (int u = 0; u < nUsers; u++) {
    userFreq[u] = data.trainMat->rowptr[u+1] - data.trainMat->rowptr[u];
  }

  double rmse = 0;
  unsigned long int count = 0;
#pragma omp parallel for reduction(+:rmse,count)  
  for (int item = 0; item < nItems; item++) {

    if (invalidItems.count(item) > 0) {
      continue;
    }
    
    std::vector<bool> ratedUsers(nUsers, false);
    
    for (int uu = data.trainMat->colptr[item]; 
        uu < data.trainMat->colptr[item+1]; uu++) {
      ratedUsers[data.trainMat->colind[uu]] = true;
    }
    
    for (int user = 0; user < nUsers; user++) {
      
      if (invalidUsers.count(user) > 0) {
        continue;
      }

      if (ratedUsers[user]) {
        continue;
      }
      
      float r_ui       = origModel.estRating(user, item);
      float r_ui_sgd   = sgdModel.estRating(user, item);
      float r_ui_als   = alsModel.estRating(user, item);
      float r_ui_ccdpp = ccdppModel.estRating(user, item);
      float r_ui_est   = 0;

      r_ui_est = (r_ui_sgd + r_ui_als + r_ui_ccdpp)/3; 
      /*
      if (itemFreq[user] <= 30 || userFreq[user] <= 30) {
        r_ui_est = (r_ui_sgd + r_ui_als + r_ui_ccdpp)/3; 
      } else {
        r_ui_est = r_ui_ccdpp;
      }
      */

      float diff = (r_ui - r_ui_est);
      rmse  += diff*diff;
      count += 1;
    } 
  }
  
  std::cout << "Averaged RMSE: " << sqrt(rmse/count) << " " << rmse << " " 
    << count << std::endl;

}



void compareModels(Data& data, Params& params) {

  ModelMF origModel(params, params.origUFacFile, params.origIFacFile,
      params.seed);

  ModelMF firstModel(params, 
      "als_0_1_uFac_229060X26779_20_0.010000_0.010000_0.001000.mat", 
      "als_0_1_iFac_229060X26779_20_0.010000_0.010000_0.001000.mat",
      params.seed
      );
  
  ModelMF secondModel(params, 
      "ccd++_0_1_uFac_229060X26779_20_0.010000_0.010000_0.001000.mat",
      "ccd++_0_1_iFac_229060X26779_20_0.010000_0.010000_0.001000.mat",
      params.seed
      );

  std::unordered_set<int> invalidUsers;
  std::unordered_set<int> invalidItems;
  std::string modelSign = firstModel.modelSignature();
  std::cout << "\nModel sign: " << modelSign << std::endl;    
  std::string prefix = "sgdpar_20_1_" + modelSign + "_invalUsers.txt";
  std::vector<int> invalUsersVec = readVector(prefix.c_str());
  prefix = "sgdpar_20_1_" + modelSign + "_invalItems.txt";
  std::vector<int> invalItemsVec = readVector(prefix.c_str());
  for (auto v: invalUsersVec) {
    invalidUsers.insert(v);
  }
  for (auto v: invalItemsVec) {
    invalidItems.insert(v);
  }
 
  std::cout << "No. of invalid users : " << invalidUsers.size() << std::endl;
  std::cout << "No. of invalid itemd : " << invalidItems.size() << std::endl;

  std::cout << "First model: " << std::endl;
  std::cout << "Train RMSE: " << firstModel.RMSE(data.trainMat, invalidUsers,
      invalidItems, origModel) << std::endl;
  std::cout << "Test RMSE: " << firstModel.RMSE(data.testMat, invalidUsers, 
      invalidItems, origModel) << std::endl;
  std::cout << "Val RMSE: " << firstModel.RMSE(data.valMat, invalidUsers,
      invalidItems, origModel) << std::endl;
  std::cout << "Full RMSE: " << firstModel.fullLowRankErr(data, invalidUsers, 
      invalidItems, origModel) << std::endl;
  std::cout << "Objective: " << firstModel.objective(data, invalidUsers, invalidItems)
    << std::endl;

  std::cout << "Second model: " << std::endl;
  std::cout << "Train RMSE: " << secondModel.RMSE(data.trainMat, invalidUsers,
      invalidItems, origModel) << std::endl;
  std::cout << "Test RMSE: " << secondModel.RMSE(data.testMat, invalidUsers, 
      invalidItems, origModel) << std::endl;
  std::cout << "Val RMSE: " << secondModel.RMSE(data.valMat, invalidUsers,
      invalidItems, origModel) << std::endl;
  std::cout << "Full RMSE: " << secondModel.fullLowRankErr(data, invalidUsers, 
      invalidItems, origModel) << std::endl;
  std::cout << "Objective: " << secondModel.objective(data, invalidUsers, invalidItems)
    << std::endl;

  float epsilon = 0.25;
  int nUsers     = data.trainMat->nrows;
  int nItems     = data.trainMat->ncols;

  std::vector<unsigned long int> bothModelsAccuCount(nItems, 0);
  std::vector<unsigned long int> bothModelsInaccuCount(nItems, 0);
  std::vector<unsigned long int> firstModelAccuCount(nItems, 0);
  std::vector<unsigned long int> firstModelInaccuCount(nItems, 0);
  std::vector<unsigned long int> secondModelAccuCount(nItems, 0);
  std::vector<unsigned long int> secondModelInaccuCount(nItems, 0);

#pragma omp parallel for
  for (int item = 0; item < nItems; item++) {
    
    if (invalidItems.count(item) > 0) {
      continue;
    }
    
    std::vector<bool> ratedUsers(nUsers, false);
    
    for (int uu = data.trainMat->colptr[item]; 
        uu < data.trainMat->colptr[item+1]; uu++) {
      ratedUsers[data.trainMat->colind[uu]] = true;
    }

    for (int user = 0; user < nUsers; user++) {
      
      if (invalidUsers.count(user) > 0) {
        continue;
      }
      
      if (ratedUsers[user]) {
        continue;
      }
      
      float r_ui        = origModel.estRating(user, item);
      float r_ui_first  = firstModel.estRating(user, item);
      float r_ui_second = secondModel.estRating(user, item);

      float firstDiff  = fabs(r_ui - r_ui_first);
      float secondDiff = fabs(r_ui - r_ui_second);
     
      if (firstDiff <= epsilon) {
        firstModelAccuCount[item] += 1;
      } else {
        firstModelInaccuCount[item] += 1;
      }
      
      if (secondDiff <= epsilon) {
        secondModelAccuCount[item] += 1;
      } else {
        secondModelInaccuCount[item] += 1;
      }

      if (firstDiff <= epsilon && secondDiff <= epsilon) {
        bothModelsAccuCount[item] += 1;
      } else if (firstDiff > epsilon && secondDiff > epsilon) {
        bothModelsInaccuCount[item] += 1;
      }

    }

  }

  std::string opFName = std::string(params.prefix) + "_firstSecAccuCount.txt"; 
  std::ofstream opFile(opFName.c_str());
  for (int item = 0; item < nItems; item++) {
    if (invalidItems.count(item) == 0) {
      opFile << item << " " << firstModelAccuCount[item] << " " 
        << secondModelAccuCount[item] << " " << bothModelsAccuCount[item] << " "
        << firstModelInaccuCount[item] << " " << secondModelInaccuCount[item] 
        << " " << bothModelsInaccuCount[item] << std::endl;
    }  
  }
  opFile.close();

}


void compJaccSimAccuMeth(Data& data, Params& params) {

  ModelMF origModel(params, params.origUFacFile, params.origIFacFile,
      params.seed);
  ModelMF sgdModel(params, params.seed), alsModel(params, params.seed), ccdModel(params, params.seed);
 
  /*
  ModelMF sgdModel(params, 
      "sgd_10_1_uFac_229060X26779_10_0.010000_0.010000_0.001000.mat", 
      "sgd_10_1_iFac_229060X26779_10_0.010000_0.010000_0.001000.mat",
      params.seed
      );
  ModelMF alsModel(params, 
      "als_10_1_uFac_229060X26779_10_0.010000_0.010000_0.001000.mat", 
      "als_10_1_iFac_229060X26779_10_0.010000_0.010000_0.001000.mat",
      params.seed
      );
  ModelMF ccdModel(params, 
      "ccd++_10_1_uFac_229060X26779_10_0.010000_0.010000_0.001000.mat", 
      "ccd++_10_1_iFac_229060X26779_10_0.010000_0.010000_0.001000.mat",
      params.seed
      );
  
  std::string prefix;
  */
  
  //std::string prefix = "sgdpar_20";// + std::to_string(params.facDim) + "_1";
  std::string prefix = "sgdpar_20_1";
  sgdModel.loadFacs(prefix.c_str());

  //prefix = "als2_20";// + std::to_string(params.facDim) + "_1";
  prefix = "als_20_1";
  alsModel.loadFacs(prefix.c_str());

  //prefix = "ccd++_20";// + std::to_string(params.facDim) + "_1"; 
  prefix = "ccd++_20_1"; 
  ccdModel.loadFacs(prefix.c_str());
  

  std::unordered_set<int> invalidUsers;
  std::unordered_set<int> invalidItems;
  std::string modelSign = sgdModel.modelSignature();
  std::cout << "\nModel sign: " << modelSign << std::endl;    
  prefix = "sgdpar_20_1_" + modelSign + "_invalUsers.txt";
  std::vector<int> invalUsersVec = readVector(prefix.c_str());
  prefix = "sgdpar_20_1_" + modelSign + "_invalItems.txt";
  std::vector<int> invalItemsVec = readVector(prefix.c_str());
  for (auto v: invalUsersVec) {
    invalidUsers.insert(v);
  }
  for (auto v: invalItemsVec) {
    invalidItems.insert(v);
  }
 
  std::cout << "No. of invalid users : " << invalidUsers.size() << std::endl;
  std::cout << "No. of invalid itemd : " << invalidItems.size() << std::endl;


  std::cout << "SGD model: " << std::endl;
  std::cout << "Train RMSE: " << sgdModel.RMSE(data.trainMat, invalidUsers,
      invalidItems, origModel) << std::endl;
  std::cout << "Test RMSE: " << sgdModel.RMSE(data.testMat, invalidUsers, 
      invalidItems, origModel) << std::endl;
  std::cout << "Val RMSE: " << sgdModel.RMSE(data.valMat, invalidUsers,
      invalidItems, origModel) << std::endl;
  std::cout << "Full RMSE: " << sgdModel.fullLowRankErr(data, invalidUsers, 
      invalidItems, origModel) << std::endl;
  std::cout << "Objective: " << sgdModel.objective(data, invalidUsers, invalidItems)
    << std::endl;

  std::cout << "ALS model: " << std::endl;
  std::cout << "Train RMSE: " << alsModel.RMSE(data.trainMat, invalidUsers,
      invalidItems, origModel) << std::endl;
  std::cout << "Test RMSE: " << alsModel.RMSE(data.testMat, invalidUsers, 
      invalidItems, origModel) << std::endl;
  std::cout << "Val RMSE: " << alsModel.RMSE(data.valMat, invalidUsers,
      invalidItems, origModel) << std::endl;
  std::cout << "Full RMSE: " << alsModel.fullLowRankErr(data, invalidUsers, 
      invalidItems, origModel) << std::endl;
  std::cout << "Objective: " << alsModel.objective(data, invalidUsers, invalidItems)
    << std::endl;
  
  std::cout << "CCD++ model: " << std::endl;
  std::cout << "Train RMSE: " << ccdModel.RMSE(data.trainMat, invalidUsers,
      invalidItems, origModel) << std::endl;
  std::cout << "Test RMSE: " << ccdModel.RMSE(data.testMat, invalidUsers, 
      invalidItems, origModel) << std::endl;
  std::cout << "Val RMSE: " << ccdModel.RMSE(data.valMat, invalidUsers,
      invalidItems, origModel) << std::endl;
  std::cout << "Full RMSE: " << ccdModel.fullLowRankErr(data, invalidUsers, 
      invalidItems, origModel) << std::endl;
  std::cout << "Objective: " << ccdModel.objective(data, invalidUsers, invalidItems)
    << std::endl;

  std::vector<double> epsilons = {0.1, 0.25, 0.5, 1.0};
  //std::vector<double> epsilons = {0.5};
  int nUsers     = data.trainMat->nrows;
  int nItems     = data.trainMat->ncols;

  std::vector<int> itemFreq(nItems, 0);
  std::vector<float> meanGTRating(nItems, 0), varianceGTRating(nItems, 0);
  
  std::vector<std::pair<double, double>> trainMeanVar;
  trainMeanVar = trainItemsMeanVar(data.trainMat);
  
  float maxRat = 0, minRat = 100;
  
#pragma omp parallel for reduction(max: maxRat), reduction(min: minRat)
  for (int item = 0; item < nItems; item++) {
    maxRat = 0, minRat = 100;
    itemFreq[item] = (data.trainMat->colptr[item+1] - 
        data.trainMat->colptr[item]);
    
    std::vector<double> userRatings;
    for (int user = 0; user < nUsers; user++) {
      float rating = origModel.estRating(user, item);
      if (maxRat < rating) {maxRat = rating;}
      if (minRat > rating) {minRat = rating;}
      userRatings.push_back(rating);
    }

    std::pair<double, double> meanNStd = meanStdDev(userRatings);
    meanGTRating[item]                 = meanNStd.first;
    varianceGTRating[item]             = meanNStd.second;
  }
  std::cout << "maxRat: " << maxRat << " minRat: " << minRat << std::endl;

  for (auto&& epsilon: epsilons) {
    std::cout << "epsilon: " << epsilon << std::endl;
    std::vector<std::vector<float>> itemsJacSims(nItems);
    std::vector<std::vector<float>> itemAccuCount(nItems);
    std::vector<std::vector<float>> itemPearsonCorr(nItems);
    std::vector<float> sgdItemsAvgRMSE(nItems, 0);
    std::vector<float> sgdItemsAvgPred(nItems, 0);
    std::vector<float> alsItemsAvgRMSE(nItems, 0);
    std::vector<float> alsItemsAvgPred(nItems, 0);
    std::vector<float> ccdItemsAvgRMSE(nItems, 0);
    std::vector<float> ccdItemsAvgPred(nItems, 0);
    std::vector<float> allMethItemsAvgRMSE(nItems, 0);
    
    float rmseAvg = 0;
    unsigned long int nnz = 0;

#pragma omp parallel for reduction(+:rmseAvg, nnz)
    for (int item = 0; item < nItems; item++) {
      if (invalidItems.count(item) > 0) {
        continue;
      }
      
      std::unordered_set<int> sgdAccuItems;
      std::unordered_set<int> alsAccuItems;
      std::unordered_set<int> ccdAccuItems;
      
      std::vector<float> sgdErr, alsErr, ccdErr;
      float sgdErrMean = 0, alsErrMean = 0, ccdErrMean = 0;
      float sgdItemAvgRMSE = 0, alsItemAvgRMSE = 0, ccdItemAvgRMSE = 0;
      float sgdItemAvgPred = 0, alsItemAvgPred = 0, ccdItemAvgPred = 0;
      float allMethItemAvgRMSE = 0;
      std::vector<bool> ratedUsers(nUsers, false);
      unsigned long int count = 0;

      for (int uu = data.trainMat->colptr[item]; 
          uu < data.trainMat->colptr[item+1]; uu++) {
        ratedUsers[data.trainMat->colind[uu]] = true;
      }

      for (int user = 0; user < nUsers; user++) {
        
        if (invalidUsers.count(user) > 0) {
          continue;
        }

        if (ratedUsers[user]) {
          continue;
        }
        
        float r_ui     = origModel.estRating(user, item);
        float r_ui_sgd = sgdModel.estRating(user, item);
        float r_ui_als = alsModel.estRating(user, item);
        float r_ui_ccd = ccdModel.estRating(user, item);
        float r_ui_avg = (r_ui_sgd + r_ui_als + r_ui_ccd)/3;
        
        rmseAvg += (r_ui - r_ui_avg)*(r_ui - r_ui_avg);
        nnz++;
        
        allMethItemAvgRMSE += (r_ui - r_ui_avg)*(r_ui - r_ui_avg);

        float sgdDiff  = fabs(r_ui - r_ui_sgd);
        sgdItemAvgRMSE += sgdDiff*sgdDiff;
        sgdItemAvgPred += fabs(r_ui_sgd);
        sgdErrMean += sgdDiff; 
        sgdErr.push_back(sgdDiff);
        if (sgdDiff <= epsilon) {
          sgdAccuItems.insert(user);
        }
        
        float alsDiff = fabs(r_ui - r_ui_als);
        alsItemAvgRMSE += alsDiff*alsDiff;
        alsItemAvgPred += fabs(r_ui_als);
        alsErrMean += alsDiff;
        alsErr.push_back(alsDiff);
        if (alsDiff <= epsilon) {
          alsAccuItems.insert(user);
        }

        float ccdDiff = fabs(r_ui - r_ui_ccd);
        ccdItemAvgRMSE += ccdDiff*ccdDiff;
        ccdItemAvgPred += fabs(r_ui_ccd);
        ccdErrMean += ccdDiff;
        ccdErr.push_back(ccdDiff);
        if (ccdDiff <= epsilon) {
          ccdAccuItems.insert(user);
        }

        count++;
      }
      
      sgdErrMean /= count;
      alsErrMean /= count;
      ccdErrMean /= count;
      
      sgdItemsAvgRMSE[item] = std::sqrt(sgdItemAvgRMSE/count);
      alsItemsAvgRMSE[item] = std::sqrt(alsItemAvgRMSE/count);
      ccdItemsAvgRMSE[item] = std::sqrt(ccdItemAvgRMSE/count);
      allMethItemsAvgRMSE[item] = std::sqrt(allMethItemAvgRMSE/count);

      sgdItemsAvgPred[item] = sgdItemAvgPred/count; 
      alsItemsAvgPred[item] = alsItemAvgPred/count; 
      ccdItemsAvgPred[item] = ccdItemAvgPred/count; 

      itemAccuCount[item].push_back(sgdAccuItems.size()); 
      itemAccuCount[item].push_back(alsAccuItems.size()); 
      itemAccuCount[item].push_back(ccdAccuItems.size()); 
      
      float sgdALSIntersectCount = (float) setIntersect(sgdAccuItems, alsAccuItems);
      float sgdALSUnionCount = (float) setUnion(sgdAccuItems, alsAccuItems);
      float sgdALSJacSim = 0;
      if (sgdALSUnionCount > 0) {
        sgdALSJacSim = sgdALSIntersectCount/sgdALSUnionCount;
      }
      itemsJacSims[item].push_back(sgdALSJacSim);
      itemPearsonCorr[item].push_back(pearsonCorr(sgdErr, alsErr, sgdErrMean, 
            alsErrMean));

      float alsCCDIntersectCount = (float) setIntersect(alsAccuItems, ccdAccuItems);
      float alsCCDUnionCount = (float) setUnion(alsAccuItems, ccdAccuItems);;
      float alsCCDJacSim = 0; 
      if (alsCCDUnionCount > 0) {
        alsCCDJacSim = alsCCDIntersectCount/alsCCDUnionCount;
      }
      itemsJacSims[item].push_back(alsCCDJacSim);
      itemPearsonCorr[item].push_back(pearsonCorr(alsErr, ccdErr, alsErrMean, 
            ccdErrMean));

      float sgdCCDIntersectCount = (float) setIntersect(sgdAccuItems, ccdAccuItems);
      float sgdCCDUnionCount  = (float) setUnion(sgdAccuItems, ccdAccuItems);
      float sgdCCDJacSim = 0;
      if (sgdCCDUnionCount > 0) {
        sgdCCDJacSim = sgdCCDIntersectCount/sgdCCDUnionCount;
      }
      itemsJacSims[item].push_back(sgdCCDJacSim);
      itemPearsonCorr[item].push_back(pearsonCorr(sgdErr, ccdErr, sgdErrMean, 
            ccdErrMean)); 
      
    }
    
    std::cout << "rmseAvg: " << rmseAvg << " nnz: " << nnz << std::endl;
    std::cout << "rmseAvg: " << std::sqrt(rmseAvg/nnz) << std::endl;

    std::string opFName = std::string(params.prefix) + "_pairwise_" + 
      std::to_string(epsilon) + "_modelsJacSim.txt";
    
    std::string opFName2 = std::string(params.prefix) + "_pairwise_" + 
      std::to_string(epsilon) + "_modelsPearson.txt";
    
    std::string opFName3 = std::string(params.prefix) + "_pairwise_" +
      std::to_string(epsilon) + "_itemAvgRMSE.txt";

    std::cout << "Writing... " << opFName << std::endl;

    std::ofstream opFile(opFName.c_str());
    std::ofstream opFile2(opFName2.c_str());
    std::ofstream opFile3(opFName3.c_str());

    for (int item  = 0; item < nItems; item++) {
      opFile << item << " ";
      for (auto&& sim: itemsJacSims[item]) {
        opFile << sim << " ";
      }
      for (auto&& accu: itemAccuCount[item]) {
        opFile << accu << " ";
      }
      opFile << std::endl;

      opFile2 << item << " ";
      for (auto&& corr: itemPearsonCorr[item]) {
        opFile2 << corr << " ";
      }
      opFile2 << std::endl;
      
      if (invalidItems.count(item) == 0) {
        opFile3 << item << " " 
          << sgdItemsAvgRMSE[item] << " " << sgdItemsAvgPred[item] << " " 
          << alsItemsAvgRMSE[item] << " " << alsItemsAvgPred[item] << " " 
          << ccdItemsAvgRMSE[item] << " " << ccdItemsAvgPred[item] << " "
          << allMethItemsAvgRMSE[item] << " "
          << std::endl;
      }

    }

    opFile.close();
    opFile2.close();
    opFile3.close();
  }

  std::string opFName = std::string(params.prefix) + "_" + 
    "itemFreqMeanVar.txt";
  std::ofstream opFile(opFName.c_str());
  for (int item = 0; item < nItems; item++) {
    opFile << item << " " << itemFreq[item] << " " 
      << trainMeanVar[item].first << " " 
      << trainMeanVar[item].second << " " 
      << meanGTRating[item] << " " 
      << varianceGTRating[item] << std::endl;
  } 
  opFile.close();

}


void analyzeAccuracy(Data& data, Params& params) {
  
  std::vector<ModelMF> mfModels;
  std::vector<ModelMF> origModels;

  for (int i = 1; i < 3; i++) {
    std::string prefix = "mf_0_0_10_" + std::to_string(i);
    ModelMF fullModel(params, params.seed);
    fullModel.loadFacs(prefix.c_str());
    mfModels.push_back(fullModel);

    std::string uFName = "uFac_" + std::to_string(fullModel.nUsers) + "_10_0.txt";
    std::string iFName = "iFac_" + std::to_string(fullModel.nItems+1) + "_10_0.txt";
    ModelMF origModel(params, uFName.c_str(), iFName.c_str(), params.seed);
    origModels.push_back(origModel);
  }

  /*
  for (int i = 0; i < 5; i++) {
    std::string prefix = "mf_" + std::to_string(i) + "_0_10";
    ModelMF fullModel(params, params.seed);
    fullModel.loadFacs(prefix.c_str());
    mfModels.push_back(fullModel);

    std::string uFName = "uFac_" + std::to_string(fullModel.nUsers) + "_10_" 
      + std::to_string(i) + ".txt";
    std::string iFName = "iFac_" + std::to_string(fullModel.nItems) + "_10_" 
      + std::to_string(i) + ".txt";
    ModelMF origModel(params, uFName.c_str(), iFName.c_str(), params.seed);
    origModels.push_back(origModel);
  }
  */

  int nUsers = data.trainMat->nrows;
  int nItems = data.trainMat->ncols;
  auto rowColFreq = getRowColFreq(data.trainMat);
  auto userFreq = rowColFreq.first;
  auto itemFreq = rowColFreq.second;
  //double epsilon = 0.025;
  std::vector<double> epsilons = {0.025, 0.05, 0.1, 0.25, 0.5, 1.0};

  std::unordered_set<int> invalidUsers;
  std::unordered_set<int> invalidItems;
  std::string modelSign = mfModels[0].modelSignature();
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
  
  for (int i = 0; i < mfModels.size(); i++) {
    for (auto&& epsilon: epsilons) {
      auto& fullModel = mfModels[i];
      auto& origModel = origModels[i];
      modelSign = fullModel.modelSignature();
      
      std::cout << "Model: " << i << " " << modelSign << std::endl;
      std::cout << "Train RMSE: " << fullModel.RMSE(data.trainMat, invalidUsers, 
          invalidItems, origModel) << std::endl;
      std::cout << "Test RMSE: " << fullModel.RMSE(data.testMat, invalidUsers, 
          invalidItems, origModel) << std::endl;
      std::cout << "Val RMSE: " << fullModel.RMSE(data.valMat, invalidUsers, 
          invalidItems, origModel) << std::endl;
      //std::cout << "Full RMSE: " << fullModel.fullLowRankErr(data, invalidUsers, 
      //    invalidItems, origModel) << std::endl;

      std::vector<std::pair<double, double>> itemsMeanVar = origModel.itemsMeanVar(data.trainMat);
      std::vector<std::pair<double, double>> usersMeanVar = origModel.usersMeanVar(data.trainMat);

      std::vector<double> itemAccuPreds(nItems, 0); 
      std::vector<double> userAccuPreds(nUsers, 0); 
      std::vector<double> itemSecFreq(nItems, 0);
      std::vector<double> itemSecVar(nItems, 0);
      std::vector<double> itemSecMean(nItems, 0);
      std::vector<double> itemSecAccuPreds(nItems, 0);

      double avgAccuPreds = 0;

      for (int u = 0; u < nUsers; u++) {
        if (invalidUsers.find(u) != invalidUsers.end()) {
          continue;
        }
        std::vector<bool> ratedItems(nItems, false);
        for (int ii = data.trainMat->rowptr[u]; ii < data.trainMat->rowptr[u+1]; 
            ii++) {
          ratedItems[data.trainMat->rowind[ii]] = true;
        }
        double uAccuPred = 0;
#pragma omp parallel for reduction(+:avgAccuPreds, uAccuPred)
        for (int item = 0; item < nItems; item++) {
          if (invalidItems.find(item) != invalidItems.end()) {
            continue;
          }
          if (ratedItems[item]) {
            continue;
          }
          double r_ui = origModel.estRating(u, item);
          double r_ui_est = fullModel.estRating(u, item);
          if (fabs(r_ui - r_ui_est) <= epsilon) {
            itemAccuPreds[item] += 1;
            uAccuPred += 1;
            avgAccuPreds++;
          }
        }

        userAccuPreds[u] += uAccuPred;
      }
      
      avgAccuPreds /= nItems;

#pragma omp parallel for
      for (int item = 0; item < data.trainMat->ncols; item++) {
        int nRatings = data.trainMat->colptr[item+1] - data.trainMat->colptr[item];
        for (int uu = data.trainMat->colptr[item]; 
            uu < data.trainMat->colptr[item+1]; uu++) {
          int user = data.trainMat->colind[uu];
          itemSecFreq[item] += userFreq[user]; 
          itemSecMean[item] += usersMeanVar[user].first;
          itemSecVar[item] += usersMeanVar[user].second;
          itemSecAccuPreds[item] += userAccuPreds[user];
        }
        itemSecFreq[item] = itemSecFreq[item]/nRatings;
        itemSecMean[item] = itemSecMean[item]/nRatings;
        itemSecVar[item] = itemSecVar[item]/nRatings;
        itemSecAccuPreds[item] = itemSecAccuPreds[item]/nRatings;
      }

      std::string opFName = std::string(params.prefix) + "_" + std::to_string(epsilon) 
                              + "_itemFreqAccu.txt";
      std::ofstream opFile(opFName.c_str()); 
      for (int item = 0; item < nItems; item++) {
        opFile << item << "\t" << itemFreq[item] << "\t" << itemSecFreq[item] 
          << "\t" << itemAccuPreds[item] << "\t" << avgAccuPreds << "\t" 
          << itemsMeanVar[item].first << "\t" << itemsMeanVar[item].second 
          << "\t" << itemSecMean[item] << "\t" << itemSecVar[item] << std::endl;
      }

      opFile.close();
    }
  }
}


void analyzeAccuracySingleOrigModel(Data& data, Params& params) {
  
  std::vector<ModelMF> mfModels;
  ModelMF origModel(params, params.origUFacFile, params.origIFacFile,
      params.seed);

  for (int i = 1; i <= 10; i++) {
    std::string prefix = std::string(params.prefix) + "_" + std::to_string(i);
    ModelMF fullModel(params, params.seed);
    fullModel.loadFacs(prefix.c_str());
    mfModels.push_back(fullModel);
  }

  int nUsers = data.trainMat->nrows;
  int nItems = data.trainMat->ncols;
  auto rowColFreq = getRowColFreq(data.trainMat);
  auto userFreq = rowColFreq.first;
  auto itemFreq = rowColFreq.second;
  //double epsilon = 0.025;
  std::vector<double> epsilons = {0.1, 0.25, 0.5, 1.0};

  std::unordered_set<int> invalidUsers;
  std::unordered_set<int> invalidItems;
  std::string modelSign = mfModels[0].modelSignature();
  std::cout << "\nModel sign: " << modelSign << std::endl;    
  
  std::string prefix = std::string(params.prefix) + "_1_" + modelSign + "_invalUsers.txt";
  std::vector<int> invalUsersVec = readVector(prefix.c_str());
  prefix = std::string(params.prefix) + "_1_" + modelSign + "_invalItems.txt";
  std::vector<int> invalItemsVec = readVector(prefix.c_str());
  
  for (auto v: invalUsersVec) {
    invalidUsers.insert(v);
  }
  
  for (auto v: invalItemsVec) {
    invalidItems.insert(v);
  }
  
  for (int i = 0; i < mfModels.size(); i++) {
      
    auto& fullModel = mfModels[i];
    modelSign = fullModel.modelSignature();
    std::cout << "Model: " << i << " " << modelSign << std::endl;
    std::cout << "Train RMSE: " << fullModel.RMSE(data.trainMat, invalidUsers, 
        invalidItems, origModel) << std::endl;
    std::cout << "Test RMSE: " << fullModel.RMSE(data.testMat, invalidUsers, 
        invalidItems, origModel) << std::endl;
    std::cout << "Val RMSE: " << fullModel.RMSE(data.valMat, invalidUsers, 
        invalidItems, origModel) << std::endl;

    for (auto&& epsilon: epsilons) {
      
      //std::cout << "Full RMSE: " << fullModel.fullLowRankErr(data, invalidUsers, 
      //    invalidItems, origModel) << std::endl;

      std::vector<std::pair<double, double>> itemsMeanVar = origModel.itemsMeanVar(data.trainMat);
      std::vector<std::pair<double, double>> usersMeanVar = origModel.usersMeanVar(data.trainMat);

      std::vector<double> itemAccuPreds(nItems, 0); 
      std::vector<double> userAccuPreds(nUsers, 0); 
      std::vector<double> itemSecFreq(nItems, 0);
      std::vector<double> itemSecVar(nItems, 0);
      std::vector<double> itemSecMean(nItems, 0);
      std::vector<double> itemSecAccuPreds(nItems, 0);

      double avgAccuPreds = 0;

      for (int u = 0; u < nUsers; u++) {
        if (invalidUsers.find(u) != invalidUsers.end()) {
          continue;
        }
        std::vector<bool> ratedItems(nItems, false);
        for (int ii = data.trainMat->rowptr[u]; ii < data.trainMat->rowptr[u+1]; 
            ii++) {
          ratedItems[data.trainMat->rowind[ii]] = true;
        }
        double uAccuPred = 0;
#pragma omp parallel for reduction(+:avgAccuPreds, uAccuPred)
        for (int item = 0; item < nItems; item++) {
          if (invalidItems.find(item) != invalidItems.end()) {
            continue;
          }
          if (ratedItems[item]) {
            continue;
          }
          double r_ui = origModel.estRating(u, item);
          double r_ui_est = fullModel.estRating(u, item);
          if (fabs(r_ui - r_ui_est) <= epsilon) {
            itemAccuPreds[item] += 1;
            uAccuPred += 1;
            avgAccuPreds++;
          }
        }

        userAccuPreds[u] += uAccuPred;
      }
      
      avgAccuPreds /= nItems;

#pragma omp parallel for
      for (int item = 0; item < data.trainMat->ncols; item++) {
        int nRatings = data.trainMat->colptr[item+1] - data.trainMat->colptr[item];
        for (int uu = data.trainMat->colptr[item]; 
            uu < data.trainMat->colptr[item+1]; uu++) {
          int user = data.trainMat->colind[uu];
          itemSecFreq[item] += userFreq[user]; 
          itemSecMean[item] += usersMeanVar[user].first;
          itemSecVar[item] += usersMeanVar[user].second;
          itemSecAccuPreds[item] += userAccuPreds[user];
        }
        itemSecFreq[item] = itemSecFreq[item]/nRatings;
        itemSecMean[item] = itemSecMean[item]/nRatings;
        itemSecVar[item] = itemSecVar[item]/nRatings;
        itemSecAccuPreds[item] = itemSecAccuPreds[item]/nRatings;
      }

      std::string opFName = std::string(params.prefix) + "_" + std::to_string(epsilon) 
                              + "_itemFreqAccu.txt";
      std::ofstream opFile(opFName.c_str()); 
      for (int item = 0; item < nItems; item++) {
        opFile << item << "\t" << itemFreq[item] << "\t" << itemSecFreq[item] 
          << "\t" << itemAccuPreds[item] << "\t" << avgAccuPreds << "\t" 
          << itemsMeanVar[item].first << "\t" << itemsMeanVar[item].second 
          << "\t" << itemSecMean[item] << "\t" << itemSecVar[item] << std::endl;
      }

      opFile.close();
    }
  }
}


void sampleUsers(int nUsers, std::unordered_set<int> invalidUsers, 
    std::unordered_set<int>& sampledUsers, int sz, std::mt19937& mt) {

  sampledUsers.clear();
  std::uniform_int_distribution<> dis(0, nUsers-1);
  while (sampledUsers.size() <= sz) {
    int u = dis(mt);
    if (invalidUsers.count(u) > 0) {
      continue;
    }
    sampledUsers.insert(u);
  }
}


void meanAndVarSameGroundSampUsers(Data& data, Params& params) {

  std::cout << "meanAndVarSameGroundSampUsers" << std::endl;

  std::vector<ModelMF> mfModels;

  for (int i = 1; i <= 20; i++) {
    std::string prefix = "mf_0_0_10_" + std::to_string(i);
    ModelMF fullModel(params, params.seed);
    fullModel.loadFacs(prefix.c_str());
    mfModels.push_back(fullModel);
  }

  std::string uFName = "uFac_" + std::to_string(mfModels[0].nUsers) + "_10_0.txt";
  std::string iFName = "iFac_" + std::to_string(mfModels[0].nItems) + "_10_0.txt";
  ModelMF origModel(params, uFName.c_str(), iFName.c_str(), params.seed);

  int nModels    = mfModels.size();
  int nUsers     = data.trainMat->nrows;
  int nItems     = data.trainMat->ncols;
  std::mt19937 mt(params.seed);
  std::unordered_set<int> invalidUsers;
  std::unordered_set<int> invalidItems;
  std::string modelSign = mfModels[0].modelSignature();
  std::cout << "\nModel sign: " << modelSign << std::endl;    
  std::string prefix = std::string(params.prefix) + "_1_" + modelSign + "_invalUsers.txt";
  std::vector<int> invalUsersVec = readVector(prefix.c_str());
  prefix = std::string(params.prefix) + "_1_" + modelSign + "_invalItems.txt";
  std::vector<int> invalItemsVec = readVector(prefix.c_str());
  for (auto v: invalUsersVec) {
    invalidUsers.insert(v);
  }
  for (auto v: invalItemsVec) {
    invalidItems.insert(v);
  }

  std::string opFName = std::string(params.prefix) + "_ui_mean_var.txt";
  std::ofstream opFile(opFName.c_str());

  std::vector<std::unordered_set<int>> itemUsers;
  int sampUserSz = 0.25*nUsers;
  for (int item = 0; item < nItems; item++) {
    //sample 1% users who have not rated the item
    std::unordered_set<int> sampledUsers;
    std::unordered_set<int> ratedUsers;
    for (int uu = data.trainMat->colptr[item]; 
        uu < data.trainMat->colptr[item+1]; uu++) {
      int user = data.trainMat->colind[uu];
      ratedUsers.insert(user);
    }
    for (auto&& user: invalidUsers) {
      ratedUsers.insert(user);
    }
    sampleUsers(nUsers, ratedUsers, sampledUsers, sampUserSz, mt);
    itemUsers.push_back(sampledUsers);
  }
  std::cout << "Sampled " << sampUserSz << " per item." << std::endl; 

#pragma omp parallel for
  for (int item = 0; item < nItems; item++) {
    if (invalidItems.count(item) > 0)  {
      continue;
    }
    
    std::unordered_set<int>& sampledUsers = itemUsers[item];
    std::vector<std::tuple<int, float, float, float>> meanVarErr;
    //compute mean, var and abs err for each user
    for (auto&& user: sampledUsers) {
      float uMean = 0, uVar = 0, uErr = 0, diff = 0;
      for (int i = 0; i < nModels; i++) {
        uMean += mfModels[i].estRating(user, item);
      }
      uMean = uMean/nModels;
      uErr = fabs(uMean - origModel.estRating(user, item));

      for (int i = 0; i < nModels; i++) {
        diff = mfModels[i].estRating(user, item) - uMean;
        uVar += diff*diff;
      }
      uVar = uVar/nModels;
     
      meanVarErr.push_back(std::make_tuple(user, uMean, uVar, uErr));
    } 



#pragma omp critical 
{
    for (int i = 0; i < meanVarErr.size(); i++) {
      
      int user    = std::get<0>(meanVarErr[i]);
      float uMean = std::get<1>(meanVarErr[i]);
      float uVar  = std::get<2>(meanVarErr[i]);
      float uErr  = std::get<3>(meanVarErr[i]);
      
      opFile << user << " " << item << " ";
      opFile << std::fixed << std::setprecision(5) << uMean << " " << uVar << " " 
        << uErr << std::endl;
    }
    
}
  }

  opFile.close();

}


void meanAndVarSameGroundAllUsers(Data& data, Params& params) {
  
  std::cout << "meanAndVarSameGroundAllUsers" << std::endl;
 
  std::vector<ModelMF> mfModels;

  for (int i = 1; i <= 10; i++) {
    std::string prefix = "ccd++_01_seed_" + std::to_string(i);
    ModelMF fullModel(params, params.seed);
    fullModel.loadFacs(prefix.c_str());
    mfModels.push_back(fullModel);
    std::cout << "uFac norm: " << fullModel.uFac.norm() << " iFac norm: " 
      << fullModel.iFac.norm() << std::endl;
  }

  //std::string uFName = "uFac_" + std::to_string(mfModels[0].nUsers) + "_10_1.txt";
  //std::string iFName = "iFac_" + std::to_string(mfModels[0].nItems+1) + "_10_1.txt";
  ModelMF origModel(params, params.origUFacFile, params.origIFacFile, 
      params.seed);

  int nModels    = mfModels.size();
  int nUsers     = data.trainMat->nrows;
  int nItems     = data.trainMat->ncols;
  std::mt19937 mt(params.seed);
  std::unordered_set<int> invalidUsers;
  std::unordered_set<int> invalidItems;
  std::string modelSign = mfModels[0].modelSignature();
  std::cout << "\nModel sign: " << modelSign << std::endl;    
  std::string prefix = "ccd++_01_seed_1_" + modelSign + "_invalUsers.txt";
  std::vector<int> invalUsersVec = readVector(prefix.c_str());
  prefix = "ccd++_01_seed_1_" + modelSign + "_invalItems.txt";
  std::vector<int> invalItemsVec = readVector(prefix.c_str());
  for (auto v: invalUsersVec) {
    invalidUsers.insert(v);
  }
  for (auto v: invalItemsVec) {
    invalidItems.insert(v);
  }
  
  std::cout << "Original RMSE: " << origModel.fullLowRankErr(data, 
      invalidUsers, invalidItems) << std::endl;

  std::vector<double> rmseModels(nModels, 0);
  double avgRMSE = 0;
  double nnz = 0;

  std::string opFName = std::string(params.prefix) + "_ui_mean_var.txt";
  std::ofstream opFile(opFName.c_str());

//#pragma omp parallel for
#pragma omp parallel
{
  std::vector<double> loc_rmseModels(nModels, 0);

  #pragma omp for reduction(+: avgRMSE, nnz)
  for (int item = 0; item < nItems; item++) {
    if (invalidItems.count(item) > 0)  {
      continue;
    }
    
    std::unordered_set<int> ratedUsers;
    for (int uu = data.trainMat->colptr[item]; 
        uu < data.trainMat->colptr[item+1]; uu++) {
      int user = data.trainMat->colind[uu];
      ratedUsers.insert(user);
    }
    
    std::vector<std::tuple<int, float, float, float>> meanVarErr;
    
    //compute mean, var and abs err for each user
    for (int user = 0; user < nUsers; user++) {
      if (invalidUsers.count(user) > 0 || ratedUsers.count(user) > 0) {
        continue;
      }

      float uMean = 0, uVar = 0, uErr = 0, diff = 0;
      float r_ui = origModel.estRating(user, item);

      for (int i = 0; i < nModels; i++) {
        float r_ui_est = mfModels[i].estRating(user, item);
        diff = r_ui_est - r_ui;
        loc_rmseModels[i] += diff*diff;
        uMean += r_ui_est;
      }
      uMean = uMean/nModels;
      
      uErr = fabs(uMean - r_ui);
      avgRMSE += uErr*uErr;

      diff = 0;
      for (int i = 0; i < nModels; i++) {
        diff = mfModels[i].estRating(user, item) - uMean;
        uVar += diff*diff;
      }
      uVar = uVar/nModels;
     
      meanVarErr.push_back(std::make_tuple(user, uMean, uVar, uErr));

      nnz++;
    } 

//#pragma omp critical 
//{
    
    /*
    for (int i = 0; i < meanVarErr.size(); i++) {
      
      int user    = std::get<0>(meanVarErr[i]);
      float uMean = std::get<1>(meanVarErr[i]);
      float uVar  = std::get<2>(meanVarErr[i]);
      float uErr  = std::get<3>(meanVarErr[i]);
      
      opFile << user << " " << item << " ";
      opFile << std::fixed << std::setprecision(5) << uMean << " " << uVar << " " 
        << uErr << std::endl;
    }
    */
    
//}

  }

#pragma omp critical 
{
    for (int i = 0; i < nModels; i++) {
      rmseModels[i] += loc_rmseModels[i];
    }
}

}

  double minRMSE = 100;
  avgRMSE = sqrt(avgRMSE/nnz);
  for (int i = 0; i < nModels; i++) {
    rmseModels[i] = sqrt(rmseModels[i]/nnz);
    if (rmseModels[i] < minRMSE) {
      minRMSE = rmseModels[i];
    }
    std::cout << "Model " << i << " RMSE: " << rmseModels[i] << std::endl;
  }
  std::cout << "Best model RMSE: " << minRMSE << std::endl;
  std::cout << "Average models RMSE: " << avgRMSE << std::endl;

  opFile.close();

}

