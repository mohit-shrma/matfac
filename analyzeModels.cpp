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

  for (int i = 1; i <= 3; i++) {
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

  for (int i = 1; i <= 3; i++) {
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
    std::string prefix = "als_10_" + std::to_string(i);
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
  std::string prefix = std::string(params.prefix) + "_10_1_" + modelSign + "_invalUsers.txt";
  std::vector<int> invalUsersVec = readVector(prefix.c_str());
  prefix = std::string(params.prefix) + "_10_1_" + modelSign + "_invalItems.txt";
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

