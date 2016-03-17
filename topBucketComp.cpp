#include "topBucketComp.h"


void updateBucketsArr(std::vector<double>& bucketScores,
    std::vector<double>& bucketNNZ, std::vector<double>& arr, int nBuckets) {
    int nItems = arr.size(); 
    int nItemsPerBuck = nItems/nBuckets;
    for (int bInd = 0; bInd < nBuckets; bInd++) {
      int start = bInd*nItemsPerBuck;
      int end = (bInd+1)*nItemsPerBuck;
      if (bInd == nBuckets-1 || end > nItems) {
        end = nItems;
      }
      for (int j = start; j < end; j++) {
        bucketScores[bInd] += arr[j];
        bucketNNZ[bInd] += 1;
      }
    }
}


void itemRMSEsProb(int user, gk_csr_t *graphMat, float lambda, int nUsers,
    int nItems, std::unordered_set<int>& invalItems,
    Model& fullModel, Model& origModel,
    std::vector<double>& itemsRMSE, std::vector<double>& itemsProb) {
  
  std::vector<std::pair<int, double>> itemScores;
  float *pr = (float*) malloc(sizeof(float)*graphMat->nrows);
  memset(pr, 0, sizeof(float)*graphMat->nrows);
  pr[user] = 1.0;
  
  //run personalized page rank on the graph w.r.t. u
  gk_rw_PageRank(graphMat, lambda, 0.0001, MAX_PR_ITER, pr);
  
  //get pr scores of items
  itemScores.clear();
  for (int i = nUsers; i < nUsers + nItems; i++) {
    int item = i - nUsers;
    //skip if item is invalid
    auto search = invalItems.find(item);
    if (search != invalItems.end()) {
      //found, invalid, skip
      continue;
    }
    itemScores.push_back(std::make_pair(item, pr[i]));
  }
  
  //sort items in decreasing order of score
  std::sort(itemScores.begin(), itemScores.end(), descComp);

  itemsRMSE.clear();
  itemsProb.clear();
  for (auto&& itemScore: itemScores) {
    int item = itemScore.first;
    double score = itemScore.second;
    double r_ui = origModel.estRating(user, item);
    double r_ui_est = fullModel.estRating(user, item);
    double se = (r_ui - r_ui_est)*(r_ui - r_ui_est);
    itemsRMSE.push_back(se);
    itemsProb.push_back(score);
  }

  free(pr);
}


void pprUsersRMSEProb(gk_csr_t *graphMat, 
    int nUsers, int nItems, Model& origModel, Model& fullModel,
    float lambda, std::unordered_set<int>& invalUsers, 
    std::unordered_set<int>& invalItems, 
    std::vector<int> users, std::string& prefix) {
  
  int nBuckets = 10;
  int nItemsPerBuck = nItems/nBuckets;

  std::vector<double> uRMSEScores(nBuckets, 0);
  std::vector<double> uProbs(nBuckets, 0);
  std::vector<double> uBucketNNZ(nBuckets, 0);

  float *pr = (float*)malloc(sizeof(float)*graphMat->nrows);
  
  std::vector<std::pair<int, double>> itemScores;
  std::vector<double> itemsRMSE;
  std::vector<double> itemsProb;

  std::string rmseFName = prefix + "_uRMSE.txt" ;
  std::ofstream rmseOpFile(rmseFName.c_str()); 

  std::string probFName = prefix + "_uProbs.txt";
  std::ofstream probOpFile(probFName.c_str());

  for (auto&& user: users) {
    //skip if user is invalid
    auto search = invalUsers.find(user);
    if (search != invalUsers.end()) {
      //found n skip
      continue;
    }
 
    itemRMSEsProb(user, graphMat, lambda, nUsers, nItems, invalItems, 
        fullModel, origModel, itemsRMSE, itemsProb);
    
    //reset user specific vec to 0
    std::fill(uBucketNNZ.begin(), uBucketNNZ.end(), 0); 
    std::fill(uRMSEScores.begin(), uRMSEScores.end(), 0);
    updateBucketsArr(uRMSEScores, uBucketNNZ, itemsRMSE, nBuckets);
    
    //reset user specific vec to 0
    std::fill(uBucketNNZ.begin(), uBucketNNZ.end(), 0); 
    std::fill(uProbs.begin(), uProbs.end(), 0);
    updateBucketsArr(uProbs, uBucketNNZ, itemsProb, nBuckets);

    //write and update aggregated buckets
    rmseOpFile << user << " ";
    probOpFile << user << " ";
    for (int i = 0; i < nBuckets; i++) {
      rmseOpFile << sqrt(uRMSEScores[i]/uBucketNNZ[i]) << " ";
      probOpFile << uProbs[i]/uBucketNNZ[i] << " ";
    }
    rmseOpFile << std::endl;
    probOpFile << std::endl;
  }

  free(pr);

  rmseOpFile.close();
  probOpFile.close();
}


void pprSampUsersRMSEProb(gk_csr_t *graphMat, 
    int nUsers, int nItems, Model& origModel, Model& fullModel,
    float lambda, int max_niter, std::unordered_set<int>& invalUsers, 
    std::unordered_set<int>& invalItems, std::unordered_set<int>& filtItems, 
    int nSampUsers, int seed, std::string& prefix) {
  
  int nBuckets = 10;
  int nItemsPerBuck = nItems/nBuckets;

  std::vector<double> rmseScores(nBuckets, 0);
  std::vector<double> uRMSEScores(nBuckets, 0);
  
  std::vector<double> probs(nBuckets, 0);
  std::vector<double> uProbs(nBuckets, 0);
  
  std::vector<double> bucketNNZ(nBuckets, 0);
  std::vector<double> uBucketNNZ(nBuckets, 0);

  float *pr = (float*)malloc(sizeof(float)*graphMat->nrows);
  
  std::vector<std::pair<int, double>> itemScores;
  std::vector<double> itemsRMSE;
  std::vector<double> itemsProb;

  //initialize random engine
  std::mt19937 mt(seed);
  //user distribution to sample users
  std::uniform_int_distribution<int> uDist(0, nUsers-1);

  std::unordered_set<int> sampUsers;

  std::string rmseFName = prefix + "_uRMSE.txt" ;
  std::ofstream rmseOpFile(rmseFName.c_str()); 

  std::string probFName = prefix + "_uProbs.txt";
  std::ofstream probOpFile(probFName.c_str());

  while (sampUsers.size() < nSampUsers) {
    //sample user
    int user = uDist(mt);

    auto search = sampUsers.find(user);
    if (search != sampUsers.end()) {
      //already sampled user
      continue;
    }
  
    //skip if user is invalid
    search = invalUsers.find(user);
    if (search != invalUsers.end()) {
      //found n skip
      continue;
    }
 
    //insert the sampled user
    sampUsers.insert(user);

    itemRMSEsProb(user, graphMat, lambda, nUsers, nItems, invalItems, 
        fullModel, origModel, itemsRMSE, itemsProb);
    
    //reset user specific vec to 0
    std::fill(uBucketNNZ.begin(), uBucketNNZ.end(), 0); 
    std::fill(uRMSEScores.begin(), uRMSEScores.end(), 0);
    updateBucketsArr(uRMSEScores, uBucketNNZ, itemsRMSE, nBuckets);
    
    //reset user specific vec to 0
    std::fill(uBucketNNZ.begin(), uBucketNNZ.end(), 0); 
    std::fill(uProbs.begin(), uProbs.end(), 0);
    updateBucketsArr(uProbs, uBucketNNZ, itemsProb, nBuckets);

    //write and update aggregated buckets
    rmseOpFile << user << " ";
    probOpFile << user << " ";
    for (int i = 0; i < nBuckets; i++) {
      bucketNNZ[i]  += uBucketNNZ[i];
      rmseScores[i] += uRMSEScores[i];
      rmseOpFile << sqrt(uRMSEScores[i]/uBucketNNZ[i]) << " ";
      probs[i]      += uProbs[i];
      probOpFile << uProbs[i]/uBucketNNZ[i] << " ";
    }
    rmseOpFile << std::endl;
    probOpFile << std::endl;
    
    if (sampUsers.size() % PROGU == 0) {
      std::cout << "Done... " << sampUsers.size() << " :" << lambda << std::endl;
    }

  }

  free(pr);

  rmseOpFile.close();
  probOpFile.close();
}


//return a map of item- users
std::map<int, std::vector<int>> pprSampTopNItemsUsers(gk_csr_t *graphMat, 
    int nUsers, int nItems, Model& origModel, Model& fullModel,
    float lambda, int max_niter, std::unordered_set<int>& invalUsers, 
    std::unordered_set<int>& invalItems, std::unordered_set<int>& filtItems, 
    int nSampUsers, int seed, int N, std::string& prefix) {
  
  int nBuckets = 10;
  int nItemsPerBuck = N/nBuckets;

  std::vector<double> bucketScores(nBuckets, 0);
  std::vector<double> bucketNNZ(nBuckets, 0);

  float *pr = (float*)malloc(sizeof(float)*graphMat->nrows);
  
  std::vector<std::pair<int, double>> itemScores;
  std::map<int, std::vector<int>> itemUsers;

  //initialize random engine
  std::mt19937 mt(seed);
  //user distribution to sample users
  std::uniform_int_distribution<int> uDist(0, nUsers-1);

  std::unordered_set<int> sampUsers;

  while (sampUsers.size() < nSampUsers) {
    //sample user
    int user = uDist(mt);

    auto search = sampUsers.find(user);
    if (search != sampUsers.end()) {
      //already sampled user
      continue;
    }
  
    //skip if user is invalid
    search = invalUsers.find(user);
    if (search != invalUsers.end()) {
      //found n skip
      continue;
    }
 
    //insert the sampled user
    sampUsers.insert(user);
    
    memset(pr, 0, sizeof(float)*graphMat->nrows);
    pr[user] = 1.0;

    //run personalized page rank on the graph w.r.t. u
    gk_rw_PageRank(graphMat, lambda, 0.0001, max_niter, pr);

    //get pr score of items
    itemScores.clear();
    for (int i = nUsers; i < nUsers + nItems; i++) {
      int item = i - nUsers;
      //skip if item is invalid
      search = invalItems.find(item);
      if (search != invalItems.end()) {
        //found, invalid, skip
        continue;
      }
      itemScores.push_back(std::make_pair(item, pr[i]));
    }

    //sort items in decreasing order of score
    std::sort(itemScores.begin(), itemScores.end(), descComp);
    std::vector<int> sortedItems;
    int i = 0;
    while (sortedItems.size() < N) {
      int item = itemScores[i++].first;

      auto search = filtItems.find(item);
      if (search != filtItems.end()) {
        //found in filtered items, skip
        continue;
      }

      auto search2 = itemUsers.find(item);
      if (search2 == itemUsers.end()) {
        //item not found
        itemUsers[item] = std::vector<int> ();
      }
      itemUsers[item].push_back(user);

      sortedItems.push_back(item);
    }
    
    nItemsPerBuck = sortedItems.size() / nBuckets;
    updateBucketsSorted(user, bucketScores, bucketNNZ, sortedItems, origModel,
        fullModel, nBuckets, nItemsPerBuck);

    if (sampUsers.size() % PROGU == 0) {
      std::cout << "Done... " << sampUsers.size() << " :" << lambda << std::endl;
    }

  }

  for (size_t i = 0; i < bucketScores.size(); i++) {
    bucketScores[i] = sqrt(bucketScores[i]/bucketNNZ[i]);
  }

  free(pr);

  //write out bucket scores
  std::string scoreFName = prefix + "_" + std::to_string(lambda) + "_ppr_bucket.txt";
  writeVector(bucketScores, scoreFName.c_str());

  //write out bucket nnz
  std::string nnzFName = prefix + "_" + std::to_string(lambda) + "_nnz_bucket.txt";
  writeVector(bucketNNZ, nnzFName.c_str());

  return itemUsers;
}


std::map<int, double> itemUsersRMSE(Model& origModel, Model& fullModel,
    std::map<int, std::vector<int>>& itemUsers) {
  
  std::map<int, double> itemUsersRMSE;
  for (auto&& kv: itemUsers) {
    auto item  = kv.first;
    auto users = kv.second;
    double se = 0, diff = 0;
    for (auto u: users) {
       diff = origModel.estRating(u, item) - fullModel.estRating(u, item);
       se += diff*diff;
    }
    itemUsersRMSE[item] = sqrt(se/users.size());
  }
  
  return itemUsersRMSE;
}


std::map<int, double> itemAllRMSE(Model& origModel, Model& fullModel, 
    std::vector<int>& items,
    std::unordered_set<int>& invalidUsers, 
    std::unordered_set<int>& invalidItems) {
  
  std::map<int, double> itemRMSE;
  for (auto item: items) {
    //ignore if invalid item
    auto search = invalidItems.find(item);
    if (search != invalidItems.end()) {
      //found n skip
      continue;
    }

    double se = 0, diff =0;
    int uCount = 0;
    for (int u = 0; u < origModel.nUsers; u++) {
      //ignore if invalid user
      search = invalidUsers.find(u);
      if (search != invalidUsers.end()) {
        //found n skip
        continue;
      }
      diff = fullModel.estRating(u, item) - origModel.estRating(u, item);
      se += diff*diff;
      uCount++;
    }
    itemRMSE[item] = sqrt(se/uCount);
  }

  return itemRMSE;
}


void writeTopBuckRMSEs(Model& origModel, Model& fullModel, gk_csr_t* graphMat,
    float lambda, int max_niter, std::unordered_set<int>& invalUsers, 
    std::unordered_set<int>& invalItems, std::unordered_set<int>& filtItems,
    int nSampUsers, int seed, int N, std::string prefix) {
  
  std::cout << "\nGetting itemUsers... : " << lambda << std::endl;

  auto itemUsers = pprSampTopNItemsUsers(graphMat, fullModel.nUsers, 
      fullModel.nItems, origModel, fullModel, lambda, max_niter, invalUsers, 
      invalItems, filtItems, nSampUsers, seed, N, prefix);

  std::cout << "\nBefore filter itemUsers size: " << itemUsers.size() << " : " 
    << lambda << std::endl; 

  //remove filtItems from itemUsers map
  for (auto&& item: filtItems) {
    itemUsers.erase(item);
  }

  std::cout << "\nAfter filter itemUsers size: " << itemUsers.size() << " : " 
    << lambda << std::endl; 

  std::cout << "\nGetting itemURMSE... : " << lambda << std::endl;
  auto itemURMSE = itemUsersRMSE(origModel, fullModel, itemUsers);
  
  std::vector<int> items;
  for (auto&& kv: itemURMSE) {
    items.push_back(kv.first);
  }

  std::cout << "\nGetting itemARMSE... : " << lambda << std::endl;
  auto itemARMSE = itemAllRMSE(origModel, fullModel, items, invalUsers, invalItems);
  
  std::string fname = prefix + "_pprtop_" + std::to_string(N) + "_freq_" 
    + std::to_string(filtItems.size()) + "_ItemRMSE.txt";
  
  std::cout << "\nWriting op... : " << lambda << " " << fname << std::endl;
  std::ofstream opFile(fname);
  if (opFile.is_open()) {
    for (auto&& item: items) {
      opFile << item << " " << itemUsers[item].size() << " "
        << itemURMSE[item] << " " << itemARMSE[item] << std::endl;
    }
    opFile.close();
  }

}


