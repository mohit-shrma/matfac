#include "topBucketComp.h"

//return a map of item- users
std::map<int, std::vector<int>> pprSampTopNItemsUsers(gk_csr_t *graphMat, 
    int nUsers, int nItems,
    float lambda, int max_niter, std::unordered_set<int>& invalUsers, 
    std::unordered_set<int>& invalItems, int nSampUsers, int seed, int N) {
  
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
    for (int i = 0; i < N; i++) {
      int item = itemScores[i].first;
      auto search = itemUsers.find(item);
      if (search == itemUsers.end()) {
        //item not found
        itemUsers[item] = std::vector<int> ();
      }
      itemUsers[item].push_back(user);
    }

    if (sampUsers.size() % PROGU == 0) {
      std::cout << "Done... " << sampUsers.size() << " :" << lambda << std::endl;
    }
  }

  free(pr);

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
    int nSampUsers, int seed, int N, std::string& prefix) {
  
  std::cout << "\nGetting itemUsers... : " << lambda << std::endl;

  auto itemUsers = pprSampTopNItemsUsers(graphMat, fullModel.nUsers, 
      fullModel.nItems, lambda, max_niter, invalUsers, invalItems, nSampUsers,
      seed, N);

  std::cout << "\nSize itemUsers: " << itemUsers.size() << " : " 
    << lambda << std::endl; 

  std::cout << "\nGetting itemURMSE... : " << lambda << std::endl;
  auto itemURMSE = itemUsersRMSE(origModel, fullModel, itemUsers);
  
  std::vector<int> items;
  for (auto&& kv: itemURMSE) {
    items.push_back(kv.first);
  }

  std::cout << "\nGetting itemARMSE... : " << lambda << std::endl;
  auto itemARMSE = itemAllRMSE(origModel, fullModel, items, invalUsers, invalItems);
  
  std::string fname = prefix + "_topBuckItemRMSE.txt";
  
  std::cout << "\nWriting op... : " << lambda << std::endl;
  std::ofstream opFile(fname);
  if (opFile.is_open()) {
    for (auto&& item: items) {
      //ignore if item in filtItems
      auto search = filtItems.find(item);
      if (search != filtItems.end()) {
        //found in filtered items
        continue;
      }
      opFile << item << " " << itemURMSE[item] << " " << itemARMSE[item];
    }
    opFile.close();
  }

}

