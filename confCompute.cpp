#include "confCompute.h"


double confScore(int user, int item, std::vector<Model>& models) {
  int nModels = models.size();
  std::vector<double> predRats(nModels);
  for (int i = 0; i < nModels; i++) {
    predRats[i] = models[i].estRating(user, item);
  }
  //compute std dev in pred rats
  double std = stddev(predRats);
  double score = -1.0;
  if (0 != std) {
    score = 1.0/std;
  }
  return score;
}


void updateBuckets(int user, std::vector<double>& bucketScores, 
    std::vector<double>& bucketNNZ, 
    std::vector<std::pair<int, double>>& itemScores, Model& origModel,
    Model& fullModel, int nBuckets, int nItemsPerBuck, int nItems) { 
  
    auto comparePair = [](std::pair<int, double> a, std::pair<int, double> b) { 
      return a.second > b.second; 
    };
    
    //sort items by DECREASING order in score
    std::sort(itemScores.begin(), itemScores.end(), comparePair);  
   
    for (int bInd = 0; bInd < nBuckets; bInd++) {
      int start = bInd*nItemsPerBuck;
      int end = (bInd+1)*nItemsPerBuck;
      if (bInd == nBuckets-1 || end > nItems) {
        end = nItems;
      }
      for (int j = start; j < end; j++) {
        int item = itemScores[j].first;
        //compute square err for item
        double r_ui = origModel.estRating(user, item);
        double r_ui_est = fullModel.estRating(user, item);
        double se = (r_ui - r_ui_est)*(r_ui - r_ui_est);
        bucketScores[bInd] += se;
        bucketNNZ[bInd] += 1;
      }

    }
}


std::vector<double> confBucketRMSEs(Model& origModel, Model& fullModel,
    std::vector<Model>& models, int nUsers, int nItems, int nBuckets) {
  
  int nItemsPerBuck = nItems/nBuckets;
  std::vector<double> bucketScores(nBuckets, 0.0);
  std::vector<double> bucketNNZ(nBuckets, 0.0);
  double score;
  std::vector<std::pair<int, double>> itemScores;
  std::cout << "\nconfBucketRMSEs: \n"; 
  for (int user = 0; user < nUsers; user++) {
    itemScores.clear();
    for (int item = 0; item < nItems; item++) {
      //compute confidence score
      score = confScore(user, item, models);
      itemScores.push_back(std::make_pair(item, score));
    }
    
    //add RMSEs to bucket as per ranking by itemscores
    updateBuckets(user, bucketScores, bucketNNZ, itemScores, origModel, fullModel,
        nBuckets, nItemsPerBuck, nItems);
   
    if (0 == user%1000) {
      std::cout << " u: " << user << std::endl;
    }
  
  }

  for (int i = 0; i < nBuckets; i++) {
    bucketScores[i] = sqrt(bucketScores[i]/bucketNNZ[i]);
  }
  return bucketScores;
}


std::vector<double> confBucketRMSEsWInval(Model& origModel, Model& fullModel,
    std::vector<Model>& models, int nUsers, int nItems, int nBuckets,
    std::unordered_set<int>& invalUsers, std::unordered_set<int>& invalItems) {
  
  int nInvalItems = invalItems.size();
  int nItemsPerBuck = (nItems-nInvalItems)/nBuckets;
  
  std::cout << "\nnItemsPerBuck: " << nItemsPerBuck;

  std::vector<double> bucketScores(nBuckets, 0.0);
  std::vector<double> bucketNNZ(nBuckets, 0.0);
  double score;
  std::vector<std::pair<int, double>> itemScores;
  std::cout << "\nconfBucketRMSEs: \n"; 
  for (int user = 0; user < nUsers; user++) {
    //skip if user is invalid
    auto search = invalUsers.find(user);
    if (search != invalUsers.end()) {
      //found n skip
      continue;
    }

    itemScores.clear();
    for (int item = 0; item < nItems; item++) {
      //skip item if invalid
      auto search = invalItems.find(item);
      if (search != invalItems.end()) {
        //found n skip
        continue;
      }
      //compute confidence score
      score = confScore(user, item, models);
      itemScores.push_back(std::make_pair(item, score));
    }
    
    //add RMSEs to bucket as per ranking by itemscores
    updateBuckets(user, bucketScores, bucketNNZ, itemScores, origModel, fullModel,
        nBuckets, nItemsPerBuck, nItems-nInvalItems);
   
    if (0 == user%1000) {
      std::cout << " u: " << user << std::endl;
    }
  
  }

  for (int i = 0; i < nBuckets; i++) {
    bucketScores[i] = sqrt(bucketScores[i]/bucketNNZ[i]);
  }
  return bucketScores;
}


std::vector<double> confOptBucketRMSEs(Model& origModel, Model& fullModel,
    int nUsers, int nItems, int nBuckets) {
  
  int nItemsPerBuck = nItems/nBuckets;
  std::vector<double> bucketScores(nBuckets, 0.0);
  std::vector<double> bucketNNZ(nBuckets, 0.0);
  double score;
  std::vector<std::pair<int, double>> itemScores;
 
  //function to sort pairs in ascending order
  auto comparePair = [](std::pair<int, double> a, std::pair<int, double> b) { 
    return a.second < b.second; 
  };

  std::cout << "\nconfOptBucketRMSEs: \n"; 
  for (int user = 0; user < nUsers; user++) {
    itemScores.clear();
    for (int item = 0; item < nItems; item++) {
      //compute SE score
      double r_ui = origModel.estRating(user, item);
      double r_ui_est = fullModel.estRating(user, item);
      score = (r_ui - r_ui_est)*(r_ui - r_ui_est);
      itemScores.push_back(std::make_pair(item, score));
    }

    //sort scores in ascending order
    std::sort(itemScores.begin(), itemScores.end(), comparePair);

    //update buckets with scores
    for (int bInd = 0; bInd < nBuckets; bInd++) {
      int start = bInd*nItemsPerBuck;
      int end = (bInd+1)*nItemsPerBuck;
      if (bInd == nBuckets-1 || end > nItems) {
        end = nItems;
      }
      for (int j = start; j < end; j++) {
        //add square err for item to bucket
        bucketScores[bInd] += itemScores[j].second;
        bucketNNZ[bInd] += 1;
      }
    }
   
    if (0 == user%1000) {
      std::cout << " u: " << user << std::endl;
    }
  
  }

  for (int i = 0; i < nBuckets; i++) {
    bucketScores[i] = sqrt(bucketScores[i]/bucketNNZ[i]);
  }
  return bucketScores;
}


std::vector<double> confOptBucketRMSEsWInVal(Model& origModel, Model& fullModel,
    int nUsers, int nItems, int nBuckets, std::unordered_set<int>& invalUsers,
    std::unordered_set<int>& invalItems) {
  int nInvalItems = invalItems.size(); 
  int nItemsPerBuck = (nItems- nInvalItems)/nBuckets;
  std::vector<double> bucketScores(nBuckets, 0.0);
  std::vector<double> bucketNNZ(nBuckets, 0.0);
  double score;
  std::vector<std::pair<int, double>> itemScores;
 
  //function to sort pairs in ascending order
  auto comparePair = [](std::pair<int, double> a, std::pair<int, double> b) { 
    return a.second < b.second; 
  };

  std::cout << "\nconfOptBucketRMSEs: \n"; 
  for (int user = 0; user < nUsers; user++) {
    //skip if user is invalid
    auto search = invalUsers.find(user);
    if (search != invalUsers.end()) {
      //found n skip
      continue;
    }
    itemScores.clear();
    for (int item = 0; item < nItems; item++) {
      //skip item if invalid
      auto search = invalItems.find(item);
      if (search != invalItems.end()) {
        //found n skip
        continue;
      }
      //compute SE score
      double r_ui = origModel.estRating(user, item);
      double r_ui_est = fullModel.estRating(user, item);
      score = (r_ui - r_ui_est)*(r_ui - r_ui_est);
      itemScores.push_back(std::make_pair(item, score));
    }

    //sort scores in ascending order
    std::sort(itemScores.begin(), itemScores.end(), comparePair);

    //update buckets with scores
    for (int bInd = 0; bInd < nBuckets; bInd++) {
      int start = bInd*nItemsPerBuck;
      int end = (bInd+1)*nItemsPerBuck;
      if (bInd == nBuckets-1 || end > nItems-nInvalItems) {
        end = nItems-nInvalItems;
      }
      for (int j = start; j < end; j++) {
        //add square err for item to bucket
        bucketScores[bInd] += itemScores[j].second;
        bucketNNZ[bInd] += 1;
      }
    }
   
    if (0 == user%1000) {
      std::cout << " u: " << user << std::endl;
    }
  
  }

  for (int i = 0; i < nBuckets; i++) {
    bucketScores[i] = sqrt(bucketScores[i]/bucketNNZ[i]);
  }
  return bucketScores;
}


//NOTE: graphMat is structure graph of training matrix with nodes: nUsers+nItems
std::vector<double> pprBucketRMSEs(Model& origModel, Model& fullModel, int nUsers, 
    int nItems, float lambda, int max_niter, gk_csr_t *graphMat, int nBuckets) {
  
  int nItemsPerBuck = nItems/nBuckets;
  std::vector<double> bucketScores(nBuckets, 0.0);
  std::vector<double> bucketNNZ(nBuckets, 0.0);
  float *pr = (float*)malloc(sizeof(float)*graphMat->nrows);
  
  std::vector<std::pair<int, double>> itemScores;
  for (int user = 0; user < nUsers; user++) {
    memset(pr, 0, sizeof(float)*graphMat->nrows);
    pr[user] = 1.0;
    
    //run personalized page rank on the graph w.r.t. u
    gk_rw_PageRank(graphMat, lambda, 0.0001, max_niter, pr);

    //get pr score of items
    itemScores.clear();
    for (int i = nUsers; i < nUsers + nItems; i++) {
      itemScores.push_back(std::make_pair(i - nUsers, pr[i]));
    }

    //add RMSEs to bucket as per ranking by itemscores
    updateBuckets(user, bucketScores, bucketNNZ, itemScores, origModel, fullModel,
        nBuckets, nItemsPerBuck, nItems);
    
    if (user % 100 == 0) {
      std::cout<< "\n" << user << " Done..." << std::endl;
    }
  
  }
  
  free(pr);
  
  for (int i = 0; i < nBuckets; i++) {
    bucketScores[i] = sqrt(bucketScores[i]/bucketNNZ[i]);
  }
  return bucketScores;
}


std::vector<double> gprBucketRMSEs(Model& origModel, Model& fullModel, int nUsers, 
    int nItems, float lambda, int max_niter, gk_csr_t *graphMat, int nBuckets) {
  
  int nItemsPerBuck = nItems/nBuckets;
  std::vector<double> bucketScores(nBuckets, 0.0);
  std::vector<double> bucketNNZ(nBuckets, 0.0);
  std::vector<std::pair<int, double>> itemScores;

  float *pr = (float*)malloc(sizeof(float)*graphMat->nrows);
  memset(pr, 0, sizeof(float)*graphMat->nrows);
  //assign all users equal restart probability
  for (int user = 0; user < nUsers; user++) {
    pr[user] = 1.0/nUsers;
  }
  
  //run global page rank on the graph w.r.t. users
  gk_rw_PageRank(graphMat, lambda, 0.0001, max_niter, pr);

  //get pr score of items
  for (int i = nUsers; i < nUsers + nItems; i++) {
    itemScores.push_back(std::make_pair(i - nUsers, pr[i]));
  }

  //sort items by global page rank
  auto comparePair = [](std::pair<int, double> a, std::pair<int, double> b) { 
    return a.second > b.second; 
  };
  
  //sort items by DECREASING order in score
  std::sort(itemScores.begin(), itemScores.end(), comparePair);  

  //add RMSEs to bucket as per ranking by itemscores
  for (int user = 0; user < nUsers; user++) {
    for (int bInd = 0; bInd < nBuckets; bInd++) {
      int start = bInd*nItemsPerBuck;
      int end = (bInd+1)*nItemsPerBuck;
      if (bInd == nBuckets-1 || end > nItems) {
        end = nItems;
      }
      for (int j = start; j < end; j++) {
        int item = itemScores[j].first;
        //compute square err for item
        double r_ui = origModel.estRating(user, item);
        double r_ui_est = fullModel.estRating(user, item);
        double se = (r_ui - r_ui_est)*(r_ui - r_ui_est);
        bucketScores[bInd] += se;
        bucketNNZ[bInd] += 1;
      }
    }
  }

  for (int i = 0; i < nBuckets; i++) {
    bucketScores[i] = sqrt(bucketScores[i]/bucketNNZ[i]);
  }
    
  free(pr);
  
  return bucketScores;
}


std::vector<double> gprBucketRMSEsWInVal(Model& origModel, Model& fullModel, int nUsers, 
    int nItems, float lambda, int max_niter, gk_csr_t *graphMat, int nBuckets,
    std::unordered_set<int>& invalUsers, std::unordered_set<int>& invalItems) {
  
  int nInvalItems = invalItems.size();
  int nItemsPerBuck = (nItems-nInvalItems)/nBuckets;
  std::vector<double> bucketScores(nBuckets, 0.0);
  std::vector<double> bucketNNZ(nBuckets, 0.0);
  std::vector<std::pair<int, double>> itemScores;

  float *pr = (float*)malloc(sizeof(float)*graphMat->nrows);
  memset(pr, 0, sizeof(float)*graphMat->nrows);
  //assign all users equal restart probability
  for (int user = 0; user < nUsers; user++) {
    pr[user] = 1.0/nUsers;
  }
  
  //run global page rank on the graph w.r.t. users
  gk_rw_PageRank(graphMat, lambda, 0.0001, max_niter, pr);

  //get pr score of items
  for (int i = nUsers; i < nUsers + nItems; i++) {
    int item = i - nUsers;
    //skip item if invalid
    auto search = invalItems.find(item);
    if (search != invalItems.end()) {
      //found n skip
      continue;
    }
    itemScores.push_back(std::make_pair(item, pr[i]));
  }

  //sort items by global page rank
  auto comparePair = [](std::pair<int, double> a, std::pair<int, double> b) { 
    return a.second > b.second; 
  };
  
  //sort items by DECREASING order in score
  std::sort(itemScores.begin(), itemScores.end(), comparePair);  

  //add RMSEs to bucket as per ranking by itemscores
  for (int user = 0; user < nUsers; user++) {
    //skip if user is invalid
    auto search = invalUsers.find(user);
    if (search != invalUsers.end()) {
      //found n skip
      continue;
    }
    for (int bInd = 0; bInd < nBuckets; bInd++) {
      int start = bInd*nItemsPerBuck;
      int end = (bInd+1)*nItemsPerBuck;
      if (bInd == nBuckets-1 || end > nItems-nInvalItems) {
        end = nItems-nInvalItems;
      }
      for (int j = start; j < end; j++) {
        int item = itemScores[j].first;
        //compute square err for item
        double r_ui = origModel.estRating(user, item);
        double r_ui_est = fullModel.estRating(user, item);
        double se = (r_ui - r_ui_est)*(r_ui - r_ui_est);
        bucketScores[bInd] += se;
        bucketNNZ[bInd] += 1;
      }
    }
  }

  for (int i = 0; i < nBuckets; i++) {
    bucketScores[i] = sqrt(bucketScores[i]/bucketNNZ[i]);
  }
    
  free(pr);
  
  return bucketScores;
}


std::vector<double> pprBucketRMSEsFrmPR(Model& origModel, Model& fullModel, int nUsers, 
    int nItems, gk_csr_t *graphMat, int nBuckets, const char* prFName) {
  
  int nItemsPerBuck = nItems/nBuckets;
  std::vector<double> bucketScores(nBuckets, 0.0);
  std::vector<double> bucketNNZ(nBuckets, 0.0);
  
  std::vector<std::pair<int, double>> itemScores;
  std::ifstream inFile (prFName);
  std::string line, token;
  std::string delimiter = " ";
  size_t pos;
  int item;
  double score;

  if (inFile.is_open()) {
    std::cout << "\npprBucketRMSEsFrmPR: \n";
    for (int user = 0; user < nUsers; user++) {
     
      getline(inFile, line);
      
      itemScores.clear();

      //split the line
      for (int i = 0; i < nItems; i++) {
        pos = line.find(delimiter);
        token = line.substr(0, pos);
        item = std::stoi(token)-nUsers; 
        line.erase(0, pos + delimiter.length());
        pos = line.find(delimiter);
        token = line.substr(0, pos);
        score = std::stod(token);
        line.erase(0, pos + delimiter.length());
      
        itemScores.push_back(std::make_pair(item, score));
      }


      //add RMSEs to bucket as per ranking by itemscores
      updateBuckets(user, bucketScores, bucketNNZ, itemScores, origModel, fullModel,
          nBuckets, nItemsPerBuck, nItems);
      
      if (user % 1000 == 0) {
        std::cout<< " u: " << user << std::endl;
      }
    
    }
   
    inFile.close();
  } else {
    std::cerr << "\nFailed to open file: " << prFName << std::endl;
  }

  
  for (int i = 0; i < nBuckets; i++) {
    bucketScores[i] = sqrt(bucketScores[i]/bucketNNZ[i]);
  }
  return bucketScores;
}

