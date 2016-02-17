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


std::vector<double> genConfidenceCurve(
    std::vector<std::tuple<int, int, double>> matConfScores, Model& origModel,
    Model& fullModel, int nBuckets, float alpha) {
  
  std::vector<double> binWidths;
  int nScores = matConfScores.size();
  std::cout << "\nnScores: " << nScores << std::endl;
  int nItemsPerBuck = nScores/nBuckets;
  auto compareTriplets = [] (std::tuple<int, int, double> a, 
                              std::tuple<int, int, double> b) {
    return std::get<2>(a) > std::get<2>(b);
  };

  //sort in descending order of confidence
  std::sort(matConfScores.begin(), matConfScores.end(), compareTriplets);
  
  std::vector<double> widths;
  for (int bInd = 0; bInd < nBuckets; bInd++) {
    int start = bInd*nItemsPerBuck;
    int end = (bInd+1)*nItemsPerBuck;
    if (bInd == nBuckets-1 || end > nScores) {
      end = nScores;
    }

    //find half-width of the confidence-interval for bin
    //S.T. (1-alpha)fraction of predicted ratings are with in +- w of actual
    //ratings
    widths.clear();
    for (int j = start; j < end; j++) {
      int user = std::get<0>(matConfScores[j]);
      int item = std::get<1>(matConfScores[j]);
      //compute square err for item
      double r_ui = origModel.estRating(user, item);
      double r_ui_est = fullModel.estRating(user, item);
      double w = fabs(r_ui - r_ui_est);
      widths.push_back(w);
    }
    //sort widths in ascending order
    std::sort(widths.begin(), widths.end());
    binWidths.push_back(widths[(1-alpha)*widths.size()]);
  }


  return binWidths;
}


std::vector<double> computeModConf(gk_csr_t* mat, 
    std::vector<Model>& models, std::unordered_set<int>& invalUsers,
    std::unordered_set<int>& invalItems, Model& origModel,
    Model& fullModel, int nBuckets, float alpha) {
  
  std::vector<std::tuple<int, int, double>> matConfScores;
  double score;

  for (int u = 0 ; u < mat->nrows; u++) {
    
    //ignore if invalid user
    //skip if user is invalid
    auto search = invalUsers.find(u);
    if (search != invalUsers.end()) {
      //found n skip
      continue;
    }

    for (int ii = mat->rowptr[u]; ii < mat->rowptr[u+1]; ii++) {
      int item = mat->rowind[ii];
      //ignore if invalid item
      search = invalItems.find(item);
      if (search != invalItems.end()) {
        //found n skip
        continue;
      }
      score = confScore(u, item, models);
      matConfScores.push_back(std::make_tuple(u, item, score));
    }

  }
  return genConfidenceCurve(matConfScores, origModel, fullModel, nBuckets, alpha);
}


//compute confidence on pairs which are not in training
std::vector<double> computeMissingModConf(gk_csr_t* trainMat, 
    std::vector<Model>& models, std::unordered_set<int>& invalUsers,
    std::unordered_set<int>& invalItems, Model& origModel,
    Model& fullModel, int nBuckets, float alpha) {
  
  std::vector<std::tuple<int, int, double>> matConfScores;
  double score;
  std::unordered_set<int> trainItemSet;

  for (int u = 0 ; u < trainMat->nrows; u++) {
    
    //ignore if invalid user
    //skip if user is invalid
    auto search = invalUsers.find(u);
    if (search != invalUsers.end()) {
      //found n skip
      continue;
    }
    
    trainItemSet.clear();
    for (int ii = trainMat->rowptr[u]; ii < trainMat->rowptr[u+1]; ii++) {
      int item = trainMat->rowind[ii];
      trainItemSet.insert(item);
    }

    for (int item = 0; item < trainMat->ncols; item++) {
      
      //ignore if invalid item
      search = invalItems.find(item);
      if (search != invalItems.end()) {
        //found n skip
        continue;
      }

      //ignore if item present in users' training set
      search = trainItemSet.find(item);
      if (search != trainItemSet.end()) {
        //found n skip
        continue;
      }

      score = confScore(u, item, models);
      matConfScores.push_back(std::make_tuple(u, item, score));
    }

  }
  return genConfidenceCurve(matConfScores, origModel, fullModel, nBuckets, alpha);
}


std::vector<double> computeMissingModConfSamp(gk_csr_t* trainMat, 
    std::vector<Model>& models, std::unordered_set<int>& invalUsers,
    std::unordered_set<int>& invalItems, Model& origModel,
    Model& fullModel, int nBuckets, float alpha, int seed) {
  
  std::vector<std::tuple<int, int, double>> matConfScores;
  double score;
  int nUsers = trainMat->nrows;
  int nItems = trainMat->ncols;
  std::vector<std::unordered_set<int>> uTrItemSet(nUsers);
  
  int trNNz = 0;
  for (int u = 0 ; u < trainMat->nrows; u++) {
    for (int ii = trainMat->rowptr[u]; ii < trainMat->rowptr[u+1]; ii++) {
      int item = trainMat->rowind[ii];
      uTrItemSet[u].insert(item);
      trNNz++;
    }
  } 

  double halfRatCount = ((double)nUsers*(double)nItems)/2.0;
  int nScores = std::min((double)MAX_MISS_RATS, halfRatCount);
  std::cout << "\nnScores: " << nScores << " halfRatCount: " << halfRatCount 
    << " nScores: " << nScores  << std::endl;

  //random engine
  std::mt19937 mt(seed);
  //user dist
  std::uniform_int_distribution<int> uDist(0, nUsers-1);
  std::uniform_int_distribution<int> iDist(0, nItems-1);

  while (matConfScores.size() < nScores) {
    //sample u
    int user = uDist(mt);
    //skip if user is invalid
    auto search = invalUsers.find(user);
    if (search != invalUsers.end()) {
      //found n skip
      continue;
    }

    //sample item
    int item = iDist(mt);
    //ignore if invalid item
    search = invalItems.find(item);
    if (search != invalItems.end()) {
      //found n skip
      continue;
    }

    //ignore if item in training set
    search = uTrItemSet[user].find(item);
    if (search != uTrItemSet[user].end()) {
      //found n skip
      continue;
    }

    score = confScore(user, item, models);
    matConfScores.push_back(std::make_tuple(user, item, score));
  }

  return genConfidenceCurve(matConfScores, origModel, fullModel, nBuckets, alpha);
}


std::vector<double> computeMissingGPRConf(gk_csr_t* trainMat, 
    gk_csr_t* graphMat, std::unordered_set<int>& invalUsers,
    std::unordered_set<int>& invalItems, float lambda, int max_niter, Model& origModel,
    Model& fullModel, int nBuckets, float alpha) {
  
  std::vector<std::tuple<int, int, double>> matConfScores;
  double score;
  std::unordered_set<int> trainItemSet;
  int nUsers = trainMat->nrows;

  float *pr = (float*)malloc(sizeof(float)*graphMat->nrows);
  memset(pr, 0, sizeof(float)*graphMat->nrows);
  //assign all users equal restart probability
  for (int user = 0; user < nUsers; user++) {
    pr[user] = 1.0/nUsers;
  }
  
  //run global page rank on the graph w.r.t. users
  gk_rw_PageRank(graphMat, lambda, 0.0001, max_niter, pr);
  
  for (int u = 0 ; u < trainMat->nrows; u++) {
    
    //ignore if invalid user
    //skip if user is invalid
    auto search = invalUsers.find(u);
    if (search != invalUsers.end()) {
      //found n skip
      continue;
    }
      
    trainItemSet.clear();
    for (int ii = trainMat->rowptr[u]; ii < trainMat->rowptr[u+1]; ii++) {
      int item = trainMat->rowind[ii];
      trainItemSet.insert(item);
    }

    for (int item = 0; item < trainMat->ncols; item++) {
      //ignore if invalid item
      search = invalItems.find(item);
      if (search != invalItems.end()) {
        //found n skip
        continue;
      }
      
      //ignore if item present in users' training set
      search = trainItemSet.find(item);
      if (search != trainItemSet.end()) {
        //found n skip
        continue;
      }

      score = pr[nUsers + item];
      matConfScores.push_back(std::make_tuple(u, item, score));
    }

  }


  free(pr);
  return genConfidenceCurve(matConfScores, origModel, fullModel, nBuckets, alpha);
}


std::vector<double> computeMissingGPRConfSamp(gk_csr_t* trainMat, 
    gk_csr_t* graphMat, std::unordered_set<int>& invalUsers,
    std::unordered_set<int>& invalItems, float lambda, int max_niter, Model& origModel,
    Model& fullModel, int nBuckets, float alpha, int seed) {
  
  std::vector<std::tuple<int, int, double>> matConfScores;
  double score;
  int nUsers = trainMat->nrows;
  int nItems = trainMat->ncols;
  std::vector<std::unordered_set<int>> uTrItemSet(nUsers);

  int trNNz = 0;
  for (int u = 0 ; u < trainMat->nrows; u++) {
    for (int ii = trainMat->rowptr[u]; ii < trainMat->rowptr[u+1]; ii++) {
      int item = trainMat->rowind[ii];
      uTrItemSet[u].insert(item);
      trNNz++;
    }
  }

  double halfRatCount = ((double)nUsers*(double)nItems)/2.0;
  int nScores = std::min((double)MAX_MISS_RATS, halfRatCount);
  std::cout << "\nnScores: " << nScores << std::endl;
 
  float *pr = (float*)malloc(sizeof(float)*graphMat->nrows);
  memset(pr, 0, sizeof(float)*graphMat->nrows);
  //assign all users equal restart probability
  for (int user = 0; user < nUsers; user++) {
    pr[user] = 1.0/nUsers;
  }
  
  //run global page rank on the graph w.r.t. users
  gk_rw_PageRank(graphMat, lambda, 0.0001, max_niter, pr);
 
   //random engine
  std::mt19937 mt(seed);
  //user dist
  std::uniform_int_distribution<int> uDist(0, nUsers-1);
  std::uniform_int_distribution<int> iDist(0, nItems-1);
  
  while (matConfScores.size() < nScores) {
    //sample u
    int user = uDist(mt);
    //skip if user is invalid
    auto search = invalUsers.find(user);
    if (search != invalUsers.end()) {
      //found n skip
      continue;
    }

    //sample item
    int item = iDist(mt);
    //ignore if invalid item
    search = invalItems.find(item);
    if (search != invalItems.end()) {
      //found n skip
      continue;
    }

    //ignore if item in training set
    search = uTrItemSet[user].find(item);
    if (search != uTrItemSet[user].end()) {
      //found n skip
      continue;
    }

    score = pr[nUsers + item];
    matConfScores.push_back(std::make_tuple(user, item, score));
  }


  free(pr);
  return genConfidenceCurve(matConfScores, origModel, fullModel, nBuckets, alpha);
}


std::vector<double> computeGPRConf(gk_csr_t* mat, 
    gk_csr_t* graphMat, std::unordered_set<int>& invalUsers,
    std::unordered_set<int>& invalItems, float lambda, int max_niter, Model& origModel,
    Model& fullModel, int nBuckets, float alpha) {
  
  std::vector<std::tuple<int, int, double>> matConfScores;
  double score;
  int nUsers = mat->nrows;

  float *pr = (float*)malloc(sizeof(float)*graphMat->nrows);
  memset(pr, 0, sizeof(float)*graphMat->nrows);
  //assign all users equal restart probability
  for (int user = 0; user < nUsers; user++) {
    pr[user] = 1.0/nUsers;
  }
  
  //run global page rank on the graph w.r.t. users
  gk_rw_PageRank(graphMat, lambda, 0.0001, max_niter, pr);
  
  for (int u = 0 ; u < mat->nrows; u++) {
    
    //ignore if invalid user
    //skip if user is invalid
    auto search = invalUsers.find(u);
    if (search != invalUsers.end()) {
      //found n skip
      continue;
    }

    for (int ii = mat->rowptr[u]; ii < mat->rowptr[u+1]; ii++) {
      int item = mat->rowind[ii];
      //ignore if invalid item
      search = invalItems.find(item);
      if (search != invalItems.end()) {
        //found n skip
        continue;
      }
      score = pr[nUsers + item];
      matConfScores.push_back(std::make_tuple(u, item, score));
    }

  }

  free(pr);
  return genConfidenceCurve(matConfScores, origModel, fullModel, nBuckets, alpha);
}


std::vector<double> computePPRConf(gk_csr_t* mat, 
    gk_csr_t* graphMat, std::unordered_set<int>& invalUsers,
    std::unordered_set<int>& invalItems, float lambda, int max_niter, Model& origModel,
    Model& fullModel, int nBuckets, float alpha) {
  
  std::vector<std::tuple<int, int, double>> matConfScores;
  double score;
  int nUsers = mat->nrows;
  float *pr = (float*)malloc(sizeof(float)*graphMat->nrows);
  
  for (int u = 0 ; u < mat->nrows; u++) {
    //ignore if invalid user
    //skip if user is invalid
    auto search = invalUsers.find(u);
    if (search != invalUsers.end()) {
      //found n skip
      continue;
    }

    memset(pr, 0, sizeof(float)*graphMat->nrows);
    pr[u] = 1.0;
    
    //run personalized page rank on the graph w.r.t. u
    gk_rw_PageRank(graphMat, lambda, 0.0001, max_niter, pr);

    for (int ii = mat->rowptr[u]; ii < mat->rowptr[u+1]; ii++) {
      int item = mat->rowind[ii];
      //ignore if invalid item
      search = invalItems.find(item);
      if (search != invalItems.end()) {
        //found n skip
        continue;
      }
      score = pr[nUsers + item];
      matConfScores.push_back(std::make_tuple(u, item, score));
    }
  }

  free(pr);
  return genConfidenceCurve(matConfScores, origModel, fullModel, nBuckets, alpha);
}


std::vector<double> computeMissingPPRConf(gk_csr_t* trainMat, 
    gk_csr_t* graphMat, std::unordered_set<int>& invalUsers,
    std::unordered_set<int>& invalItems, float lambda, int max_niter, Model& origModel,
    Model& fullModel, int nBuckets, float alpha) {
  
  std::vector<std::tuple<int, int, double>> matConfScores;
  double score;
  int nUsers = trainMat->nrows;
  std::unordered_set<int> trainItemSet;
  float *pr = (float*)malloc(sizeof(float)*graphMat->nrows);
  
  for (int u = 0 ; u < trainMat->nrows; u++) {
    //ignore if invalid user
    //skip if user is invalid
    auto search = invalUsers.find(u);
    if (search != invalUsers.end()) {
      //found n skip
      continue;
    }

    memset(pr, 0, sizeof(float)*graphMat->nrows);
    pr[u] = 1.0;
    
    //run personalized page rank on the graph w.r.t. u
    gk_rw_PageRank(graphMat, lambda, 0.0001, max_niter, pr);
    
    trainItemSet.clear();
    for (int ii = trainMat->rowptr[u]; ii < trainMat->rowptr[u+1]; ii++) {
      int item = trainMat->rowind[ii];
      trainItemSet.insert(item);
    }

    for (int item = 0; item < trainMat->ncols; item++) {
      //ignore if invalid item
      search = invalItems.find(item);
      if (search != invalItems.end()) {
        //found n skip
        continue;
      }
      //ignore if item in training set
      search = trainItemSet.find(item);
      if (search != trainItemSet.end()) {
        //found n skip
        continue;
      }
      score = pr[nUsers + item];
      matConfScores.push_back(std::make_tuple(u, item, score));
    }
  }

  free(pr);
  return genConfidenceCurve(matConfScores, origModel, fullModel, nBuckets, 
      alpha);
}


std::vector<double> computeMissingPPRConfExt(gk_csr_t* trainMat, 
    gk_csr_t* graphMat, std::unordered_set<int>& invalUsers,
    std::unordered_set<int>& invalItems, float lambda, int max_niter, Model& origModel,
    Model& fullModel, int nBuckets, float alpha, const char* prFName) {
  
  std::vector<std::tuple<int, int, double>> matConfScores;
  std::ifstream inFile (prFName);
  std::string line, token;
  std::string delimiter = " ";
  size_t pos;
  double score;
  int nUsers = trainMat->nrows;
  std::unordered_set<int> trainItemSet;
  float *pr = (float*)malloc(sizeof(float)*graphMat->nrows);
 
  if (inFile.is_open()) {

    for (int u = 0 ; u < trainMat->nrows; u++) {
      
      getline(inFile, line);
      
      //ignore if invalid user
      //skip if user is invalid
      auto search = invalUsers.find(u);
      if (search != invalUsers.end()) {
        //found n skip
        continue;
      }

      memset(pr, 0, sizeof(float)*graphMat->nrows);
      pr[u] = 1.0;
      
      //run personalized page rank on the graph w.r.t. u
      gk_rw_PageRank(graphMat, lambda, 0.0001, max_niter, pr);
      
      trainItemSet.clear();
      for (int ii = trainMat->rowptr[u]; ii < trainMat->rowptr[u+1]; ii++) {
        int item = trainMat->rowind[ii];
        trainItemSet.insert(item);
      }

      for (int i = 0; i < trainMat->ncols; i++) {
        
        //split the line
        pos = line.find(delimiter);
        token = line.substr(0, pos);
        int item = std::stoi(token)-nUsers; 
        
        line.erase(0, pos + delimiter.length());
        
        pos = line.find(delimiter);
        token = line.substr(0, pos);
        score = std::stod(token);
        line.erase(0, pos + delimiter.length());
 
        //ignore if invalid item
        search = invalItems.find(item);
        if (search != invalItems.end()) {
          //found n skip
          continue;
        }
        //ignore if item in training set
        search = trainItemSet.find(item);
        if (search != trainItemSet.end()) {
          //found n skip
          continue;
        }
        score = pr[nUsers + item];
        matConfScores.push_back(std::make_tuple(u, item, score));
      }
    }

  } else {
    std::cerr << "\nCan't open file: " << prFName;
  }

  free(pr);
  return genConfidenceCurve(matConfScores, origModel, fullModel, nBuckets, 
      alpha);
}


std::vector<double> computeMissingPPRConfExtSamp(gk_csr_t* trainMat, 
    gk_csr_t* graphMat, std::unordered_set<int>& invalUsers,
    std::unordered_set<int>& invalItems, float lambda, int max_niter, Model& origModel,
    Model& fullModel, int nBuckets, float alpha, const char* prFName, int seed) {
  
  std::vector<std::tuple<int, int, double>> matConfScores;
  std::ifstream inFile (prFName);
  std::string line, token;
  std::string delimiter = " ";
  size_t pos;
  double score;
  int nUsers = trainMat->nrows;
  int nItems = trainMat->ncols;
  std::vector<std::unordered_set<int>> uTrItemSet(nUsers);
  
  int trNNz = 0;
  for (int u = 0 ; u < trainMat->nrows; u++) {
    for (int ii = trainMat->rowptr[u]; ii < trainMat->rowptr[u+1]; ii++) {
      int item = trainMat->rowind[ii];
      uTrItemSet[u].insert(item);
      trNNz++;
    }
  }
  
  
  double halfRatCount = ((double)nUsers*(double)nItems)/2.0;
  int nScores = std::min((double)MAX_MISS_RATS, halfRatCount);
  int nScoresPerUser = nScores/(nUsers - invalUsers.size());
  
  std::cout << "\nnScores: " << nScores << " nScoresPerUser: " 
    << nScoresPerUser << std::endl;
 
  float *pr = (float*)malloc(sizeof(float)*graphMat->nrows);
 
  if (inFile.is_open()) {

    //random engine
    std::mt19937 mt(seed);
    std::uniform_int_distribution<int> iDist(0, nItems-1);

    for (int user = 0 ; user < trainMat->nrows; user++) {
      
      getline(inFile, line);
      
      //ignore if invalid user
      //skip if user is invalid
      auto search = invalUsers.find(user);
      if (search != invalUsers.end()) {
        //found n skip
        continue;
      }

      memset(pr, 0, sizeof(float)*graphMat->nrows);
      pr[user] = 1.0;
      
      //run personalized page rank on the graph w.r.t. u
      gk_rw_PageRank(graphMat, lambda, 0.0001, max_niter, pr);
      
      //sample valid items
      std::unordered_set<int> uValItems;
      while (uValItems.size() < nScoresPerUser) {
        int item = iDist(mt);
        //ignore if invalid item
        search = invalItems.find(item);
        if (search != invalItems.end()) {
          //found n skip
          continue;
        }

        //ignore if item in training set
        search = uTrItemSet[user].find(item);
        if (search != uTrItemSet[user].end()) {
          //found n skip
          continue;
        }
        uValItems.insert(item);
      }

      for (int i = 0; i < trainMat->ncols; i++) {
        
        //split the line
        pos = line.find(delimiter);
        token = line.substr(0, pos);
        int item = std::stoi(token)-nUsers; 
        
        line.erase(0, pos + delimiter.length());
        
        pos = line.find(delimiter);
        token = line.substr(0, pos);
        score = std::stod(token);
        line.erase(0, pos + delimiter.length());
 
        //check if item was sampled
        search = uValItems.find(item);
        if (search != uValItems.end()) {
          //found
          score = pr[nUsers + item];
          matConfScores.push_back(std::make_tuple(user, item, score));
        }
        
      }

      if (user%PROGU == 0) {
        std::cout << user << " done..." << std::endl; 
      }
    
    }

  } else {
    std::cerr << "\nCan't open file: " << prFName;
  }

  free(pr);
  return genConfidenceCurve(matConfScores, origModel, fullModel, nBuckets, 
      alpha);
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
   
    if (0 == user%PROGU) {
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
   
    if (0 == user % PROGU) {
      std::cout << " u: " << user << std::endl;
    }
  
  }

  for (int i = 0; i < nBuckets; i++) {
    bucketScores[i] = sqrt(bucketScores[i]/bucketNNZ[i]);
  }
  return bucketScores;
}


std::vector<double> confBucketRMSEsWInvalOpPerUser(Model& origModel, Model& fullModel,
    std::vector<Model>& models, int nUsers, int nItems, int nBuckets,
    std::unordered_set<int>& invalUsers, std::unordered_set<int>& invalItems,
    std::string opFileName) {
  
  int nInvalItems = invalItems.size();
  int nItemsPerBuck = (nItems-nInvalItems)/nBuckets;
  
  std::cout << "\nnItemsPerBuck: " << nItemsPerBuck;

  std::vector<double> bucketScores(nBuckets, 0.0);
  std::vector<double> uBucketScores(nBuckets, 0.0);
  std::vector<double> bucketNNZ(nBuckets, 0.0);
  std::vector<double> uBucketNNZ(nBuckets, 0.0);
  double score;
  std::vector<std::pair<int, double>> itemScores;
  std::cout << "\nconfBucketRMSEs: \n"; 
  std::ofstream opFile(opFileName);

  if (opFile.is_open()) {
    
    for (int user = 0; user < nUsers; user++) {
      //skip if user is invalid
      auto search = invalUsers.find(user);
      bool uInval = false;
      if (search != invalUsers.end()) {
        //found 
        uInval = true;
        //continue;
      }

      itemScores.clear();
      for (int item = 0; item < nItems; item++) {
        //skip item if invalid
        auto search = invalItems.find(item);
        bool iInval = false;
        if (search != invalItems.end()) {
          //found
          iInval = true;
          //continue;
        }
        //compute confidence score
        if (uInval || iInval) {
          score = -1;
        } else {
          score = confScore(user, item, models);
          itemScores.push_back(std::make_pair(item, score));
        }
        opFile << score << " ";
      }
      opFile << "\n";
     
      if (uInval) {
        continue;
      }

      //add RMSEs to bucket as per ranking by itemscores
      std::fill(uBucketScores.begin(), uBucketScores.end(), 0);
      std::fill(uBucketNNZ.begin(), uBucketNNZ.end(), 0);
      updateBuckets(user, uBucketScores, uBucketNNZ, itemScores, origModel, fullModel,
          nBuckets, nItemsPerBuck, nItems-nInvalItems);
     
      for (int i = 0; i < nBuckets; i++) {
        bucketScores[i] += uBucketScores[i];
        bucketNNZ[i] += uBucketNNZ[i];
        //write out the u bucket rmse
        //opFile << sqrt(uBucketScores[i]/uBucketNNZ[i]) << " ";
      }
      //opFile << std::endl;

      if (0 == user%PROGU) {
        std::cout << " u: " << user << std::endl;
      }
    
    }
      
    for (int i = 0; i < nBuckets; i++) {
      bucketScores[i] = sqrt(bucketScores[i]/bucketNNZ[i]);
    }
   
    opFile.close();
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
   
    if (0 == user%PROGU) {
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
   
    if (0 == user%PROGU) {
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


std::vector<double> pprBucketRMSEsWInVal(Model& origModel, Model& fullModel, int nUsers, 
    int nItems, float lambda, int max_niter, gk_csr_t *graphMat, int nBuckets, 
    std::unordered_set<int> invalUsers, std::unordered_set<int> invalItems) {
  
  int nInvalItems = invalItems.size();
  int nItemsPerBuck = (nItems-nInvalItems)/nBuckets;
  std::vector<double> bucketScores(nBuckets, 0.0);
  std::vector<double> bucketNNZ(nBuckets, 0.0);
  float *pr = (float*)malloc(sizeof(float)*graphMat->nrows);
  
  std::vector<std::pair<int, double>> itemScores;
  std::cout << "\npprBucketRMSEsWInVal: " << std::endl;
  for (int user = 0; user < nUsers; user++) {
    //skip if user is invalid
    auto search = invalUsers.find(user);
    if (search != invalUsers.end()) {
      //found n skip
      continue;
    }

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

    //add RMSEs to bucket as per ranking by itemscores
    updateBuckets(user, bucketScores, bucketNNZ, itemScores, origModel, fullModel,
        nBuckets, nItemsPerBuck, nItems);
    
    if (user % PROGU == 0) {
      std::cout<< user << " Done..." << std::endl;
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
      
      if (user % PROGU == 0) {
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


std::vector<double> pprBucketRMSEsFrmPRWInVal(Model& origModel, Model& fullModel, int nUsers, 
    int nItems, gk_csr_t *graphMat, int nBuckets, const char* prFName,
    std::unordered_set<int>& invalUsers, std::unordered_set<int>& invalItems) {
  
  int nInvalItems = invalItems.size(); 
  int nItemsPerBuck = (nItems- nInvalItems)/nBuckets;
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
      
      //read prank items of the current user
      getline(inFile, line);
      
      //skip if user is invalid
      auto search = invalUsers.find(user);
      if (search != invalUsers.end()) {
        //found n skip
        continue;
      }

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
        
        //insert only if item is valid
        search = invalItems.find(item);
        if (search == invalItems.end()) {
          //notfound, valid item, insert
          itemScores.push_back(std::make_pair(item, score));
        }
      }

      //add RMSEs to bucket as per ranking by itemscores
      updateBuckets(user, bucketScores, bucketNNZ, itemScores, origModel, fullModel,
          nBuckets, nItemsPerBuck, nItems-nInvalItems);
      
      if (user % PROGU == 0) {
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


