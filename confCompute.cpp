#include "confCompute.h"



void comparePPR2GPR(int nUsers, int nItems, gk_csr_t* graphMat, float lambda,
    int max_niter, const char* prFName, const char* opFName) {
 
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
    itemScores.push_back(std::make_pair(item, pr[i]));
  }

  //sort items by global page rank
  auto comparePair = [](std::pair<int, double> a, std::pair<int, double> b) { 
    return a.second > b.second; 
  };
  
  //sort items by DECREASING order in score
  std::sort(itemScores.begin(), itemScores.end(), comparePair);  
  
  std::vector<int> gprSortedItems;
  for (auto const& itemScore: itemScores) {
    gprSortedItems.push_back(itemScore.first);
  }

  std::unordered_set<int> topGPRSet;
  for (int i = 0; i < 0.25*nItems; i++) {
    topGPRSet.insert(gprSortedItems[i]);
  }

  std::unordered_set<int> mid1GPRSet;
  for (int i = 0.25*nItems; i < 0.5*nItems; i++) {
    mid1GPRSet.insert(gprSortedItems[i]);
  }

  std::unordered_set<int> mid2GPRSet;
  for (int i = 0.5*nItems; i < 0.75*nItems; i++) {
    mid2GPRSet.insert(gprSortedItems[i]);
  }

  std::unordered_set<int> botGPRSet;
  for (int i = 0.75*nItems; i < nItems; i++) {
    botGPRSet.insert(gprSortedItems[i]);
  }

  std::ifstream inFile (prFName);
  std::ofstream opFile (opFName);
  std::string line, token;
  std::string delimiter = " ";
  size_t pos;
  int item;
  double score;
  
  if (inFile.is_open() && opFile.is_open()) {
    std::cout << "\nReading " << prFName << " ..." << std::endl;
    for (int user = 0; user < nUsers; user++) {
  
      //read prank items of the current user
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
        
        //notfound, valid item, insert
        itemScores.push_back(std::make_pair(item, score));
      }
   
      //compute overlap of top 1/4th  item  and bottom 1/4th item
      int topOvCount = 0;
      for (int i = 0; i < nItems*0.25; i++) {
        auto search = topGPRSet.find(itemScores[i].first);
        if (search != topGPRSet.end()) {
          //found
          topOvCount++;
        }
      }
      float topOvPc = (float)topOvCount/(float)(nItems*0.25);

      int mid1OvCount = 0;
      for (int i = 0.25*nItems; i < 0.5*nItems; i++) {
        auto search = mid1GPRSet.find(itemScores[i].first);
        if (search != mid1GPRSet.end()) {
          //found
          mid1OvCount++;
        }
      }
      float mid1OvPc = (float)mid1OvCount/(float)(nItems*0.25);

      int mid2OvCount = 0;
      for (int i = 0.5*nItems; i < 0.75*nItems; i++) {
        auto search = mid2GPRSet.find(itemScores[i].first);
        if (search != mid2GPRSet.end()) {
          //found
          mid2OvCount++;
        }
      }
      float mid2OvPc = (float)mid2OvCount/(float)(nItems*0.25);

      int botOvCount = 0;
      for (int i = 0.75*nItems; i < nItems; i++) {
        auto search = botGPRSet.find(itemScores[i].first);
        if (search != botGPRSet.end()) {
          //found
          botOvCount++;
        }
      }
      float botOvPc = (float)botOvCount/(float)(nItems*0.25);

      opFile << user << " " << topOvPc << " "  << mid1OvPc << " " 
        << mid2OvPc << " " << botOvPc << std::endl;
      
      if (user % PROGU == 0) {
        std::cout << " u: " << user << "," 
          << topOvCount << "," << topOvPc << ","
          << mid1OvCount << "," << mid1OvPc << ","
          << mid2OvCount << "," << mid2OvPc << ","
          << botOvCount << "," << botOvPc << std::endl;
      }

    }
  } else {
    std::cerr << "\nFile NOT opened: " << prFName;
  }


}


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
    std::vector<std::tuple<int, int, double>>& matConfScores, Model& origModel,
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
    int ind = ((float)(1.0 - alpha))*widths.size();
    binWidths.push_back(widths[ind]);
  }


  return binWidths;
}


//passed a vector of <conf score, diff>
std::vector<double> genRMSECurve(
    std::vector<std::pair<double, double>>& confActPredDiffs,
    int nBuckets) { 
   
    std::vector<double> bucketScores(nBuckets, 0);
    std::vector<int> bucketNNZ(nBuckets, 0);

    int nScores = confActPredDiffs.size();
    std::cout << "\nnScores: " << nScores << std::endl;
    int nItemsPerBuck = nScores/nBuckets;
    
    auto comparePair = [](std::pair<double, double> a, 
        std::pair<double, double> b) { 
      bool ret;
      ret = a.first > b.first;
      /* 
      if (a.first == b.first) {
        //break tie randomly
        ret = std::rand() % 2;
      }
      */
      return ret; 
    };
    
    //sort items by DECREASING order in score
    std::sort(confActPredDiffs.begin(), confActPredDiffs.end(), comparePair);  
   
    for (int bInd = 0; bInd < nBuckets; bInd++) {
      int start = bInd*nItemsPerBuck;
      int end = (bInd+1)*nItemsPerBuck;
      if (bInd == nBuckets-1 || end > nScores) {
        end = nScores;
      }
      for (int j = start; j < end; j++) {
        bucketScores[bInd] += confActPredDiffs[j].second*confActPredDiffs[j].second;
        bucketNNZ[bInd] += 1;
      }
    }
    
    double sumSE = 0;
    for (int bInd = 0; bInd < nBuckets; bInd++) {
      sumSE +=  bucketScores[bInd];
      bucketScores[bInd] = sqrt(bucketScores[bInd]/bucketNNZ[bInd]);
    }
    
    std::cout << "\nsumSE = " << sumSE;
    std::cout << "\nbucketCounts = \n" << std::endl;
    for (auto v: bucketNNZ) {
      std::cout << v << " ";
    }
    return bucketScores;
}


std::vector<double> genOptConfRMSECurve(
    std::vector<std::pair<int, int>>& testPairs, Model& origModel,
    Model& fullModel, int nBuckets) {
  std::vector<double> bucketScores(nBuckets, 0);
  std::vector<int> bucketNNZ(nBuckets, 0);
  int nScores = testPairs.size();
  std::cout << "\nnScores: " << nScores << std::endl;
  int nItemsPerBuck = nScores/nBuckets;
  
  std::vector<double> scores;
  for (auto const& testPair: testPairs) {
    int user = testPair.first;
    int item = testPair.second;
    double r_ui = origModel.estRating(user, item);
    double r_ui_est = fullModel.estRating(user, item);
    double w = fabs(r_ui - r_ui_est);
    scores.push_back(w);
  }
  
  //sort scores in ascending order
  std::sort(scores.begin(), scores.end());
 
  std::vector<double> widths;
  for (int bInd = 0; bInd < nBuckets; bInd++) {
    int start = bInd*nItemsPerBuck;
    int end = (bInd+1)*nItemsPerBuck;
    if (bInd == nBuckets-1 || end > nScores) {
      end = nScores;
    }
    for (int j = start; j < end; j++) {
      bucketScores[bInd] += scores[j]*scores[j];
      bucketNNZ[bInd] += 1;
    }
  }
  double sumSE = 0;
  for (int bInd = 0; bInd < nBuckets; bInd++) {
    sumSE += bucketScores[bInd];
    bucketScores[bInd] = sqrt(bucketScores[bInd]/bucketNNZ[bInd]);
  }
  std::cout << "\nsumSE = " << sumSE;
  return bucketScores;
}


std::vector<double> genUserConfRMSECurve(std::vector<std::pair<int, int>>& testPairs, 
    Model& origModel, Model& fullModel, int nBuckets,  
    std::vector<double>& userFreq) {
  //confscore, width
  std::vector<std::pair<double, double>> scores;
  int nScores = testPairs.size();
  std::cout << "\nnScores: " << nScores << std::endl;

  for (auto const& testPair: testPairs) {
    int user = testPair.first;
    int item = testPair.second;
    double r_ui = origModel.estRating(user, item);
    double r_ui_est = fullModel.estRating(user, item);
    double w = fabs(r_ui - r_ui_est);
    scores.push_back(std::make_pair(userFreq[user], w));
  }
  
  return genRMSECurve(scores, nBuckets);
}


std::vector<double> genItemConfRMSECurve(std::vector<std::pair<int, int>>& testPairs, 
    Model& origModel, Model& fullModel, int nBuckets,  
    std::vector<double>& itemFreq) {
  //confscore, width
  std::vector<std::pair<double, double>> scores;
  int nScores = testPairs.size();
  std::cout << "\nnScores: " << nScores << std::endl;

  for (auto const& testPair: testPairs) {
    int user = testPair.first;
    int item = testPair.second;
    double r_ui = origModel.estRating(user, item);
    double r_ui_est = fullModel.estRating(user, item);
    double w = fabs(r_ui - r_ui_est);
    scores.push_back(std::make_pair(itemFreq[item], w));
  }
  
  return genRMSECurve(scores, nBuckets);
}


std::vector<double> genModelConfRMSECurve(std::vector<std::pair<int, int>>& testPairs, 
    Model& origModel, Model& fullModel, std::vector<Model>& models,
    int nBuckets) {
  //confscore, width
  std::vector<std::pair<double, double>> scores;
  int nScores = testPairs.size();
  std::cout << "\nnScores: " << nScores << std::endl;

  for (auto const& testPair: testPairs) {
    int user = testPair.first;
    int item = testPair.second;
    double r_ui = origModel.estRating(user, item);
    double r_ui_est = fullModel.estRating(user, item);
    double w = fabs(r_ui - r_ui_est);
    double cScore = confScore(user, item, models);
    scores.push_back(std::make_pair(cScore, w));
  }
  
  return genRMSECurve(scores, nBuckets);
}


std::vector<double> genGPRConfRMSECurve(std::vector<std::pair<int, int>>& testPairs, 
    Model& origModel, Model& fullModel, gk_csr_t* graphMat, float lambda,
    int max_niter, int nBuckets) {
  //confscore, width
  std::vector<std::pair<double, double>> scores;
  int nScores = testPairs.size();
  std::cout << "\nnScores: " << nScores << std::endl;

  int nUsers = origModel.nUsers;

  float *pr = (float*)malloc(sizeof(float)*graphMat->nrows);
  memset(pr, 0, sizeof(float)*graphMat->nrows);
  //assign all users equal restart probability
  for (int user = 0; user < nUsers; user++) {
    pr[user] = 1.0/nUsers;
  }
  
  //run global page rank on the graph w.r.t. users
  gk_rw_PageRank(graphMat, lambda, 0.0001, max_niter, pr);
 
  for (auto const& testPair: testPairs) {
    int user = testPair.first;
    int item = testPair.second;
    double r_ui = origModel.estRating(user, item);
    double r_ui_est = fullModel.estRating(user, item);
    double w = fabs(r_ui - r_ui_est);
    double confScore = pr[nUsers + item];
    scores.push_back(std::make_pair(confScore, w));
  }
  
  return genRMSECurve(scores, nBuckets);
}


std::vector<double> genPPRConfRMSECurve(std::vector<std::pair<int, int>>& testPairs, 
    Model& origModel, Model& fullModel, gk_csr_t* graphMat, float lambda,
    int max_niter, const char* prFName, int nBuckets) {
  
  std::ifstream inFile (prFName);
  std::string line, token;
  std::string delimiter = " ";
  size_t pos;
  double score;
  int nUsers = origModel.nUsers;
  int nItems = origModel.nItems;

  //confscore, width
  std::vector<std::pair<double, double>> scores;
  
  //ascending order comp
  auto comparePairs = [] (std::pair<int, int> a, std::pair<int, int> b) {
    return a.first < b.first;
  };
  //sort testPairs in ascending order i.e., by user id
  std::sort(testPairs.begin(), testPairs.end(), comparePairs);
  
  if (inFile.is_open()) {

    int testInd = 0;

    for (int user = 0 ; user < nUsers; user++) {
     
      //read ppr for user
      getline(inFile, line);
      
      std::unordered_set<int> uValItems;
      while (testPairs[testInd].first == user) {
        //curr test user is same as user, collect the user's item
        int item = testPairs[testInd].second;  
        uValItems.insert(item);
        testInd++;
      }

      if (uValItems.size() == 0) {
        continue;
      }

      int foundItems = 0;   
      for (int i = 0; i < nItems; i++) {
        
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
        auto search = uValItems.find(item);
        if (search != uValItems.end()) {
          //found
          foundItems++;
          double r_ui = origModel.estRating(user, item);
          double r_ui_est = fullModel.estRating(user, item);
          double w = fabs(r_ui - r_ui_est);
          scores.push_back(std::make_pair(score, w));
        }
        
        if (foundItems == uValItems.size()) {
          //allvalid items found
          break;
        }
      }

      if (user%PROGU == 0) {
        std::cout << user << " done..." << std::endl; 
      }
    
    }

  } else {
    std::cerr << "\nCan't open file: " << prFName;
  }
  
  return genRMSECurve(scores, nBuckets);
}


std::vector<double> genOptConfidenceCurve(
    std::vector<std::pair<int, int>>& testPairs, Model& origModel,
    Model& fullModel, int nBuckets, float alpha) {
  std::vector<double> binWidths;
  std::vector<double> scores;
  int nScores = testPairs.size();
  std::cout << "\nnScores: " << nScores << std::endl;
  int nItemsPerBuck = nScores/nBuckets;

  for (auto const& testPair: testPairs) {
    int user = testPair.first;
    int item = testPair.second;
    double r_ui = origModel.estRating(user, item);
    double r_ui_est = fullModel.estRating(user, item);
    double w = fabs(r_ui - r_ui_est);
    scores.push_back(w);
  }
  
  //sort scores in ascending order
  std::sort(scores.begin(), scores.end());
 
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
      widths.push_back(scores[j]);
    }
    binWidths.push_back(widths[(1-alpha)*widths.size()]);
  }

  return binWidths;
}


std::vector<double> genUserConfCurve(std::vector<std::pair<int, int>>& testPairs, 
    Model& origModel, Model& fullModel, int nBuckets, float alpha, 
    std::vector<double>& userFreq) {
  std::vector<double> binWidths;
  //confscore, width
  std::vector<std::pair<double, double>> scores;
  int nScores = testPairs.size();
  std::cout << "\nnScores: " << nScores << std::endl;
  int nItemsPerBuck = nScores/nBuckets;

  for (auto const& testPair: testPairs) {
    int user = testPair.first;
    int item = testPair.second;
    double r_ui = origModel.estRating(user, item);
    double r_ui_est = fullModel.estRating(user, item);
    double w = fabs(r_ui - r_ui_est);
    scores.push_back(std::make_pair(userFreq[user], w));
  }
  
  //descending order comp of confscore
  auto comparePairs = [] (std::pair<double, double> a, 
                          std::pair<double, double> b) {
    return a.first > b.first;
  };
  //sort scores in descending order by user frequency
  std::sort(scores.begin(), scores.end(), comparePairs);
 
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
      widths.push_back(scores[j].second);
    }
    binWidths.push_back(widths[(1-alpha)*widths.size()]);
  }

  return binWidths;
}


std::vector<double> genItemConfCurve(std::vector<std::pair<int, int>>& testPairs, 
    Model& origModel, Model& fullModel, int nBuckets, float alpha, 
    std::vector<double>& itemFreq) {
  std::vector<double> binWidths;
  //confscore, width
  std::vector<std::pair<double, double>> scores;
  int nScores = testPairs.size();
  std::cout << "\nnScores: " << nScores << std::endl;
  int nItemsPerBuck = nScores/nBuckets;

  for (auto const& testPair: testPairs) {
    int user = testPair.first;
    int item = testPair.second;
    double r_ui = origModel.estRating(user, item);
    double r_ui_est = fullModel.estRating(user, item);
    double w = fabs(r_ui - r_ui_est);
    scores.push_back(std::make_pair(itemFreq[item], w));
  }
  
  //descending order comp of confscore
  auto comparePairs = [] (std::pair<double, double> a, 
                          std::pair<double, double> b) {
    return a.first > b.first;
  };
  //sort scores in descending order by user frequency
  std::sort(scores.begin(), scores.end(), comparePairs);
 
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
      widths.push_back(scores[j].second);
    }
    binWidths.push_back(widths[(1-alpha)*widths.size()]);
  }

  return binWidths;
}


std::vector<std::pair<int, int>> getTestPairs(gk_csr_t* mat, 
    std::unordered_set<int>& invalUsers, std::unordered_set<int>& invalItems,
    int testSize, int seed) {
  
  int nUsers = mat->nrows;
  int nItems = mat->ncols;
  std::vector<std::pair<int, int>> testPairs;
  std::vector<std::unordered_set<int>> uTrItemSet(nUsers);
  
  //random engine
  std::mt19937 mt(seed);
  //user dist
  std::uniform_int_distribution<int> uDist(0, nUsers-1);
  std::uniform_int_distribution<int> iDist(0, nItems-1);

  int trNNz = 0;
  for (int u = 0 ; u < mat->nrows; u++) {
    for (int ii = mat->rowptr[u]; ii < mat->rowptr[u+1]; ii++) {
      int item = mat->rowind[ii];
      uTrItemSet[u].insert(item);
      trNNz++;
    }
  }

  while (testPairs.size() < testSize) {
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
    
    testPairs.push_back(std::make_pair(user, item));
  }
  
  return testPairs;
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


std::vector<double> computeMissingModConfSamp(std::vector<Model>& models,
    Model& origModel, Model& fullModel, int nBuckets, float alpha, 
    std::vector<std::pair<int,int>> testPairs) {
  
  std::vector<std::tuple<int, int, double>> matConfScores;
  double score;
 
  for(const auto &testPair : testPairs) {
    int user = testPair.first;
    int item = testPair.second;
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


std::vector<double> computeMissingGPRConfSamp(gk_csr_t* graphMat, float lambda, 
    int max_niter, Model& origModel, Model& fullModel, int nBuckets, 
    float alpha, std::vector<std::pair<int,int>> testPairs, int nUsers) {
  
  std::vector<std::tuple<int, int, double>> matConfScores;
  double score;
 
  float *pr = (float*)malloc(sizeof(float)*graphMat->nrows);
  memset(pr, 0, sizeof(float)*graphMat->nrows);
  //assign all users equal restart probability
  for (int user = 0; user < nUsers; user++) {
    pr[user] = 1.0/nUsers;
  }
  
  //run global page rank on the graph w.r.t. users
  gk_rw_PageRank(graphMat, lambda, 0.0001, max_niter, pr);
 
  for(const auto &testPair : testPairs) {
    int user = testPair.first;
    int item = testPair.second;
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
        matConfScores.push_back(std::make_tuple(u, item, score));
      }
    }

  } else {
    std::cerr << "\nCan't open file: " << prFName;
  }

  return genConfidenceCurve(matConfScores, origModel, fullModel, nBuckets, 
      alpha);
}


std::vector<double> computeMissingPPRConfExtSamp(gk_csr_t* trainMat, 
    gk_csr_t* graphMat, float lambda, int max_niter, Model& origModel,
    Model& fullModel, int nBuckets, float alpha, const char* prFName, 
    std::vector<std::pair<int, int>> testPairs) {
  
  std::vector<std::tuple<int, int, double>> matConfScores;
  std::ifstream inFile (prFName);
  std::string line, token;
  std::string delimiter = " ";
  size_t pos;
  double score;
  int nUsers = trainMat->nrows;
  int nItems = trainMat->ncols;
  std::vector<std::unordered_set<int>> uTrItemSet(nUsers);
  
  //ascending order comp
  auto comparePairs = [] (std::pair<int, int> a, std::pair<int, int> b) {
    return a.first < b.first;
  };
  //sort testPairs in ascending order
  std::sort(testPairs.begin(), testPairs.end(), comparePairs);

  if (inFile.is_open()) {

    int testInd = 0;

    for (int user = 0 ; user < trainMat->nrows; user++) {
     
      //read ppr for user
      getline(inFile, line);
      
      std::unordered_set<int> uValItems;
      while (testPairs[testInd].first == user) {
        //curr test user is same as user, collect the user's item
        int item = testPairs[testInd].second;  
        uValItems.insert(item);
        testInd++;
      }

      if (uValItems.size() == 0) {
        continue;
      }

      int foundItems = 0;   
      for (int i = 0; i < nItems; i++) {
        
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
        auto search = uValItems.find(item);
        if (search != uValItems.end()) {
          //found
          foundItems++;
          matConfScores.push_back(std::make_tuple(user, item, score));
        }
        
        if (foundItems == uValItems.size()) {
          //allvalid items found
          break;
        }
      }

      if (user%PROGU == 0) {
        std::cout << user << " done..." << std::endl; 
      }
    
    }

  } else {
    std::cerr << "\nCan't open file: " << prFName;
  }

  return genConfidenceCurve(matConfScores, origModel, fullModel, nBuckets, 
      alpha);
}


void updateBuckets(int user, std::vector<double>& bucketScores, 
    std::vector<double>& bucketNNZ, 
    std::vector<std::pair<int, double>>& itemScores, 
    std::map<int, float>& itemRat,
    Model& fullModel, int nBuckets) { 
  
    int nItems = itemScores.size();
    int nItemsPerBuck = nItems/nBuckets;
    
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
        //check if item in itemRat map
        auto search = itemRat.find(item);
        if(search != itemRat.end()) {
          //compute square err for item
          float r_ui = search->second;
          double r_ui_est = fullModel.estRating(user, item);
          double se = (r_ui - r_ui_est)*(r_ui - r_ui_est);
          bucketScores[bInd] += se;
          bucketNNZ[bInd] += 1;
        }
      }
    }
    
}

void updateBuckets(int user, std::vector<double>& bucketScores, 
    std::vector<double>& bucketNNZ, 
    std::vector<std::pair<int, double>>& itemScores, 
    std::map<int, float>& itemRat,
    Model& fullModel, int nBuckets, std::unordered_set<int>& filtItems) { 
  
    int nItems = itemScores.size();
    int nItemsPerBuck = nItems/nBuckets;
    
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
        //check if item in itemRat map
        auto search = itemRat.find(item);
        if(search != itemRat.end()) {
          auto search2 = filtItems.find(item);
          if (search2 == filtItems.end()) { //not found in filtered items
            //compute square err for item
            float r_ui = search->second;
            double r_ui_est = fullModel.estRating(user, item);
            double se = (r_ui - r_ui_est)*(r_ui - r_ui_est);
            bucketScores[bInd] += se;
            bucketNNZ[bInd] += 1;
          }
        }
      }

    }
}

void updateBuckets(int user, std::vector<double>& bucketScores, 
    std::vector<double>& bucketNNZ, 
    std::vector<std::pair<int, double>>& itemScores, Model& origModel,
    Model& fullModel, int nBuckets) { 
    
    int nItems = itemScores.size();
    int nItemsPerBuck = nItems/nBuckets;

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


void updateBucketsOpFile(int user, std::vector<double>& bucketScores, 
    std::vector<double>& bucketNNZ, 
    std::vector<std::pair<int, double>>& itemScores, Model& origModel,
    Model& fullModel, int nBuckets, 
    std::ofstream& opFile) { 
  
    int nItems = itemScores.size();
    int nItemsPerBuck = nItems/nBuckets;
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
      
      double uBuckSE = 0;
      int uBucketNNZ = 0;
      for (int j = start; j < end; j++) {
        int item = itemScores[j].first;
        //compute square err for item
        double r_ui = origModel.estRating(user, item);
        double r_ui_est = fullModel.estRating(user, item);
        double se = (r_ui - r_ui_est)*(r_ui - r_ui_est);
        bucketScores[bInd] += se;
        bucketNNZ[bInd] += 1;
        uBuckSE += se;
        uBucketNNZ += 1;
      }
      opFile << sqrt(uBuckSE/uBucketNNZ) << " ";
    }

    opFile << std::endl;
}


void updateBucketsSorted(int user, std::vector<double>& bucketScores, 
    std::vector<double>& bucketNNZ, 
    std::vector<int>& sortedItems, Model& origModel,
    Model& fullModel, int nBuckets, int nItemsPerBuck) {

    int nItems = sortedItems.size(); 
    for (int bInd = 0; bInd < nBuckets; bInd++) {
      int start = bInd*nItemsPerBuck;
      int end = (bInd+1)*nItemsPerBuck;
      if (bInd == nBuckets-1 || end > nItems) {
        end = nItems;
      }
      for (int j = start; j < end; j++) {
        int item = sortedItems[j];
        //compute square err for item
        double r_ui = origModel.estRating(user, item);
        double r_ui_est = fullModel.estRating(user, item);
        double se = (r_ui - r_ui_est)*(r_ui - r_ui_est);
        bucketScores[bInd] += se;
        bucketNNZ[bInd] += 1;
      }
    }
}


void updateBucketsSorted(int user, std::vector<double>& bucketScores, 
    std::vector<double>& bucketNNZ, 
    std::vector<int>& sortedItems, std::map<int, float>& itemRat,
    Model& fullModel, int nBuckets, int nItemsPerBuck, 
    std::unordered_set<int>& remItems) {

    int nItems = sortedItems.size(); 
    for (int bInd = 0; bInd < nBuckets; bInd++) {
      int start = bInd*nItemsPerBuck;
      int end = (bInd+1)*nItemsPerBuck;
      if (bInd == nBuckets-1 || end > nItems) {
        end = nItems;
      }
      for (int j = start; j < end; j++) {
        int item = sortedItems[j];
        //check if item in itemRat map
        auto search = itemRat.find(item);
        if(search != itemRat.end()) {
          
          auto search2 = remItems.find(item);
          if (search2 != remItems.end()) {
            //found n skip
            continue;
          }

          //compute square err for item
          double r_ui = search->second;
          double r_ui_est = fullModel.estRating(user, item);
          double se = (r_ui - r_ui_est)*(r_ui - r_ui_est);
          bucketScores[bInd] += se;
          bucketNNZ[bInd] += 1;
        }
      }
    }
}

void updateBucketsSorted(int user, std::vector<double>& bucketScores, 
    std::vector<double>& bucketNNZ, 
    std::vector<int>& sortedItems, std::map<int, float>& itemRat,
    Model& fullModel, int nBuckets, int nItemsPerBuck) {

    int nItems = sortedItems.size(); 
    for (int bInd = 0; bInd < nBuckets; bInd++) {
      int start = bInd*nItemsPerBuck;
      int end = (bInd+1)*nItemsPerBuck;
      if (bInd == nBuckets-1 || end > nItems) {
        end = nItems;
      }
      for (int j = start; j < end; j++) {
        int item = sortedItems[j];
        //check if item in itemRat map
        auto search = itemRat.find(item);
        if(search != itemRat.end()) {
          //compute square err for item
          double r_ui = search->second;
          double r_ui_est = fullModel.estRating(user, item);
          double se = (r_ui - r_ui_est)*(r_ui - r_ui_est);
          bucketScores[bInd] += se;
          bucketNNZ[bInd] += 1;
        }
      }
    }
}

void updateBucketsSortedOpFile(int user, std::vector<double>& bucketScores, 
    std::vector<double>& bucketNNZ, 
    std::vector<int>& sortedItems, Model& origModel,
    Model& fullModel, int nBuckets, int nItemsPerBuck, std::ofstream& opFile) {

    int nItems = sortedItems.size(); 
    for (int bInd = 0; bInd < nBuckets; bInd++) {
      int start = bInd*nItemsPerBuck;
      int end = (bInd+1)*nItemsPerBuck;
      if (bInd == nBuckets-1 || end > nItems) {
        end = nItems;
      }
      double uBuckSE = 0;
      int uBuckNNZ = 0;
      for (int j = start; j < end; j++) {
        int item = sortedItems[j];
        //compute square err for item
        double r_ui = origModel.estRating(user, item);
        double r_ui_est = fullModel.estRating(user, item);
        double se = (r_ui - r_ui_est)*(r_ui - r_ui_est);
        bucketScores[bInd] += se;
        bucketNNZ[bInd] += 1;
        uBuckSE += se;
        uBuckNNZ += 1;
      }
      opFile << sqrt(uBuckSE/uBuckNNZ) << " ";
    }
    opFile << std::endl;
}


std::vector<double> confBucketRMSEs(Model& origModel, Model& fullModel,
    std::vector<Model>& models, int nUsers, int nItems, int nBuckets) {
  
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
        nBuckets);
   
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
  
  std::cout << "\nnItemsPerBuck: " << nItemsPerBuck << " nItems: " << nItems
    << " nInvalItems: " << nInvalItems << " nBuckets: " << nBuckets;

  std::vector<double> bucketScores(nBuckets, 0.0);
  std::vector<double> bucketNNZ(nBuckets, 0.0);
  double score;
  std::vector<std::pair<int, double>> itemScores;
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
        nBuckets);
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
          nBuckets);
     
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
  std::cout << "\nnItemsPerBuck: " << nItemsPerBuck;
  std::vector<double> bucketScores(nBuckets, 0.0);
  std::vector<double> bucketNNZ(nBuckets, 0.0);
  double score;
  std::vector<std::pair<int, double>> itemScores;
 
  //function to sort pairs in ascending order
  auto comparePair = [](std::pair<int, double> a, std::pair<int, double> b) { 
    return a.second < b.second; 
  };

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
  
  }

  for (int i = 0; i < nBuckets; i++) {
    bucketScores[i] = sqrt(bucketScores[i]/bucketNNZ[i]);
  }
  return bucketScores;
}


//NOTE: graphMat is structure graph of training matrix with nodes: nUsers+nItems
std::vector<double> pprBucketRMSEs(Model& origModel, Model& fullModel, int nUsers, 
    int nItems, float lambda, int max_niter, gk_csr_t *graphMat, int nBuckets) {
  
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
        nBuckets);
    
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
        nBuckets);
    
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


std::vector<double> pprSampBucketRMSEsWInVal(Model& fullModel, gk_csr_t *mat, 
    float lambda, int max_niter, gk_csr_t *graphMat, int nBuckets, 
    std::unordered_set<int>& invalUsers, std::unordered_set<int>& invalItems, 
    int nSampUsers, int seed) {
  
  int nUsers = mat->nrows;
  int nItems = mat->ncols;

  std::cout << "\nnUsers: " << nUsers;
  std::cout << "\nnItems: " << nItems;
  
  std::vector<double> bucketScores(nBuckets, 0.0);
  std::vector<double> bucketNNZ(nBuckets, 0.0);
  float *pr = (float*)malloc(sizeof(float)*graphMat->nrows);
  
  std::vector<std::pair<int, double>> itemScores;
  std::cout << "\npprBucketRMSEsWInVal: " << std::endl;
  std::map<int, float> itemRatings;
  //initialize random engine
  std::mt19937 mt(seed);
  //user distribution to sample users
  std::uniform_int_distribution<int> uDist(0, nUsers-1);
 
  std::unordered_set<int> sampUsers;

  while(sampUsers.size() < nSampUsers) {
    //sample user
    int user = uDist(mt);
    auto search = sampUsers.find(user);
    if (search != sampUsers.end()) {
      //already sampled user
      continue;
    }
    sampUsers.insert(user);

    //skip if user is invalid
    search = invalUsers.find(user);
    if (search != invalUsers.end()) {
      //found n skip
      continue;
    }

    if (mat->rowptr[user] - mat->rowptr[user+1] == 0) {
      //no items found for user
      continue;
    }
 
    //get map of items,rating for user
    itemRatings.clear();
    for (int ii = mat->rowptr[user]; ii < mat->rowptr[user+1]; ii++) {
      int item = mat->rowind[ii];
      //check if invalid item
      search = invalItems.find(item);
      if (search != invalItems.end()) {
        // skip if invalid
        continue;
      }
      float itemRat = mat->rowval[ii];
      itemRatings[item] = itemRat;
    }
   
    if (itemRatings.size() == 0) {
      //cant find ratings due to invalid items
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

    //update buckets for given items of the user
    updateBuckets(user, bucketScores, bucketNNZ, itemScores, itemRatings, fullModel,
        nBuckets);

    if (sampUsers.size() % PROGU == 0) {
      std::cout << sampUsers.size() << " done..." << std::endl;
    }

  }


  std::cout << "\nNo. samp users: " << sampUsers.size();

  free(pr);
 
  std::cout << "\nppr bucket nnz: ";
  for (int i = 0; i < nBuckets; i++) {
    if (bucketNNZ[i] > 0) {
      bucketScores[i] = sqrt(bucketScores[i]/bucketNNZ[i]);
    }
    std::cout << bucketNNZ[i] << " ";
  }
  return bucketScores;
}


std::vector<double> pprSampBucketRMSEsWInVal(Model& fullModel, gk_csr_t *mat, 
    float lambda, int max_niter, gk_csr_t *graphMat, int nBuckets, 
    std::unordered_set<int>& invalUsers, std::unordered_set<int>& invalItems, 
    std::unordered_set<int>& filtItems, int nSampUsers, int seed) {
  
  int nUsers = mat->nrows;
  int nItems = mat->ncols;

  std::cout << "\nnUsers: " << nUsers;
  std::cout << "\nnItems: " << nItems;
  
  std::vector<double> bucketScores(nBuckets, 0.0);
  std::vector<double> bucketNNZ(nBuckets, 0.0);
  float *pr = (float*)malloc(sizeof(float)*graphMat->nrows);
  
  std::vector<std::pair<int, double>> itemScores;
  std::cout << "\npprBucketRMSEsWInVal: " << std::endl;
  std::map<int, float> itemRatings;
  //initialize random engine
  std::mt19937 mt(seed);
  //user distribution to sample users
  std::uniform_int_distribution<int> uDist(0, nUsers-1);
  
  std::unordered_set<int> sampUsers;
  
  while(sampUsers.size() < nSampUsers) {
    int user = uDist(mt);
    //skip if user already sampled
    auto search = sampUsers.find(user);
    if (search != sampUsers.end()) {
      //found n skip
      continue;
    }

    //skip if user is invalid
    search = invalUsers.find(user);
    if (search != invalUsers.end()) {
      //found n skip
      continue;
    }

    if (mat->rowptr[user] - mat->rowptr[user+1] == 0) {
      //no items found for user
      continue;
    }
 
    //get map of items,rating for user
    itemRatings.clear();
    for (int ii = mat->rowptr[user]; ii < mat->rowptr[user+1]; ii++) {
      int item = mat->rowind[ii];
      //check if invalid item
      search = invalItems.find(item);
      if (search != invalItems.end()) {
        // skip if invalid
        continue;
      }
      float itemRat = mat->rowval[ii];
      itemRatings[item] = itemRat;
    }
   
    if (itemRatings.size() == 0) {
      //cant find ratings due to invalid items
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

    //update buckets for given items of the user
    updateBuckets(user, bucketScores, bucketNNZ, itemScores, itemRatings, fullModel,
        nBuckets, filtItems);

    if (sampUsers.size() % PROGU == 0) {
      std::cout << sampUsers.size() << " done..." << std::endl;
    }

  }
  
  free(pr);
 
  std::cout << "\nppr bucket nnz: ";
  for (int i = 0; i < nBuckets; i++) {
    if (bucketNNZ[i] > 0) {
      bucketScores[i] = sqrt(bucketScores[i]/bucketNNZ[i]);
    }
    std::cout << bucketNNZ[i] << " ";
  }
  std::cout << std::endl;
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
  
  std::vector<int> sortedItems;
  for (auto const& itemScore: itemScores) {
    sortedItems.push_back(itemScore.first);
  }

  //add RMSEs to bucket as per ranking by itemscores
  for (int user = 0; user < nUsers; user++) {
    //skip if user is invalid
    auto search = invalUsers.find(user);
    if (search != invalUsers.end()) {
      //found n skip
      continue;
    }
    
    updateBucketsSorted(user, bucketScores, bucketNNZ, sortedItems, origModel,
        fullModel, nBuckets, nItemsPerBuck);
  }

  for (int i = 0; i < nBuckets; i++) {
    bucketScores[i] = sqrt(bucketScores[i]/bucketNNZ[i]);
  }
    
  free(pr);
  return bucketScores;
}


std::vector<double> gprSampBucketRMSEsWInVal(Model& fullModel, gk_csr_t *mat,
    float lambda, int max_niter, gk_csr_t *graphMat, int nBuckets,
    std::unordered_set<int>& invalUsers, std::unordered_set<int>& invalItems, 
    int nSampUsers, int seed) {


  int nUsers = mat->nrows;
  int nItems = mat->ncols;

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
  
  std::vector<int> sortedItems;
  for (auto const& itemScore: itemScores) {
    sortedItems.push_back(itemScore.first);
  }

  std::map<int, float> itemRatings;
  //initialize random engine
  std::mt19937 mt(seed);
  //user distribution to sample users
  std::uniform_int_distribution<int> uDist(0, nUsers-1);
 
  std::unordered_set<int> sampUsers;

 
  //add RMSEs to bucket as per ranking by itemscores
  while (sampUsers.size() < nSampUsers) {
    //sample user
    int user = uDist(mt);
    
    auto search = sampUsers.find(user);
    if (search != sampUsers.end()) {
      //already sampled user
      continue;
    }

    sampUsers.insert(user);
    
    //skip if user is invalid
    search = invalUsers.find(user);
    if (search != invalUsers.end()) {
      //found n skip
      continue;
    }
    
    if (mat->rowptr[user] - mat->rowptr[user+1] == 0) {
      //no items found for user
      continue;
    }

    //get map of items,rating for user
    itemRatings.clear();
    for (int ii = mat->rowptr[user]; ii < mat->rowptr[user+1]; ii++) {
      int item = mat->rowind[ii];
      //check if invalid item
      search = invalItems.find(item);
      if (search != invalItems.end()) {
        // skip if invalid
        continue;
      }
      float itemRat = mat->rowval[ii];
      itemRatings[item] = itemRat;
    }

    if (itemRatings.size() == 0) {
      //cant find ratings due to invalid items
      continue;
    }

    updateBucketsSorted(user, bucketScores, bucketNNZ, sortedItems, itemRatings,
        fullModel, nBuckets, nItemsPerBuck);
    
  }
  
  std::cout << "\nNo. samp users: " << sampUsers.size();

  std::cout << "\ngpr bucket NNZ: ";
  for (int i = 0; i < nBuckets; i++) {
    bucketScores[i] = sqrt(bucketScores[i]/bucketNNZ[i]);
    std::cout << bucketNNZ[i] << " ";
  }
  free(pr);
  return bucketScores;
}


std::vector<double> gprSampBucketRMSEsWInVal(Model& fullModel, gk_csr_t *mat,
    float lambda, int max_niter, gk_csr_t *graphMat, int nBuckets,
    std::unordered_set<int>& invalUsers, std::unordered_set<int>& invalItems, 
    std::unordered_set<int>& freqItems, int nSampUsers, int seed) {


  int nUsers = mat->nrows;
  int nItems = mat->ncols;

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
  
  std::vector<int> sortedItems;
  for (auto const& itemScore: itemScores) {
    sortedItems.push_back(itemScore.first);
  }

  std::map<int, float> itemRatings;
  //initialize random engine
  std::mt19937 mt(seed);
  //user distribution to sample users
  std::uniform_int_distribution<int> uDist(0, nUsers-1);
  
  
  std::unordered_set<int> sampUsers;

  //add RMSEs to bucket as per ranking by itemscores
  while (sampUsers.size() < nSampUsers) {
    //sample user
    int user = uDist(mt); 
    auto search = sampUsers.find(user);
    if (search != sampUsers.end()) {
      //already sampled user
      continue;
    }
    sampUsers.insert(user);
    //skip if user is invalid
    search = invalUsers.find(user);
    if (search != invalUsers.end()) {
      //found n skip
      continue;
    }
    
    if (mat->rowptr[user] - mat->rowptr[user+1] == 0) {
      //no items found for user
      continue;
    }

    //get map of items,rating for user
    itemRatings.clear();
    for (int ii = mat->rowptr[user]; ii < mat->rowptr[user+1]; ii++) {
      int item = mat->rowind[ii];
      //check if invalid item
      search = invalItems.find(item);
      if (search != invalItems.end()) {
        // skip if invalid
        continue;
      }
      float itemRat = mat->rowval[ii];
      itemRatings[item] = itemRat;
    }

    if (itemRatings.size() == 0) {
      //cant find ratings due to invalid items
      continue;
    }

    updateBucketsSorted(user, bucketScores, bucketNNZ, sortedItems, itemRatings,
        fullModel, nBuckets, nItemsPerBuck, freqItems);
    
  }

  std::cout << "\ngpr bucket NNZ: ";
  for (int i = 0; i < nBuckets; i++) {
    bucketScores[i] = sqrt(bucketScores[i]/bucketNNZ[i]);
    std::cout << bucketNNZ[i] << " ";
  }
  std::cout << std::endl;

  free(pr);
  return bucketScores;
}


std::vector<double> itemFreqBucketRMSEsWInVal(Model& origModel, 
    Model& fullModel, int nUsers, int nItems, 
    std::vector<double>& itemFreq, int nBuckets, 
    std::unordered_set<int>& invalUsers, std::unordered_set<int>& invalItems) {
  
  int nInvalItems = invalItems.size();
  int nItemsPerBuck = (nItems-nInvalItems)/nBuckets;
  std::vector<double> bucketScores(nBuckets, 0.0);
  std::vector<double> bucketNNZ(nBuckets, 0.0);
  std::vector<std::pair<int, double>> itemScores;

  for (int item = 0; item < nItems; item++) {
    //skip item if invalid
    auto search = invalItems.find(item);
    if (search != invalItems.end()) {
      //found n skip
      continue;
    }
    itemScores.push_back(std::make_pair(item, itemFreq[item])); 
  }

  //sort items by item frequency in decreasing order
  auto comparePair = [](std::pair<int, double> a, std::pair<int, double> b) { 
    return a.second > b.second; 
  };
  
  //sort items by DECREASING order in score
  std::sort(itemScores.begin(), itemScores.end(), comparePair);  
  
  std::vector<int> sortedItems;
  for (auto const& itemScore: itemScores) {
    sortedItems.push_back(itemScore.first);
  }

  //add RMSEs to bucket as per ranking by itemscores
  for (int user = 0; user < nUsers; user++) {
    //skip if user is invalid
    auto search = invalUsers.find(user);
    if (search != invalUsers.end()) {
      //found n skip
      continue;
    }
    
    updateBucketsSorted(user, bucketScores, bucketNNZ, sortedItems, origModel,
        fullModel, nBuckets, nItemsPerBuck);
  }

  for (int i = 0; i < nBuckets; i++) {
    bucketScores[i] = sqrt(bucketScores[i]/bucketNNZ[i]);
  }
  
  return bucketScores;
}


std::vector<double> itemFreqSampBucketRMSEsWInVal(gk_csr_t* mat, 
    Model& fullModel, 
    std::vector<double>& itemFreq, int nBuckets, 
    std::unordered_set<int>& invalUsers, 
    std::unordered_set<int>& invalItems, int nSampUsers, int seed) {
  
  int nUsers = mat->nrows;
  int nItems = mat->ncols;
  
  int nInvalItems = invalItems.size();
  int nItemsPerBuck = (nItems-nInvalItems)/nBuckets;
  std::vector<double> bucketScores(nBuckets, 0.0);
  std::vector<double> bucketNNZ(nBuckets, 0.0);
  std::vector<std::pair<int, double>> itemScores;

  for (int item = 0; item < nItems; item++) {
    //skip item if invalid
    auto search = invalItems.find(item);
    if (search != invalItems.end()) {
      //found n skip
      continue;
    }
    itemScores.push_back(std::make_pair(item, itemFreq[item])); 
  }

  //sort items by item frequency in decreasing order
  auto comparePair = [](std::pair<int, double> a, std::pair<int, double> b) { 
    return a.second > b.second; 
  };
  
  //sort items by DECREASING order in score
  std::sort(itemScores.begin(), itemScores.end(), comparePair);  
  
  std::vector<int> sortedItems;
  for (auto const& itemScore: itemScores) {
    sortedItems.push_back(itemScore.first);
  }

  std::map<int, float> itemRatings;
  //initialize random engine
  std::mt19937 mt(seed);
  //user distribution to sample users
  std::uniform_int_distribution<int> uDist(0, nUsers-1);
  
  std::unordered_set<int> sampUsers;

  while(sampUsers.size() < nSampUsers) {
    //sample user
    int user = uDist(mt); 
    
    auto search = sampUsers.find(user);
    if (search != sampUsers.end()) {
      //already sampled
      continue;
    }
    
    sampUsers.insert(user);
    
    //skip if user is invalid
    search = invalUsers.find(user);
    if (search != invalUsers.end()) {
      //found n skip
      continue;
    }
    
    if (mat->rowptr[user] - mat->rowptr[user+1] == 0) {
      //no items found for user
      continue;
    }

    //get map of items,rating for user
    itemRatings.clear();
    for (int ii = mat->rowptr[user]; ii < mat->rowptr[user+1]; ii++) {
      int item = mat->rowind[ii];
      //check if invalid item
      search = invalItems.find(item);
      if (search != invalItems.end()) {
        // skip if invalid
        continue;
      }
      float itemRat = mat->rowval[ii];
      itemRatings[item] = itemRat;
    }
    
    if (itemRatings.size() == 0) {
      //couldnt find a test rating for user due to invalid items
      continue;
    }

    updateBucketsSorted(user, bucketScores, bucketNNZ, sortedItems, itemRatings,
        fullModel, nBuckets, nItemsPerBuck);

  }

  std::cout << "\nNo. samp users: " << sampUsers.size();

  std::cout << "\niFreq bucket nnz: ";
  for (int i = 0; i < nBuckets; i++) {
    bucketScores[i] = sqrt(bucketScores[i]/bucketNNZ[i]);
    std::cout << bucketNNZ[i] << " ";
  }
  

  return bucketScores;
}


std::vector<double> itemFreqSampBucketRMSEsWInVal(gk_csr_t* mat, 
    Model& fullModel, 
    std::vector<double>& itemFreq, 
    int nBuckets, 
    std::unordered_set<int>& invalUsers, 
    std::unordered_set<int>& invalItems, std::unordered_set<int>& filtItems,
    int nSampUsers, int seed) {
  
  int nUsers = mat->nrows;
  int nItems = mat->ncols;
  
  int nInvalItems = invalItems.size();
  int nItemsPerBuck = (nItems-nInvalItems)/nBuckets;
  std::vector<double> bucketScores(nBuckets, 0.0);
  std::vector<double> bucketNNZ(nBuckets, 0.0);
  std::vector<std::pair<int, double>> itemScores;

  for (int item = 0; item < nItems; item++) {
    //skip item if invalid
    auto search = invalItems.find(item);
    if (search != invalItems.end()) {
      //found n skip
      continue;
    }
    itemScores.push_back(std::make_pair(item, itemFreq[item])); 
  }

  //sort items by item frequency in decreasing order
  auto comparePair = [](std::pair<int, double> a, std::pair<int, double> b) { 
    return a.second > b.second; 
  };
  
  //sort items by DECREASING order in score
  std::sort(itemScores.begin(), itemScores.end(), comparePair);  
  
  std::vector<int> sortedItems;
  for (auto const& itemScore: itemScores) {
    sortedItems.push_back(itemScore.first);
  }

  std::map<int, float> itemRatings;
  //initialize random engine
  std::mt19937 mt(seed);
  //user distribution to sample users
  std::uniform_int_distribution<int> uDist(0, nUsers-1);
  std::unordered_set<int> sampUsers;

  while(sampUsers.size() < nSampUsers) {
    //sample user
    int user = uDist(mt); 
    auto search = sampUsers.find(user);
    if (search != sampUsers.end()) {
      //already sampled
      continue;
    }
    //add to sampled set
    sampUsers.insert(user);

    //skip if user is invalid
    search = invalUsers.find(user);
    if (search != invalUsers.end()) {
      //found n skip
      continue;
    }
    
    if (mat->rowptr[user] - mat->rowptr[user+1] == 0) {
      //no items found for user
      continue;
    }

    //get map of items,rating for user
    itemRatings.clear();
    for (int ii = mat->rowptr[user]; ii < mat->rowptr[user+1]; ii++) {
      int item = mat->rowind[ii];
      //check if invalid item
      search = invalItems.find(item);
      if (search != invalItems.end()) {
        // skip if invalid
        continue;
      }
      float itemRat = mat->rowval[ii];
      itemRatings[item] = itemRat;
    }
    
    if (itemRatings.size() == 0) {
      //couldnt find a test rating for user due to invalid items
      continue;
    }

    updateBucketsSorted(user, bucketScores, bucketNNZ, sortedItems, itemRatings,
        fullModel, nBuckets, nItemsPerBuck, filtItems);

  }
 
  std::cout << "\niFreq bucket nnz: ";
  for (int i = 0; i < nBuckets; i++) {
    bucketScores[i] = sqrt(bucketScores[i]/bucketNNZ[i]);
    std::cout << bucketNNZ[i] << " ";
  }
  std::cout << std::endl;

  return bucketScores;
}

std::vector<double> pprBucketRMSEsFrmPR(Model& origModel, Model& fullModel, int nUsers, 
    int nItems, gk_csr_t *graphMat, int nBuckets, const char* prFName) {
  
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
          nBuckets);
      
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
          nBuckets);
      
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


