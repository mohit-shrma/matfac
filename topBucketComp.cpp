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


double getSE(int user, std::vector<std::pair<int, double>> itemScores, 
    Model& origModel, Model& fullModel, int n) {
  double score = 0, diff = 0;
  for (int i = 0; i < n; i++) {
    int item = itemScores[i].first;
    diff = origModel.estRating(user, item) - fullModel.estRating(user, item);
    score += diff*diff;
  }
  return score;
}


std::vector<std::pair<int, double>> uIGraphItemScores(int user, 
    gk_csr_t *graphMat, float lambda, int nUsers, int nItems,
    std::unordered_set<int>& invalItems) {

  std::vector<std::pair<int, double>> itemScores;
  float *pr = (float*) malloc(sizeof(float)*graphMat->nrows);
  memset(pr, 0, sizeof(float)*graphMat->nrows);
  pr[user] = 1.0;

  //run personalized page rank on the graph w.r.t. u
  gk_rw_PageRank(graphMat, lambda, 0.0001, MAX_PR_ITER, pr);

  //get pr scores of items
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
  
  free(pr);

  return itemScores;
}


std::vector<std::pair<int, double>> zero1Scale(
    std::vector<std::pair<int, double>>& pairScores) {
  std::vector<std::pair<int, double>> scaledPairs;
  double maxScore = -1;
  for (auto&& pair: pairScores) {
    if (maxScore < pair.second) {
      maxScore = pair.second;
    }
  }

  for (auto&& pair: pairScores) {
    scaledPairs.push_back(std::make_pair(pair.first, pair.second/maxScore));
  } 
  return scaledPairs;
}


std::vector<std::pair<int, double>> avgScores(
    std::vector<std::pair<int, double>>& pairScores1,
    std::vector<std::pair<int, double>>& pairScores2) {
  
  std::vector<std::pair<int, double>> resScores;

  std::map<int, double> mapScores1;
  for (auto&& pairScore: pairScores1) {
    mapScores1[pairScore.first] = pairScore.second;
  }
  
  for (auto&& pairScore: pairScores2) {
    int key = pairScore.first;
    double score = pairScore.second;
    if (mapScores1.find(key) != mapScores1.end()) {
      resScores.push_back(std::make_pair(key, 
            ((.95*mapScores1[key]) + (0.05*score))));
    }
  }
  
  std::sort(resScores.begin(), resScores.end(), descComp);
  
  return resScores;
}


std::vector<std::pair<int, double>> prodScores(
    std::vector<std::pair<int, double>>& pairScores1,
    std::vector<std::pair<int, double>>& pairScores2) {
  
  std::vector<std::pair<int, double>> resScores;

  std::map<int, double> mapScores1;
  for (auto&& pairScore: pairScores1) {
    mapScores1[pairScore.first] = pairScore.second;
  }
  
  for (auto&& pairScore: pairScores2) {
    int key = pairScore.first;
    double score = pairScore.second;
    if (mapScores1.find(key) != mapScores1.end()) {
      resScores.push_back(std::make_pair(key, mapScores1[key]*score));
    }
  }
  
  std::sort(resScores.begin(), resScores.end(), descComp);
  
  return resScores;
}


//x + alpha log(y)
std::vector<std::pair<int, double>> logSumScores(
    std::vector<std::pair<int, double>>& pairScores1,
    std::vector<std::pair<int, double>>& pairScores2, double alpha) {
  
  std::vector<std::pair<int, double>> resScores;

  std::map<int, double> mapScores1;
  for (auto&& pairScore: pairScores1) {
    mapScores1[pairScore.first] = pairScore.second;
  }
  
  for (auto&& pairScore: pairScores2) {
    int key = pairScore.first;
    double score = pairScore.second;
    if (mapScores1.find(key) != mapScores1.end()) {
      resScores.push_back(std::make_pair(key, 
            mapScores1[key] + (alpha * log(score))));
    }
  }
  
  std::sort(resScores.begin(), resScores.end(), descComp);
  
  return resScores;
}


//x - alpha log(y)
std::vector<std::pair<int, double>> logDiffScores(
    std::vector<std::pair<int, double>>& pairScores1,
    std::vector<std::pair<int, double>>& pairScores2, double alpha) {
  
  std::vector<std::pair<int, double>> resScores;

  std::map<int, double> mapScores1;
  for (auto&& pairScore: pairScores1) {
    mapScores1[pairScore.first] = pairScore.second;
  }
  
  for (auto&& pairScore: pairScores2) {
    int key = pairScore.first;
    double score = pairScore.second;
    if (mapScores1.find(key) != mapScores1.end()) {
      resScores.push_back(std::make_pair(key, 
            mapScores1[key] - (alpha * log(score))));
    }
  }
  
  std::sort(resScores.begin(), resScores.end(), descComp);
  
  return resScores;
}


//log(x) + alpha log(y)
std::vector<std::pair<int, double>> logLogSumScores(
    std::vector<std::pair<int, double>>& pairScores1,
    std::vector<std::pair<int, double>>& pairScores2, double alpha) {
  
  std::vector<std::pair<int, double>> resScores;

  std::map<int, double> mapScores1;
  for (auto&& pairScore: pairScores1) {
    mapScores1[pairScore.first] = pairScore.second;
  }
  
  for (auto&& pairScore: pairScores2) {
    int key = pairScore.first;
    double score = pairScore.second;
    if (mapScores1.find(key) != mapScores1.end()) {
      resScores.push_back(std::make_pair(key, 
            log(mapScores1[key]) + (alpha * log(score))));
    }
  }
  
  std::sort(resScores.begin(), resScores.end(), descComp);
  
  return resScores;
}


//log(x) - alpha log(y)
std::vector<std::pair<int, double>> logLogDiffScores(
    std::vector<std::pair<int, double>>& pairScores1,
    std::vector<std::pair<int, double>>& pairScores2, double alpha) {
  
  std::vector<std::pair<int, double>> resScores;

  std::map<int, double> mapScores1;
  for (auto&& pairScore: pairScores1) {
    mapScores1[pairScore.first] = pairScore.second;
  }
  
  for (auto&& pairScore: pairScores2) {
    int key = pairScore.first;
    double score = pairScore.second;
    if (mapScores1.find(key) != mapScores1.end()) {
      resScores.push_back(std::make_pair(key, 
            log(mapScores1[key]) - (alpha * log(score))));
    }
  }
  
  std::sort(resScores.begin(), resScores.end(), descComp);
  
  return resScores;
}


std::vector<std::pair<int, double>> itemGraphItemScores(int user, 
    gk_csr_t *graphMat, gk_csr_t *mat, float lambda, int nUsers, 
    int nItems, std::unordered_set<int>& invalItems, bool useRatings) {

  std::vector<std::pair<int, double>> itemScores;
  float *pr = (float*) malloc(sizeof(float)*graphMat->nrows);
  memset(pr, 0, sizeof(float)*graphMat->nrows);

  if (graphMat->nrows != nItems) {
    std::cerr << "No. of items not equal to nodes in graph" << std::endl;
  }

  int nUserRat = mat->rowptr[user+1] - mat->rowptr[user];
  float sumRat = 0;
  for (int ii = mat->rowptr[user]; ii < mat->rowptr[user+1]; ii++) {
    int item = mat->rowind[ii];
    sumRat += mat->rowval[ii];
  }
  
  for (int ii = mat->rowptr[user]; ii < mat->rowptr[user+1]; ii++) {
    int item = mat->rowind[ii];
    if (sumRat > 0 && useRatings) {
      pr[item] = mat->rowval[ii]/sumRat; 
    } else {
      pr[item] = 1.0/nUserRat;
    }
  }
  
  //run personalized page rank on the graph w.r.t. u
  int iter = gk_rw_PageRank(graphMat, lambda, 0.0001, MAX_PR_ITER, pr);
  
  if (iter > MAX_PR_ITER) {
    std::cerr << "\n page rank not converged: " << user;
  }

  //get pr scores of items
  for (int item = 0; item < nItems; item++) {
    //skip if item is invalid
    auto search = invalItems.find(item);
    if (search != invalItems.end()) {
      //found, invalid, skip
      continue;
    }
    itemScores.push_back(std::make_pair(item, pr[item]));
  }
  
  //sort items in decreasing order of score
  std::sort(itemScores.begin(), itemScores.end(), descComp);
  
  free(pr);

  return itemScores;
}


std::vector<std::pair<int, double>> itemSVDScores(Model& svdModel, int user,
    gk_csr_t *mat, int nItems, std::unordered_set<int>& invalItems) {

  std::vector<std::pair<int, double>> itemScores;
  
  //asuuming matrix is sorted get training items
  std::vector<int> trainItems;
  for (int ii = mat->rowptr[user]; ii < mat->rowptr[user+1]; ii++) {
    int item = mat->rowind[ii];
    trainItems.push_back(item);
  }

  //get pr scores of items
  for (int item = 0; item < nItems; item++) {
    //skip if item is invalid
    auto search = invalItems.find(item);
    if (search != invalItems.end()) {
      //found, invalid, skip
      continue;
    }
    /*
    //skip if found in train items
    if (std::binary_search(trainItems.begin(), trainItems.end(), item)) {
      continue;
    }
    */
    itemScores.push_back(std::make_pair(item, svdModel.estRating(user, item)));
  }
  
  //sort items in decreasing order of score
  std::sort(itemScores.begin(), itemScores.end(), descComp);
  
  return itemScores;
}


std::vector<std::pair<int, double>> itemOptScores(Model& origModel, 
    Model& fullModel, int user,
    gk_csr_t *mat, int nItems, std::unordered_set<int>& invalItems) {

  std::vector<std::pair<int, double>> itemScores;
  
  //get pr scores of items
  for (int item = 0; item < nItems; item++) {
    //skip if item is invalid
    auto search = invalItems.find(item);
    if (search != invalItems.end()) {
      //found, invalid, skip
      continue;
    }
    auto origRating = origModel.estRating(user, item);
    auto predRating = fullModel.estRating(user, item);
    auto diff = fabs(origRating - predRating);
    //since need to be sorted by ascending order add negative of diff
    itemScores.push_back(std::make_pair(item, -1*diff));
  }
  
  //sort items in decreasing order of score
  std::sort(itemScores.begin(), itemScores.end(), descComp);
  
  return itemScores;
}


std::vector<std::pair<int, double>> itemPredScores( 
    Model& fullModel, int user,
    gk_csr_t *mat, int nItems, std::unordered_set<int>& invalItems) {

  std::vector<std::pair<int, double>> itemScores;
  
  //asuuming matrix is sorted get training items
  std::vector<int> trainItems;
  for (int ii = mat->rowptr[user]; ii < mat->rowptr[user+1]; ii++) {
    int item = mat->rowind[ii];
    trainItems.push_back(item);
  }

  //get scores of items
  for (int item = 0; item < nItems; item++) {
    //skip if item is invalid
    auto search = invalItems.find(item);
    if (search != invalItems.end()) {
      //found, invalid, skip
      continue;
    }
    
    //skip if found in train items
    if (std::binary_search(trainItems.begin(), trainItems.end(), item)) {
      continue;
    }
    
    auto predRating = fullModel.estRating(user, item);
    itemScores.push_back(std::make_pair(item, predRating));
  }
  
  //sort items in decreasing order of score
  std::sort(itemScores.begin(), itemScores.end(), descComp);
  
  return itemScores;
}


std::vector<std::pair<int, double>> itemOrigScores( 
    Model& origModel, int user,
    gk_csr_t *mat, int nItems, std::unordered_set<int>& invalItems) {

  std::vector<std::pair<int, double>> itemScores;
  
  //asuuming matrix is sorted get training items
  std::vector<int> trainItems;
  for (int ii = mat->rowptr[user]; ii < mat->rowptr[user+1]; ii++) {
    int item = mat->rowind[ii];
    trainItems.push_back(item);
  }

  //get scores of items
  for (int item = 0; item < nItems; item++) {
    //skip if item is invalid
    auto search = invalItems.find(item);
    if (search != invalItems.end()) {
      //found, invalid, skip
      continue;
    }
    
    //skip if found in train items
    if (std::binary_search(trainItems.begin(), trainItems.end(), item)) {
      continue;
    }
    
    auto origRating = origModel.estRating(user, item);
    itemScores.push_back(std::make_pair(item, origRating));
  }
  
  //sort items in decreasing order of score
  std::sort(itemScores.begin(), itemScores.end(), descComp);
  
  return itemScores;
}


std::vector<std::pair<int, double>> itemFreqScores( 
    Model& fullModel, Model& origModel, std::vector<double> itemFreq,
    int user,
    gk_csr_t *mat, int nItems, std::unordered_set<int>& invalItems) {

  std::vector<std::pair<int, double>> itemScores;
  
  //asuuming matrix is sorted get training items
  std::vector<int> trainItems;
  for (int ii = mat->rowptr[user]; ii < mat->rowptr[user+1]; ii++) {
    int item = mat->rowind[ii];
    trainItems.push_back(item);
  }

  //get scores of items
  for (int item = 0; item < nItems; item++) {
    //skip if item is invalid
    auto search = invalItems.find(item);
    if (search != invalItems.end()) {
      //found, invalid, skip
      continue;
    }
    
    //skip if found in train items
    if (std::binary_search(trainItems.begin(), trainItems.end(), item)) {
      continue;
    }
    
    auto predRating = fullModel.estRating(user, item);
    auto origRating = origModel.estRating(user, item);
    itemScores.push_back(std::make_pair(item, itemFreq[item]));
  }
  
  //sort items in decreasing order of score
  std::sort(itemScores.begin(), itemScores.end(), descComp);
  
  return itemScores;
}


std::vector<std::pair<int, double>> iterSort(
    std::vector<std::pair<int, double>>& itemScoresPair1,
    std::vector<std::pair<int, double>>& itemScoresPair2, int n) {

  //first sort 1
  std::sort(itemScoresPair1.begin(), itemScoresPair1.end(), descComp);

  std::map<int, double> scoreMap;
  for (auto&& itemScore: itemScoresPair2) {
    scoreMap[itemScore.first] = itemScore.second; 
  }
  
  std::vector<std::pair<int, double>> firstNPairs;
  for (int i = 0; i < n; i++) {
    firstNPairs.push_back(std::make_pair(itemScoresPair1[i].first, 
          scoreMap[itemScoresPair1[i].first]));
  }
  std::sort(firstNPairs.begin(), firstNPairs.end(), descComp);

  return firstNPairs;
}


void compOverlapPcLaterBins(std::vector<std::pair<int, double>>& predScorePairs,
    std::vector<std::pair<int, double>>& svdScorePairs,
    std::vector<std::pair<int, double>>& origScorePairs, 
    std::vector<double>& pcPredBins, std::vector<double>& pcSVDBins,
    int topBuckN) {
  
  std::map<int, double> svdScoreMap;
  std::unordered_set<int> topItems;

  for (auto&& itemScore: svdScorePairs) {
    svdScoreMap[itemScore.first] = itemScore.second;
  }

  for (int i = 0; i < topBuckN; i++) {
    topItems.insert(origScorePairs[i].first);
  }

  auto halfTopPredScorePairs = std::vector<std::pair<int, double>> (
      predScorePairs.begin()+(topBuckN/2), predScorePairs.begin()+topBuckN);

  int nBins = pcPredBins.size();
  int nItemsPerBin = (halfTopPredScorePairs.size())/nBins;

  std::vector<std::pair<int, double>> halfTopSVDScorePairs;
  for (auto&& itemScore: halfTopPredScorePairs) {
    int item = itemScore.first;
    halfTopSVDScorePairs.push_back(std::make_pair(item, svdScoreMap[item])); 
  }
  std::sort(halfTopSVDScorePairs.begin(), halfTopSVDScorePairs.end(), descComp);

  for (int i = 0; i < nBins; i++) {
    int found1 = 0;
    int found2 = 0;
    int startInd = nItemsPerBin*i;
    int endInd = nItemsPerBin*(i+1);
    
    if (i == nBins-1) {
      endInd = halfTopPredScorePairs.size();
    }
    
    for (int j = startInd; j < endInd; j++) {
      int item = halfTopPredScorePairs[j].first;
      if (topItems.find(item) != topItems.end()) {
        found1++; 
      } 
      item = halfTopSVDScorePairs[j].first;
      if (topItems.find(item) != topItems.end()) {
        found2++; 
      } 
    }

    pcPredBins[i] += (double)found1 / (endInd - startInd);
    pcSVDBins[i] += (double)found2 / (endInd - startInd);
  } 

  
}


void compOverlapPcBins(std::vector<std::pair<int, double>>& itemScorePairs,
    std::vector<std::pair<int, double>>& origScorePairs, 
    std::vector<double>& pcBins, int topBuckN) {
 
  int nBins = pcBins.size();
  int nItemsPerBin = topBuckN/nBins;
  std::unordered_set<int> topItems;

  //assuming origScorePairs is sorted, get topBuckN items
  for (int i = 0; i < topBuckN; i++) {
    topItems.insert(origScorePairs[i].first);  
  }

  for (int i = 0; i < nBins; i++) {
    int found = 0;
    int startInd = nItemsPerBin*i;
    int endInd = nItemsPerBin*(i+1);
    if (i == nBins-1) {
      endInd = topBuckN;
      if (endInd > itemScorePairs.size()) { 
        endInd = itemScorePairs.size();
      }
    }
    
    for (int j = startInd; j < endInd; j++) {
      int item = itemScorePairs[j].first;
      if (topItems.find(item) != topItems.end()) {
        found++;
      } 
    }
    
    pcBins[i] += (double)found / (endInd - startInd);
  } 

}



void itemRMSEsProb(int user, gk_csr_t *graphMat, gk_csr_t *trainMat, 
    float lambda, int nUsers, int nItems, std::unordered_set<int>& invalItems,
    std::unordered_set<int>& filtItems, Model& fullModel, Model& origModel,
    std::vector<double>& itemsRMSE, std::vector<double>& itemsProb) {
  
  std::vector<std::pair<int, double>> itemScores = itemGraphItemScores(user, 
      graphMat, trainMat, lambda, nUsers, nItems, invalItems, true);
  
  itemsRMSE.clear();
  itemsProb.clear();
  for (auto&& itemScore: itemScores) {
    int item = itemScore.first;
    double score = itemScore.second;
    double r_ui = origModel.estRating(user, item);
    double r_ui_est = fullModel.estRating(user, item);
    double se = (r_ui - r_ui_est)*(r_ui - r_ui_est);
    itemsRMSE.push_back(se);
    itemsProb.push_back(r_ui_est);
  }
}


void itemRMSEsOrdByItemScores(int user, std::unordered_set<int>& filtItems, 
    Model& fullModel, Model& origModel, std::vector<double>& itemsRMSE, 
    std::vector<double>& itemsScore, 
    std::vector<std::pair<int,double>>& itemScoresPair) {
  itemsRMSE.clear();
  itemsScore.clear();
  for (auto&& itemScore: itemScoresPair) {
    int item = itemScore.first;
    if (filtItems.find(item) != filtItems.end()) {
      //skip found in filtered items
      continue;
    }
    double score = itemScore.second;
    double r_ui = origModel.estRating(user, item);
    double r_ui_est = fullModel.estRating(user, item);
    double se = (r_ui - r_ui_est)*(r_ui - r_ui_est);
    itemsRMSE.push_back(se);
    itemsScore.push_back(r_ui);
  }
}


void pprUsersRMSEProb(gk_csr_t *graphMat, gk_csr_t *trainMat, 
    int nUsers, int nItems, Model& origModel, Model& fullModel,
    float lambda, std::unordered_set<int>& invalUsers, 
    std::unordered_set<int>& invalItems, std::unordered_set<int>& filtItems, 
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
 
    itemRMSEsProb(user, graphMat, trainMat, lambda, nUsers, nItems, 
        invalItems, filtItems, fullModel, origModel, itemsRMSE, itemsProb);
    
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


void pprSampUsersRMSEProb(gk_csr_t *graphMat, gk_csr_t *trainMat, 
    int nUsers, int nItems, Model& origModel, Model& fullModel,
    float lambda, int max_niter, std::unordered_set<int>& invalUsers, 
    std::unordered_set<int>& invalItems, std::unordered_set<int>& filtItems, 
    int nSampUsers, int seed, std::string prefix) {
  
  int nBuckets = 10;
  int nItemsPerBuck = nItems/nBuckets;

  std::vector<double> rmseScores(nBuckets, 0);
  std::vector<double> uRMSEScores(nBuckets, 0);
  
  std::vector<double> probs(nBuckets, 0);
  std::vector<double> uProbs(nBuckets, 0);
  
  std::vector<double> bucketNNZ(nBuckets, 0);
  std::vector<double> uBucketNNZ(nBuckets, 0);

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

    itemRMSEsProb(user, graphMat, trainMat, lambda, nUsers, nItems, invalItems,
        filtItems, fullModel, origModel, itemsRMSE, itemsProb);
    
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
  
  rmseOpFile.close();
  probOpFile.close();

  for (int i = 0; i < nBuckets; i++) {
    rmseScores[i] = sqrt(rmseScores[i]/bucketNNZ[i]);
    probs[i] = probs[i]/bucketNNZ[i];
  }
  std::string rmseScoreFName = prefix + "_rmseBuckets.txt";
  writeVector(rmseScores, rmseScoreFName.c_str());
  
  std::string avgScoreFName = prefix + "_avgScoreByPR.txt";
  writeVector(probs, avgScoreFName.c_str());

  //generate stats for sampled users
  /*
  std::vector<int> users;
  for (auto&& user: sampUsers) {
    users.push_back(user);
  }
  std::string statsFName = prefix + "_userStats.txt";
  getUserStats(users, trainMat, invalItems, statsFName.c_str());
  */  
}


void freqSampUsersRMSEProb(gk_csr_t *trainMat, 
    int nUsers, int nItems, Model& origModel, Model& fullModel,
    std::unordered_set<int>& invalUsers, 
    std::unordered_set<int>& invalItems, std::unordered_set<int>& filtItems, 
    int nSampUsers, int seed, std::string prefix) {
  
  int nBuckets = 10;
  int nItemsPerBuck = nItems/nBuckets;

  std::vector<double> rmseScores(nBuckets, 0);
  std::vector<double> uRMSEScores(nBuckets, 0);
  
  std::vector<double> scores(nBuckets, 0);
  std::vector<double> uScores(nBuckets, 0);
  
  std::vector<double> bucketNNZ(nBuckets, 0);
  std::vector<double> uBucketNNZ(nBuckets, 0);

  std::vector<double> itemsRMSE;
  std::vector<double> itemsScore;

  //initialize random engine
  std::mt19937 mt(seed);
  //user distribution to sample users
  std::uniform_int_distribution<int> uDist(0, nUsers-1);

  std::unordered_set<int> sampUsers;

  std::string rmseFName = prefix + "_uRMSE.txt" ;
  std::ofstream rmseOpFile(rmseFName.c_str()); 

  std::string scoreFName = prefix + "_uScore.txt";
  std::ofstream scoreOpFile(scoreFName.c_str());

  auto rowColFreq = getRowColFreq(trainMat);
  auto userFreq = rowColFreq.first;
  auto itemFreq = rowColFreq.second;
  
  std::vector<std::pair<int, double>> itemScoresPair;
  for (int i = 0; i < itemFreq.size(); i++) {
    itemScoresPair.push_back(std::make_pair(i, itemFreq[i]));
  }
  //readItemScores(itemScores, "bwscores2.txt");
  //sort items in decreasing order of score
  std::sort(itemScoresPair.begin(), itemScoresPair.end(), descComp);

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
    
    itemRMSEsOrdByItemScores(user, filtItems, fullModel, origModel, itemsRMSE,
        itemsScore, itemScoresPair);

    //reset user specific vec to 0
    std::fill(uBucketNNZ.begin(), uBucketNNZ.end(), 0); 
    std::fill(uRMSEScores.begin(), uRMSEScores.end(), 0);
    updateBucketsArr(uRMSEScores, uBucketNNZ, itemsRMSE, nBuckets);
    
    //reset user specific vec to 0
    std::fill(uBucketNNZ.begin(), uBucketNNZ.end(), 0); 
    std::fill(uScores.begin(), uScores.end(), 0);
    updateBucketsArr(uScores, uBucketNNZ, itemsScore, nBuckets);

    //write and update aggregated buckets
    rmseOpFile << user << " ";
    scoreOpFile << user << " ";
    for (int i = 0; i < nBuckets; i++) {
      bucketNNZ[i]  += uBucketNNZ[i];
      rmseScores[i] += uRMSEScores[i];
      rmseOpFile << sqrt(uRMSEScores[i]/uBucketNNZ[i]) << " ";
      scores[i]      += uScores[i];
      scoreOpFile << uScores[i]/uBucketNNZ[i] << " ";
    }
    rmseOpFile << std::endl;
    scoreOpFile << std::endl;
    
    if (sampUsers.size() % PROGU == 0) {
      std::cout << "Done... " << sampUsers.size()  << std::endl;
    }

  }
  
  rmseOpFile.close();
  scoreOpFile.close();

  for (int i = 0; i < nBuckets; i++) {
    rmseScores[i] = sqrt(rmseScores[i]/bucketNNZ[i]);
    scores[i] = scores[i]/bucketNNZ[i];
  }
  std::string rmseScoreFName = prefix + "_rmseBuckets.txt";
  writeVector(rmseScores, rmseScoreFName.c_str());
  
  std::string avgScoreFName = prefix + "_avgScoreByFreq.txt";
  writeVector(scores, avgScoreFName.c_str());

  //generate stats for sampled users
  /*
  std::vector<int> users;
  for (auto&& user: sampUsers) {
    users.push_back(user);
  }
  std::string statsFName = prefix + "_userStats.txt";
  getUserStats(users, trainMat, invalItems, statsFName.c_str());
  */  
}


void svdSampUsersRMSEProb(gk_csr_t *trainMat, int nUsers, int nItems, 
    Model& origModel, Model& fullModel, Model& svdModel,
    std::unordered_set<int>& invalUsers, std::unordered_set<int>& invalItems, 
    std::unordered_set<int>& filtItems, 
    int nSampUsers, int seed, std::string prefix) {
  
  int nBuckets = 10;
  int nItemsPerBuck = nItems/nBuckets;

  std::vector<double> rmseScores(nBuckets, 0);
  std::vector<double> uRMSEScores(nBuckets, 0);
  
  std::vector<double> scores(nBuckets, 0);
  std::vector<double> uScores(nBuckets, 0);
  
  std::vector<double> bucketNNZ(nBuckets, 0);
  std::vector<double> uBucketNNZ(nBuckets, 0);

  std::vector<double> itemsRMSE;
  std::vector<double> itemsScore;

  //initialize random engine
  std::mt19937 mt(seed);
  //user distribution to sample users
  std::uniform_int_distribution<int> uDist(0, nUsers-1);

  std::unordered_set<int> sampUsers;

  std::string rmseFName = prefix + "_uRMSE.txt" ;
  std::ofstream rmseOpFile(rmseFName.c_str()); 

  std::string scoreFName = prefix + "_uScore.txt" ;
  std::ofstream scoreOpFile(scoreFName.c_str()); 
  
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
    
    //get item scores ordered by SVD in decreasing order
    auto itemScoresPair = itemSVDScores(svdModel, user, trainMat, nItems, 
        invalItems);
    
    //get itemsRMSE and itemsScore
    itemRMSEsOrdByItemScores(user, filtItems, fullModel, origModel, itemsRMSE, 
        itemsScore, itemScoresPair);
    
    //reset user specific vec to 0
    std::fill(uBucketNNZ.begin(), uBucketNNZ.end(), 0); 
    std::fill(uRMSEScores.begin(), uRMSEScores.end(), 0);
    updateBucketsArr(uRMSEScores, uBucketNNZ, itemsRMSE, nBuckets);

    //reset user specific vec to 0
    std::fill(uBucketNNZ.begin(), uBucketNNZ.end(), 0); 
    std::fill(uScores.begin(), uScores.end(), 0);
    updateBucketsArr(uScores, uBucketNNZ, itemsScore, nBuckets);
    
    //write and update aggregated buckets
    rmseOpFile << user << " ";
    for (int i = 0; i < nBuckets; i++) {
      bucketNNZ[i]  += uBucketNNZ[i];
      rmseScores[i] += uRMSEScores[i];
      rmseOpFile << sqrt(uRMSEScores[i]/uBucketNNZ[i]) << " ";
      scores[i]     += uScores[i];
      scoreOpFile << uScores[i]/uBucketNNZ[i] << " ";
    }
    rmseOpFile << std::endl;
    scoreOpFile << std::endl;

    if (sampUsers.size() % PROGU == 0) {
      std::cout << "Done... " << sampUsers.size() << " :" << std::endl;
    }

  }
  
  rmseOpFile.close();
  scoreOpFile.close();

  for (int i = 0; i < nBuckets; i++) {
    rmseScores[i] = sqrt(rmseScores[i]/bucketNNZ[i]);
    scores[i] = scores[i]/bucketNNZ[i];
  }
  std::string rmseScoreFName = prefix + "_rmseBuckets.txt";
  writeVector(rmseScores, rmseScoreFName.c_str());

  std::string avgScoreFName = prefix + "_avgScore.txt";
  writeVector(scores, avgScoreFName.c_str());

  /*
  //generate stats for sampled users
  std::vector<int> users;
  for (auto&& user: sampUsers) {
    users.push_back(user);
  }
  std::string statsFName = prefix + "_userStats.txt";
  getUserStats(users, trainMat, invalItems, statsFName.c_str());
  */  
}


void optSampUsersRMSEProb(gk_csr_t *trainMat, int nUsers, int nItems, 
    Model& origModel, Model& fullModel, 
    std::unordered_set<int>& invalUsers, std::unordered_set<int>& invalItems, 
    std::unordered_set<int>& filtItems, 
    int nSampUsers, int seed, std::string prefix) {
  
  int nBuckets = 10;
  int nItemsPerBuck = nItems/nBuckets;

  std::vector<double> rmseScores(nBuckets, 0);
  std::vector<double> uRMSEScores(nBuckets, 0);
  
  std::vector<double> scores(nBuckets, 0);
  std::vector<double> uScores(nBuckets, 0);
  
  std::vector<double> bucketNNZ(nBuckets, 0);
  std::vector<double> uBucketNNZ(nBuckets, 0);

  std::vector<double> itemsRMSE;
  std::vector<double> itemsScore;

  //initialize random engine
  std::mt19937 mt(seed);
  //user distribution to sample users
  std::uniform_int_distribution<int> uDist(0, nUsers-1);

  std::unordered_set<int> sampUsers;

  std::string rmseFName = prefix + "_uRMSE.txt" ;
  std::ofstream rmseOpFile(rmseFName.c_str()); 

  std::string scoreFName = prefix + "_uScore.txt" ;
  std::ofstream scoreOpFile(scoreFName.c_str()); 
  
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
    
    //get item scores ordered by SVD in decreasing order
    auto itemScoresPair = itemOptScores(origModel, fullModel, user, trainMat, 
        nItems, invalItems);
    
    //get itemsRMSE and itemsScore
    itemRMSEsOrdByItemScores(user, filtItems, fullModel, origModel, itemsRMSE, 
        itemsScore, itemScoresPair);
    
    //reset user specific vec to 0
    std::fill(uBucketNNZ.begin(), uBucketNNZ.end(), 0); 
    std::fill(uRMSEScores.begin(), uRMSEScores.end(), 0);
    updateBucketsArr(uRMSEScores, uBucketNNZ, itemsRMSE, nBuckets);

    //reset user specific vec to 0
    std::fill(uBucketNNZ.begin(), uBucketNNZ.end(), 0); 
    std::fill(uScores.begin(), uScores.end(), 0);
    updateBucketsArr(uScores, uBucketNNZ, itemsScore, nBuckets);
    
    //write and update aggregated buckets
    rmseOpFile << user << " ";
    for (int i = 0; i < nBuckets; i++) {
      bucketNNZ[i]  += uBucketNNZ[i];
      rmseScores[i] += uRMSEScores[i];
      rmseOpFile << sqrt(uRMSEScores[i]/uBucketNNZ[i]) << " ";
      scores[i]     += uScores[i];
      scoreOpFile << uScores[i]/uBucketNNZ[i] << " ";
    }
    rmseOpFile << std::endl;
    scoreOpFile << std::endl;

    if (sampUsers.size() % PROGU == 0) {
      std::cout << "Done... " << sampUsers.size() << " :" << std::endl;
    }

  }
  
  rmseOpFile.close();
  scoreOpFile.close();

  for (int i = 0; i < nBuckets; i++) {
    rmseScores[i] = sqrt(rmseScores[i]/bucketNNZ[i]);
    scores[i] = scores[i]/bucketNNZ[i];
  }
  std::string rmseScoreFName = prefix + "_rmseBuckets.txt";
  writeVector(rmseScores, rmseScoreFName.c_str());

  std::string avgScoreFName = prefix + "_avgScore.txt";
  writeVector(scores, avgScoreFName.c_str());

  /*
  //generate stats for sampled users
  std::vector<int> users;
  for (auto&& user: sampUsers) {
    users.push_back(user);
  }
  std::string statsFName = prefix + "_userStats.txt";
  getUserStats(users, trainMat, invalItems, statsFName.c_str());
  */  
}


void countInvRatings(Model& origModel, Model& fullModel, int user, 
    std::vector<std::pair<int, double>> itemPreds, float avgRat,
    int& predLowcount, int& predHighcount, int& lowCount, int& highCount, int N) {
  
  for (int i = 0; i < N; i++) {
    
    auto itemPred = itemPreds[i];
    auto item = itemPred.first;
    auto origRat = origModel.estRating(user, item);
    auto predRat = fullModel.estRating(user, item);    
    
    if (origRat >= avgRat*1.1) {
      highCount++;
      //originally rated high
      if (predRat <= avgRat*0.9) {
        //predicted low
        predLowcount++;
      }
    } else if (origRat <= avgRat*0.9) {
      //originally rated low
      lowCount++;
      if (predRat >= avgRat*1.1) {
        //predicted high
        predHighcount++;
      }
    }
    
  }
}


//items in B, also present in A
std::vector<std::pair<int, double>> orderingOverlap(
    std::vector<std::pair<int, double>> itemPairsA,
    std::vector<std::pair<int, double>> itemPairsB, int sizeA) {
  
  std::unordered_set<int> itemsA;
  std::vector<std::pair<int, double>> intersect;

  for (int i = 0; i < sizeA; i++) {
    int item = itemPairsA[i].first;
    itemsA.insert(item);
  }
  
  for (int i = 0; i < sizeA && i < itemPairsB.size(); i++) {
    int item = itemPairsB[i].first;
    if (itemsA.find(item) != itemsA.end()) {
      //intersect
      intersect.push_back(itemPairsB[i]);
    }
  }

  return intersect;
}


//items in B not in A
std::vector<std::pair<int, double>> orderingDiff(
    std::vector<std::pair<int, double>> itemPairsA,
    std::vector<std::pair<int, double>> itemPairsB, int sizeA) {
  int overlapCount = 0;
  
  std::unordered_set<int> itemsA;
  std::vector<std::pair<int, double>> diff;

  for (int i = 0; i < sizeA; i++) {
    int item = itemPairsA[i].first;
    itemsA.insert(item);
  }
  
  for (int i = 0; i < sizeA && i < itemPairsB.size(); i++) {
    int item = itemPairsB[i].first;
    if (itemsA.find(item) == itemsA.end()) {
      //diff
      diff.push_back(itemPairsB[i]);
    }
  }
  return diff;
}


std::pair<double, double> compOrderingOverlapBScores(
    std::vector<std::pair<int, double>> itemPairsA,
    std::vector<std::pair<int, double>> itemPairsB, int sizeA) {
  int overlapCount = 0;
 
  double overlap_BScore = -1;
  double notInA_BScore = -1;

  if (itemPairsA.size() == 0 || itemPairsB.size() == 0) {
    return std::make_pair(-1, -1);
  }
  
  std::unordered_set<int> itemsA;
  for (int i = 0; i < sizeA; i++) {
    int item = itemPairsA[i].first;
    itemsA.insert(item);
  }
  
  for (int i = 0; i < sizeA; i++) {
    int item = itemPairsB[i].first;
    if (itemsA.find(item) != itemsA.end()) {
      //overlap
      overlapCount++;
      overlap_BScore += itemPairsB[i].second;
    } else {
      //not in A
      notInA_BScore += itemPairsB[i].second;
    }
  }
    
  if (overlapCount > 0) {
    overlap_BScore = overlap_BScore/overlapCount;
  }

  if (sizeA - overlapCount > 0) {
    notInA_BScore = notInA_BScore/(sizeA - overlapCount);
  }

  return std::make_pair(overlap_BScore, notInA_BScore);
}


std::pair<double, double> compDiffPc(
    std::vector<std::pair<int, double>>& itemPairsA, 
    std::vector<std::pair<int, double>>& itemPairsB) {
  int ovCount = 0;
  double diffPcA = 0, diffPcB = 0;
  std::unordered_set<int> itemsA;
  if (itemPairsA.size() && itemPairsB.size()) {
    for (auto&& pair: itemPairsA) {
      itemsA.insert(pair.first);
    }
    for (auto&& pair: itemPairsB) {
      int item = pair.first;
      if (itemsA.find(item) != itemsA.end()) {
        ovCount++;
      }
    }
    
    diffPcA = ((double)(itemsA.size() - ovCount))/itemsA.size();
    diffPcB = ((double)(itemPairsB.size() - ovCount))/itemPairsB.size();
  }
  return std::make_pair(diffPcA, diffPcB);
}


double compOrderingOverlap2(std::vector<std::pair<int, double>> itemPairsA,
    std::vector<std::pair<int, double>> itemPairsB, int sizeA) {
  int overlapCount = 0;
  
  if (itemPairsA.size() == 0 || itemPairsB.size() == 0) {
    return overlapCount;
  }
  
  std::unordered_set<int> itemsA;
  for (int i = 0; i < sizeA; i++) {
    int item = itemPairsA[i].first;
    itemsA.insert(item);
  }
 
  int parseBSz = 0;
  for (int i = 0; i < itemPairsB.size() && i < sizeA; i++) {
    int item = itemPairsB[i].first;
    if (itemsA.find(item) != itemsA.end()) {
      //overlap
      overlapCount++;
    }
    parseBSz++;
  }

  return (double)overlapCount/ parseBSz;
}


double compOrderingOverlap(std::vector<std::pair<int, double>> itemPairsA,
    std::vector<std::pair<int, double>> itemPairsB, int sizeA) {
  int overlapCount = 0;
  
  if (itemPairsA.size() == 0 || itemPairsB.size() == 0) {
    return overlapCount;
  }
  
  std::unordered_set<int> itemsA;
  for (int i = 0; i < sizeA; i++) {
    int item = itemPairsA[i].first;
    itemsA.insert(item);
  }
  
  for (int i = 0; i < sizeA; i++) {
    int item = itemPairsB[i].first;
    if (itemsA.find(item) != itemsA.end()) {
      //overlap
      overlapCount++;
    }
  }
  

  return (double)overlapCount/ (double)itemsA.size();
}


void predSampUsersRMSEProb(gk_csr_t *trainMat, gk_csr_t *graphMat, 
    int nUsers, int nItems, 
    Model& origModel, Model& fullModel, Model& svdModel, 
    std::unordered_set<int>& invalUsers, std::unordered_set<int>& invalItems, 
    std::unordered_set<int>& filtItems, 
    int nSampUsers, int seed, std::string prefix) {
  
  int nBuckets = 10;
  int nItemsPerBuck = nItems/nBuckets;

  std::vector<double> rmseScores(nBuckets, 0);
  std::vector<double> uRMSEScores(nBuckets, 0);
  
  std::vector<double> scores(nBuckets, 0);
  std::vector<double> uScores(nBuckets, 0);
  
  std::vector<double> bucketNNZ(nBuckets, 0);
  std::vector<double> uBucketNNZ(nBuckets, 0);

  std::vector<double> itemsRMSE;
  std::vector<double> itemsScore;

  //initialize random engine
  std::mt19937 mt(seed);
  //user distribution to sample users
  std::uniform_int_distribution<int> uDist(0, nUsers-1);

  std::unordered_set<int> sampUsers;

  std::string rmseFName = prefix + "_uRMSE.txt" ;
  std::ofstream rmseOpFile(rmseFName.c_str()); 

  std::string scoreFName = prefix + "_uScore.txt" ;
  std::ofstream scoreOpFile(scoreFName.c_str()); 
 
  auto avgTrainRating = meanRating(trainMat);
  
  int predLowcount = 0, predHighcount = 0;
  int lowCount = 0, highCount = 0;
  
  auto rowColFreq = getRowColFreq(trainMat);
  auto userFreq = rowColFreq.first;
  auto itemFreq = rowColFreq.second;

  int topBuckN = 0.05*nItems;

  std::cout << "\ntopBuckN: " << topBuckN << std::endl;

  int nHighPredUsers = 0;

  double predOrigOverlap = 0, svdOrigOverlap = 0, predSVDOrigOverlap = 0;
  double pprOrigOverlap = 0;
  double svdNotInPredOrigOverlap = 0;
  double pprNotInPredOrigOverlap = 0;
  double freqOrigOverlap = 0, svdPredOverlap = 0;
  double freqNotInPredOrigOverlap = 0;
  double predPPROrigOverlap = 0;
  double over_svdScores = 0, notOver_svdScores = 0;

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
    
    //get item scores ordered by predicted scores in decreasing order
    //auto headItems = getHeadItems(trainMat, 0.2);

    auto itemPredScoresPair = itemPredScores(fullModel, user, trainMat, 
        nItems, invalItems);
    auto itemOrigScoresPair = itemOrigScores(origModel, user, trainMat, 
        nItems, invalItems);
    auto itemSVDScoresPair = itemSVDScores(svdModel, user, trainMat, nItems, 
        invalItems);
    //auto itemPPRScoresPair = itemGraphItemScores(user, graphMat, trainMat, 
    //    0.01, nUsers, nItems, invalItems, true);

    auto itemFreqPair = itemFreqScores(fullModel, origModel, itemFreq,
        user, trainMat, nItems, invalItems);
    auto itemPredSVDScoresPair = prodScores(itemPredScoresPair, 
        itemSVDScoresPair);
    //auto itemPredPPRScoresPair = prodScores(itemPredScoresPair, 
    //    itemPPRScoresPair);
    auto svdItemPairsNotInPred = orderingDiff(itemPredScoresPair,
        itemSVDScoresPair, topBuckN);
    
    //auto pprItemPairsNotInPred = orderingDiff(itemPredScoresPair,
    //    itemPPRScoresPair, topBuckN);

    auto freqItemPairsNotInPred = orderingDiff(itemPredScoresPair,
        itemFreqPair, topBuckN);

    svdNotInPredOrigOverlap += compOrderingOverlap(itemOrigScoresPair,
        svdItemPairsNotInPred, topBuckN);
    //pprNotInPredOrigOverlap += compOrderingOverlap(itemOrigScoresPair,
    //    pprItemPairsNotInPred, topBuckN);
    freqNotInPredOrigOverlap += compOrderingOverlap(itemOrigScoresPair,
        freqItemPairsNotInPred, topBuckN);

    predOrigOverlap += compOrderingOverlap(itemOrigScoresPair, 
        itemPredScoresPair, topBuckN);
    svdOrigOverlap += compOrderingOverlap(itemOrigScoresPair,
        itemSVDScoresPair, topBuckN);

    auto origOverScores = compOrderingOverlapBScores(itemOrigScoresPair, 
        itemSVDScoresPair, topBuckN);
    if (-1 != origOverScores.first) {
      over_svdScores += origOverScores.first;
    }

    if (-1 != origOverScores.second) {
      notOver_svdScores += origOverScores.second;
    }

    predSVDOrigOverlap += compOrderingOverlap(itemOrigScoresPair,
        itemPredSVDScoresPair, topBuckN);
    freqOrigOverlap += compOrderingOverlap(itemOrigScoresPair,
        itemFreqPair, topBuckN);
    //pprOrigOverlap += compOrderingOverlap(itemOrigScoresPair,
    //    itemPPRScoresPair, topBuckN);
    //predPPROrigOverlap += compOrderingOverlap(itemOrigScoresPair,
    //    itemPredPPRScoresPair, topBuckN);
    svdPredOverlap += compOrderingOverlap(itemSVDScoresPair,
        itemPredScoresPair, topBuckN);

    //auto itemPredPPRScoresPair = prodScores(itemPredScoresPair,
    //    itemPPRScoresPair);

    auto itemScoresPair = itemOrigScoresPair;
    auto avgUserRating = origModel.estAvgRating(user, invalItems);
    int temp = predHighcount;
    
    int uPredLowCount = 0, uPredHighCount = 0, uLowcount = 0, uHighCount = 0;

    countInvRatings(origModel, fullModel, user, itemScoresPair, 
        avgUserRating, predLowcount, predHighcount, lowCount, highCount, 
        topBuckN);
    if (predHighcount != temp) {
      nHighPredUsers++;
    }

    //get itemsRMSE and itemsScore
    itemRMSEsOrdByItemScores(user, filtItems, fullModel, origModel, itemsRMSE, 
        itemsScore, itemScoresPair);
    
    //reset user specific vec to 0
    std::fill(uBucketNNZ.begin(), uBucketNNZ.end(), 0); 
    std::fill(uRMSEScores.begin(), uRMSEScores.end(), 0);
    updateBucketsArr(uRMSEScores, uBucketNNZ, itemsRMSE, nBuckets);

    //reset user specific vec to 0
    std::fill(uBucketNNZ.begin(), uBucketNNZ.end(), 0); 
    std::fill(uScores.begin(), uScores.end(), 0);
    updateBucketsArr(uScores, uBucketNNZ, itemsScore, nBuckets);
    
    //write and update aggregated buckets
    rmseOpFile << user << " ";
    for (int i = 0; i < nBuckets; i++) {
      bucketNNZ[i]  += uBucketNNZ[i];
      rmseScores[i] += uRMSEScores[i];
      rmseOpFile << sqrt(uRMSEScores[i]/uBucketNNZ[i]) << " ";
      scores[i]     += uScores[i];
      scoreOpFile << uScores[i]/uBucketNNZ[i] << " ";
    }
    rmseOpFile << std::endl;
    scoreOpFile << std::endl;

    if (sampUsers.size() % PROGU == 0) {
      std::cout << "Done... " << sampUsers.size() << " :" << std::endl;
    }

  }
  
  rmseOpFile.close();
  scoreOpFile.close();

  for (int i = 0; i < nBuckets; i++) {
    rmseScores[i] = sqrt(rmseScores[i]/bucketNNZ[i]);
    scores[i] = scores[i]/bucketNNZ[i];
  }
  std::string rmseScoreFName = prefix + "_rmseBuckets.txt";
  writeVector(rmseScores, rmseScoreFName.c_str());

  std::string avgScoreFName = prefix + "_avgScore.txt";
  writeVector(scores, avgScoreFName.c_str());

  std::cout << "\nbuck size: " << topBuckN 
    << " predHighCount: " << (float)predHighcount << "\nlowCount: " << lowCount 
    << "\nhighPredUsers: " << nHighPredUsers
    << "\nlow rated Items pred high pc: " << (float)predHighcount/lowCount 
    << "\nlow rated Items pc: "<< (float)lowCount/((float)topBuckN*sampUsers.size()) 
    << std::endl;
  
  std::cout << "\npredLoCount: " << (float)predLowcount << " highcount: "
    << highCount << " pc: " << (float)predLowcount/highCount  
    << "\nhigh rated items pred low pc: " << (float)predLowcount/highCount 
    << "\nhigh rated items pc: " << (float)highCount/((float)topBuckN*sampUsers.size())
    << std::endl;
  
  std::cout << "predOrigOverlap: " << predOrigOverlap/sampUsers.size() << std::endl;
  std::cout << "svdOrigOverlap: " << svdOrigOverlap/sampUsers.size() << std::endl;
  std::cout << "freqOrigOverlap: " << freqOrigOverlap/sampUsers.size() << std::endl;
  std::cout << "freqNotInPredOrigOverlap: " << freqNotInPredOrigOverlap/sampUsers.size() 
    << std::endl;
  //std::cout << "pprOrigOverlap: " << pprOrigOverlap/sampUsers.size() << std::endl;
  //std::cout << "predPPROrigOverlap: " << predPPROrigOverlap/sampUsers.size() << std::endl;

  //std::cout << "pprNotInPredOrigOverlap: " << pprNotInPredOrigOverlap/sampUsers.size() 
  //  << std::endl;
  std::cout << "svdPredOverlap: " << svdPredOverlap/sampUsers.size() << std::endl;
  std::cout << "svdNotInPredOrigOverlap: " << svdNotInPredOrigOverlap/sampUsers.size() 
    << std::endl;
  std::cout << "predSVDOrigOverlap: " << predSVDOrigOverlap/sampUsers.size() << std::endl;

  //compute average of above over all users
  std::cout << "svd overlap scores: " << over_svdScores << std::endl;
  std::cout << "svd NOT overlap sccores: " << notOver_svdScores << std::endl;
  
  over_svdScores = over_svdScores/sampUsers.size();
  notOver_svdScores = notOver_svdScores/sampUsers.size();
  
  std::cout << "svd overlap scores: " << over_svdScores << std::endl;
  std::cout << "svd NOT overlap sccores: " << notOver_svdScores << std::endl;

  /*
  //generate stats for sampled users
  std::vector<int> users;
  for (auto&& user: sampUsers) {
    users.push_back(user);
  }
  std::string statsFName = prefix + "_userStats.txt";
  getUserStats(users, trainMat, invalItems, statsFName.c_str());
  */  
}


std::vector<double> aggScoresInBuckets(std::vector<double> scores, 
    int nBuckets) {

  std::vector<double> bucketScores(nBuckets, 0);
  int nScoresPerBucket = scores.size()/nBuckets;

  if (nScoresPerBucket == 0) {
    std::cout << "scores size: " << scores.size() << std::endl;
  }

  for (int i = 0; i < nBuckets; i++) {
    double sum = 0;
    int count = 0;
    int startInd = -1, endInd = -1;
    
    startInd = i*nScoresPerBucket;
    if (i == nBuckets-1) {
      endInd = scores.size()-1;
    } else {
      endInd = (i+1)*nScoresPerBucket; 
    }

    for (int j = startInd; j <= endInd; j++) {
      sum += scores[j];
      count++;
    }
    
    bucketScores[i] = sum/count;
  }

  return bucketScores;
} 


void updateMisPredBins(std::vector<double>& misPredCountBins, 
    std::vector<double>& pprScoreBins, 
    std::vector<std::pair<int, double>>& origScorePairs, 
    std::vector<std::pair<int, double>>& predNotInTopPairs, int topBuckN,
    std::map<int, double>& pprScoreMap, int user) {
  
  std::unordered_set<int> notItems;
  for (auto&& pair: predNotInTopPairs) {
    notItems.insert(pair.first);
  }
  
  int nBins = origScorePairs.size()/topBuckN;
  int nItemsPerBin = topBuckN;
  int nFound = 0;
  for (int i = 0; i < nBins; i++) {
    int startInd = nItemsPerBin*i;
    int endInd   = nItemsPerBin*(i+1);
    if (i == nBins-1) {
      endInd = origScorePairs.size();
    }
    
    for (int j = startInd; j < endInd; j++) {
      int item = origScorePairs[j].first;
      if (notItems.find(item) != notItems.end()) {
        //found item
        misPredCountBins[i] += 1;
        pprScoreBins[i] += pprScoreMap[item];
        nFound++;
      }
    }
  }
  
  if (nFound != notItems.size()) {
    std::cerr << "All mispredicted items not found: " << nFound << " " 
      << notItems.size() << std::endl;
  }
}


void updateMisPredBins(std::vector<double>& misPredCountBins, 
    std::vector<double>& svdScoreBins, 
    std::vector<std::pair<int, double>>& origScorePairs, 
    std::vector<std::pair<int, double>>& predNotInTopPairs, int topBuckN,
    Model& svdModel, int user) {
  
  std::unordered_set<int> notItems;
  for (auto&& pair: predNotInTopPairs) {
    notItems.insert(pair.first);
  }
  int nBins = origScorePairs.size()/topBuckN; 
  int nItemsPerBin = topBuckN;
  int nFound = 0;
  for (int i = 0; i < nBins; i++) {
    int startInd = nItemsPerBin*i;
    int endInd   = nItemsPerBin*(i+1);
    if (i == nBins-1) {
      endInd = origScorePairs.size();
    }
    
    for (int j = startInd; j < endInd; j++) {
      int item = origScorePairs[j].first;
      if (notItems.find(item) != notItems.end()) {
        //found item
        misPredCountBins[i] += 1;
        svdScoreBins[i] += svdModel.estRating(user, item);
        nFound++;
      }
    }
  }
  
  if (nFound != notItems.size()) {
    std::cerr << "All mispredicted items not found: " << nFound << " " 
      << notItems.size() << std::endl;
  }
}


void updateMisPredFreqBins(std::vector<double>& misPredCountBins, 
    std::vector<double>& freqScoreBins, 
    std::vector<std::pair<int, double>>& origScorePairs, 
    std::vector<std::pair<int, double>>& predNotInTopPairs, int topBuckN,
    std::vector<double>& itemFreq, int user) {
  
  std::unordered_set<int> notItems;
  for (auto&& pair: predNotInTopPairs) {
    notItems.insert(pair.first);
  }
  int nBins = origScorePairs.size()/topBuckN; 
  int nItemsPerBin = topBuckN;
  int nFound = 0;
  for (int i = 0; i < nBins; i++) {
    int startInd = nItemsPerBin*i;
    int endInd   = nItemsPerBin*(i+1);
    if (i == nBins-1) {
      endInd = origScorePairs.size();
    } 
    
    for (int j = startInd; j < endInd; j++) {
      int item = origScorePairs[j].first;
      if (notItems.find(item) != notItems.end()) {
        //found item
        misPredCountBins[i] += 1;
        freqScoreBins[i] += itemFreq[item];
        nFound++;
      }
    }
  }
  
  if (nFound != notItems.size()) {
    std::cerr << "All mispredicted items not found: " << nFound << " " 
      << notItems.size() << std::endl;
  }
}


void predSampUsersRMSEProb2(gk_csr_t *trainMat, gk_csr_t *graphMat, 
    int nUsers, int nItems, 
    Model& origModel, Model& fullModel, Model& svdModel, 
    std::unordered_set<int>& invalUsers, std::unordered_set<int>& invalItems, 
    std::unordered_set<int>& filtItems, 
    int nSampUsers, int seed, std::string prefix,
    std::vector<double> alphas) {
  
  int nBuckets = 10;
  int nItemsPerBuck = nItems/nBuckets;

  std::vector<double> rmseGTScores(nBuckets, 0);
  std::vector<double> uRMSEGTScores(nBuckets, 0);
  
  std::vector<double> rmseFreqScores(nBuckets, 0);
  std::vector<double> uRMSEFreqScores(nBuckets, 0);
 
  std::vector<double> rmsePPRScores(nBuckets, 0);
  std::vector<double> uRMSEPPRScores(nBuckets, 0);

  std::vector<double> rmseSVDScores(nBuckets, 0);
  std::vector<double> uRMSESVDScores(nBuckets, 0);

  std::vector<double> scores(nBuckets, 0);
  std::vector<double> uScores(nBuckets, 0);
  
  std::vector<double> bucketNNZ(nBuckets, 0);
  std::vector<double> uBucketNNZ(nBuckets, 0);

  std::vector<double> itemsRMSE;
  std::vector<double> itemsScore;

  //initialize random engine
  std::mt19937 mt(seed);
  //user distribution to sample users
  std::uniform_int_distribution<int> uDist(0, nUsers-1);

  std::unordered_set<int> sampUsers;

  auto avgTrainRating = meanRating(trainMat);
  
  int predLowcount = 0, predHighcount = 0;
  int lowCount = 0, highCount = 0;
  
  auto rowColFreq = getRowColFreq(trainMat);
  auto userFreq = rowColFreq.first;
  auto itemFreq = rowColFreq.second;
  std::vector<std::pair<int, double>> itemFreqScoresPair;
  for (int i = 0; i < itemFreq.size(); i++) {
    itemFreqScoresPair.push_back(std::make_pair(i, itemFreq[i]));
  }

  auto itemAvgRatings = meanItemRating(trainMat);

  int topBuckN = 0.05*nItems;

  std::cout << "\ntopBuckN: " << topBuckN << std::endl;

  int nHighPredUsers = 0;

  std::vector<double> predSVDOrigOverlaps(alphas.size(), 0.0);
  std::vector<double> aggSVDScoresInOrig(10, 0);
  std::vector<double> aggSVDScoresNotInOrig(10, 0);
  std::vector<double> aggPredScoresInOrig(10, 0);

  std::vector<double> misPredSVDCountBins(20, 0);
  std::vector<double> misPredFreqCountBins(20, 0);
  std::vector<double> misPredPPRCountBins(20, 0);
  std::vector<double> svdScoreBins(20, 0);
  std::vector<double> freqScoreBins(20, 0);
  std::vector<double> pprScoreBins(20, 0);

  std::vector<double> misGTSVDCountBins(20, 0);
  std::vector<double> misGTSVDScoreBins(20, 0);
  std::vector<double> misGTOrigCountBins(20, 0);
  std::vector<double> misGTOrigScoreBins(20, 0);
  std::vector<double> misGTMFCountBins(20, 0);
  std::vector<double> misGTMFScoreBins(20, 0);
  std::vector<double> misGTFreqScoreBins(20, 0);
  std::vector<double> misGTFreqCountBins(20, 0);
  std::vector<double> misGTAvgTrainScoreBins(20, 0);
  std::vector<double> misGTAvgTrainCountBins(20, 0);
  std::vector<double> misGTPPRCountBins(20, 0);
  std::vector<double> misGTPPRScoreBins(20, 0);

  std::vector<double> svdOrderedPredBins(10, 0);
  std::vector<double> mfOrderedPredBins(10, 0);

  double predSVDAvgOverlap = 0;

  double predOrigOverlap = 0, svdOrigOverlap = 0;
  double svdPredOverlap = 0;
  double pprOrigOverlap = 0;
  
  double svdNotInPred = 0, svdInPred = 0;

  double svdOfPredInOrig = 0, svdOfPredNotInOrig = 0;
  double svdVarOfPredInOrig = 0, svdVarOfPredNotInOrig = 0;
  double svdOfMedPredInOrig = 0;
  double svdOfMaxPredInorig = 0, svdOfMinPredInOrig = 0;
  double svdOfTopPredInOrig = 0, svdOfBotPredInOrig = 0;
  
  double svdAboveAvgInOrig = 0;

  double pprOfPredInOrig = 0, pprOfPredNotInOrig = 0;
  double pprNotInPred = 0, pprInPred = 0;
  double predPPROrigOverlap = 0;
  double iterPredSVDOrigOverlap = 0;
  
  std::vector<double> svdScores;
  std::vector<double> predScores;

  std::ofstream predOpFile("mfPredScores.txt");
  std::ofstream svdOpFile("svdPredScores.txt");
  std::ofstream predInOrigSVDFile("predInOrigSVDScores.txt");
  std::ofstream predNotInOrigSVDFile("predNotInOrigSVDScores.txt");
  
  double avgTopSvd = 0, avgBottomSVD = 0;

  //while (sampUsers.size() < nSampUsers) {
  for (int user = 0; user < nUsers; user++) {
    
    //sample user
    //int user = uDist(mt);

    /*
    auto search = sampUsers.find(user);
    if (search != sampUsers.end()) {
      //already sampled user
      continue;
    }
    */

    //skip if user is invalid
    auto search = invalUsers.find(user);
    if (search != invalUsers.end()) {
      //found n skip
      continue;
    }

    //insert the sampled user
    sampUsers.insert(user);
    
    //get item scores ordered by predicted scores in decreasing order
    //auto headItems = getHeadItems(trainMat, 0.2);

    auto itemPredScoresPair = itemPredScores(fullModel, user, trainMat, 
        nItems, invalItems);
    auto itemOrigScoresPair = itemOrigScores(origModel, user, trainMat, 
        nItems, invalItems);
    auto itemSVDScoresPair = itemSVDScores(svdModel, user, trainMat, nItems, 
        invalItems);
   
    auto itemPredTopN = std::vector<std::pair<int, double>>(
        itemPredScoresPair.begin(), itemPredScoresPair.begin()+topBuckN);
    auto itemOrigTopN = std::vector<std::pair<int, double>>(
        itemOrigScoresPair.begin(), itemOrigScoresPair.begin()+topBuckN);

    for (int i = 0; i < topBuckN; i++) {
      int item = itemPredScoresPair[i].first;
      predOpFile << itemPredScoresPair[i].second << " ";
      svdOpFile << svdModel.estRating(user, item) << " ";
    }
    predOpFile << std::endl;
    svdOpFile << std::endl;

    std::vector<std::pair<int, double>> itemPPRScoresPair;

    double avgSVD = 0;
    for (auto it = itemPredScoresPair.begin(); 
        it != itemPredScoresPair.begin() + 1*topBuckN; it++) {
      int item = (*it).first;
      avgSVD += svdModel.estRating(user, item);
    }
    avgSVD = avgSVD/topBuckN;

    //std::cout << "Average SVD scores: " << avgSVD << std::endl;

    std::vector<std::pair<int, double>> aboveAvgSVDPairs;
    for (auto it = itemPredScoresPair.begin(); 
        it != itemPredScoresPair.begin() + 2*topBuckN; it++) {
      int item = (*it).first;
      double score = svdModel.estRating(user, item);
      if (score >= avgSVD) {
        aboveAvgSVDPairs.push_back(std::make_pair(item, score));
      }
    }
    //auto svdAboveAvgInOrigTop = orderingOverlap(itemOrigScoresPair,
    //    aboveAvgSVDPairs, topBuckN);
    svdAboveAvgInOrig += compOrderingOverlap2(itemOrigScoresPair, aboveAvgSVDPairs, 
        topBuckN); 

    auto predInOrigTop = orderingOverlap(itemOrigScoresPair, 
        itemPredScoresPair, topBuckN);
    double svdScore = 0;
    double maxSvd = 0, minSvd = 1;
    
    svdScores.clear();
    predScores.clear();

    for (auto&& pairs: predInOrigTop) {
      int item = pairs.first;
      double score = svdModel.estRating(user, item);
      predInOrigSVDFile << pairs.second << " " << score << " ";
      svdScores.push_back(score);
      predScores.push_back(pairs.second);
      svdScore += score;
      if (score > maxSvd) {
        maxSvd = score;
      }
      if (score < minSvd) {
        minSvd = score;
      }
    }
    predInOrigSVDFile << std::endl;

    svdOfMaxPredInorig += maxSvd;
    svdOfMinPredInOrig += minSvd;
   
    /*
    auto aggSVDScores = aggScoresInBuckets(svdScores, 10);
    for (int i = 0; i < 10; i++) {
      aggSVDScoresInOrig[i] += aggSVDScores[i];
    }

    auto aggPredScores = aggScoresInBuckets(predScores, 10);
    for (int i = 0; i < 10; i++) {
      aggPredScoresInOrig[i] += aggPredScores[i];
    }
    */

    if (predInOrigTop.size()) {
      svdScore = svdScore/predInOrigTop.size();
      svdOfMedPredInOrig += svdModel.estRating(user, 
          predInOrigTop[predInOrigTop.size()/2].first);
      svdOfTopPredInOrig += svdModel.estRating(user,
          predInOrigTop[0].first);
      svdOfBotPredInOrig += svdModel.estRating(user,
          predInOrigTop[predInOrigTop.size()-1].first);
    }
    svdOfPredInOrig += svdScore;

    double svdVarPredInOrig = 0;
    for (auto&& pair: predInOrigTop) {
      int item = pair.first;
      svdVarPredInOrig += std::pow(
          svdModel.estRating(user, item) - svdScore, 2);      
    }
    if (predInOrigTop.size()) {
      svdVarPredInOrig = svdVarPredInOrig/predInOrigTop.size();
    }
    svdVarOfPredInOrig += svdVarPredInOrig;


    auto predNotInOrig = orderingDiff(itemOrigScoresPair,
        itemPredScoresPair, topBuckN);
    svdScore = 0;
    svdScores.clear();
    for (auto&& pairs: predNotInOrig) {
      int item = pairs.first;
      double score = svdModel.estRating(user, item);
      predNotInOrigSVDFile <<  pairs.second << " " << score << " ";
      svdScore += score;
      svdScores.push_back(score);
    }
    predNotInOrigSVDFile << std::endl;
    if (predNotInOrig.size()) {
      svdScore = svdScore/predNotInOrig.size();
    }
    svdOfPredNotInOrig += svdScore;

    //updateMisPredBins(misPredSVDCountBins, svdScoreBins, itemOrigScoresPair, 
    //    predNotInOrig, topBuckN, svdModel, user);
    updateMisPredBins(misPredSVDCountBins, svdScoreBins, itemOrigScoresPair, 
        itemPredTopN, topBuckN, svdModel, user);
    
    //updateMisPredFreqBins(misPredFreqCountBins, freqScoreBins, itemOrigScoresPair,
    //    predNotInOrig, topBuckN, itemFreq, user);
    updateMisPredFreqBins(misPredFreqCountBins, freqScoreBins, itemOrigScoresPair,
        itemPredTopN, topBuckN, itemFreq, user);

    auto gtNotInPred = orderingDiff(itemPredScoresPair, itemOrigScoresPair,
        topBuckN);
    //updateMisPredBins(misGTSVDCountBins, misGTSVDScoreBins, itemPredScoresPair,
    //    gtNotInPred, topBuckN, svdModel, user);
    updateMisPredBins(misGTSVDCountBins, misGTSVDScoreBins, itemPredScoresPair,
        itemOrigTopN, topBuckN, svdModel, user);
    
    //updateMisPredFreqBins(misGTFreqCountBins, misGTFreqScoreBins, itemPredScoresPair,
    //    gtNotInPred, topBuckN, itemFreq, user);
    updateMisPredFreqBins(misGTFreqCountBins, misGTFreqScoreBins, itemPredScoresPair,
        itemOrigTopN, topBuckN, itemFreq, user);

    //updateMisPredFreqBins(misGTAvgTrainCountBins, misGTAvgTrainScoreBins, 
    //    itemPredScoresPair, gtNotInPred, topBuckN, itemAvgRatings, user);
    updateMisPredFreqBins(misGTAvgTrainCountBins, misGTAvgTrainScoreBins, 
        itemPredScoresPair, itemOrigTopN, topBuckN, itemAvgRatings, user);
    
    //updateMisPredBins(misGTOrigCountBins, misGTOrigScoreBins, itemPredScoresPair,
    //    gtNotInPred, topBuckN, origModel, user);
    updateMisPredBins(misGTOrigCountBins, misGTOrigScoreBins, itemPredScoresPair,
        itemOrigTopN, topBuckN, origModel, user);

    //updateMisPredBins(misGTMFCountBins, misGTMFScoreBins, itemPredScoresPair,
    //    gtNotInPred, topBuckN, fullModel, user);
    updateMisPredBins(misGTMFCountBins, misGTMFScoreBins, itemPredScoresPair,
        itemOrigTopN, topBuckN, fullModel, user);
   
    /*
    if (svdScores.size() >= 10) {
     aggSVDScores = aggScoresInBuckets(svdScores, 10);
      for (int i = 0; i < 10; i++) {
        aggSVDScoresNotInOrig[i] += aggSVDScores[i];
      }
    }
    */
    double svdVarPredNotInOrig = 0;
    for (auto&& pair: predNotInOrig) {
      int item = pair.first;
      svdVarPredNotInOrig += std::pow(
          svdModel.estRating(user, item) - svdScore, 2);
    }
    if (predNotInOrig.size()) {
      svdVarPredNotInOrig = svdVarPredNotInOrig/predNotInOrig.size();
    }
    svdVarOfPredNotInOrig += svdVarPredNotInOrig;

    if (NULL != graphMat) {
      itemPPRScoresPair = itemGraphItemScores(user, graphMat, trainMat, 
        0.01, nUsers, nItems, invalItems, false);
      std::map<int, double> pprMap;
      for (auto&& itemScore: itemPPRScoresPair) {
        if (isnan(itemScore.second)) {
          std::cerr << "Found NaN: " << itemScore.first << " " 
            << itemScore.second << std::endl;
        } else {
          pprMap[itemScore.first] = itemScore.second;
        }
      }

      //updateMisPredBins(misPredPPRCountBins, pprScoreBins, itemOrigScoresPair,
      //    predNotInOrig, topBuckN, pprMap, user);
      updateMisPredBins(misPredPPRCountBins, pprScoreBins, itemOrigScoresPair,
          itemPredTopN, topBuckN, pprMap, user);

      //updateMisPredBins(misGTPPRCountBins, misGTPPRScoreBins, itemPredScoresPair,
      //    gtNotInPred, topBuckN, pprMap, user);
      updateMisPredBins(misGTPPRCountBins, misGTPPRScoreBins, itemPredScoresPair,
          itemOrigTopN, topBuckN, pprMap, user);

      double pprScore = 0;
      for (auto&& pairs: predInOrigTop) {
        int item = pairs.first;
        pprScore += pprMap[item];
      }
      if (predInOrigTop.size()) {
        pprScore = pprScore/predInOrigTop.size();
      }
      pprOfPredInOrig += pprScore;
      
      pprScore = 0;
      for (auto&& pairs: predNotInOrig) {
        int item = pairs.first;
        pprScore += pprMap[item];
      }
      if (predNotInOrig.size()) {
        pprScore = pprScore/predNotInOrig.size();
      }
      pprOfPredNotInOrig += pprScore;
      
      //updateMisPredBins(misPredCountBins, svdScoreBins, itemOrigScoresPair, 
      //  predNotInOrig, 20, svdModel, user);
    
      pprOrigOverlap += compOrderingOverlap(itemOrigScoresPair,
          itemPPRScoresPair, topBuckN);
      
      /*
      auto topPPRPairsFrmMF = iterSort(itemPredScoresPair, itemPPRScoresPair,
          topBuckN);

      compOverlapPcBins(topPPRPairsFrmMF, itemOrigScoresPair, svdOrderedPredBins,
          topBuckN); 

      compOverlapPcBins(itemPredScoresPair, itemOrigScoresPair, mfOrderedPredBins, 
          topBuckN);
      */
    }

    auto topSVDPairsFrmMF = iterSort(itemPredScoresPair, itemSVDScoresPair,
        topBuckN*2);

    iterPredSVDOrigOverlap += compOrderingOverlap(itemOrigScoresPair, 
        topSVDPairsFrmMF, topBuckN); 
    
    topSVDPairsFrmMF = iterSort(itemPredScoresPair, itemSVDScoresPair, topBuckN);
    
    compOverlapPcBins(topSVDPairsFrmMF, itemOrigScoresPair, svdOrderedPredBins,
        topBuckN); 

    compOverlapPcBins(itemPredScoresPair, itemOrigScoresPair, mfOrderedPredBins, 
        topBuckN);
    
    double avgTop = 0;
    for (int l = 0; l < topBuckN/2; l++) {
      avgTop += topSVDPairsFrmMF[l].second;
    }
    avgTopSvd += avgTop/(topBuckN/2);

    double avgBottom = 0;
    for (int l = topBuckN/2; l < topBuckN; l++) {
      avgBottom += topSVDPairsFrmMF[l].second;
    }

    avgBottomSVD += avgBottom/(topBuckN/2);

    /*
    compOverlapPcLaterBins(itemPredScoresPair, itemSVDScoresPair, 
        itemOrigScoresPair, 
        mfOrderedPredBins, svdOrderedPredBins, topBuckN);
    */

    //auto pprItemPairsNotInPred = orderingDiff(itemPredScoresPair,
    //    itemPPRScoresPair, topBuckN);

    predOrigOverlap += compOrderingOverlap(itemOrigScoresPair, 
        itemPredScoresPair, topBuckN);
    svdOrigOverlap += compOrderingOverlap(itemOrigScoresPair,
        itemSVDScoresPair, topBuckN);
   
    //auto scalePredScoresPair = zero1Scale(itemPredScoresPair);
    //auto scaleSVDScoresPair  = zero1Scale(itemSVDScoresPair);
    
    //auto predSVDAvgScoresPair = avgScores(scalePredScoresPair, 
    //    scaleSVDScoresPair);

    auto prodSVDAvgScoresPair = prodScores(itemPredScoresPair, 
        itemSVDScoresPair);
    
    predSVDAvgOverlap += compOrderingOverlap(itemOrigScoresPair, 
        prodSVDAvgScoresPair, topBuckN);
    
    svdPredOverlap += compOrderingOverlap(itemSVDScoresPair,
        itemPredScoresPair, topBuckN);

    //auto itemPredPPRScoresPair = prodScores(itemPredScoresPair,
    //    itemPPRScoresPair);
    //get itemsRMSE and itemsScore

    itemRMSEsOrdByItemScores(user, filtItems, fullModel, origModel, itemsRMSE, 
        itemsScore, itemSVDScoresPair);
    
    //reset user specific vec to 0
    std::fill(uBucketNNZ.begin(), uBucketNNZ.end(), 0); 
    std::fill(uRMSESVDScores.begin(), uRMSESVDScores.end(), 0);
    updateBucketsArr(uRMSESVDScores, uBucketNNZ, itemsRMSE, nBuckets);

    //get itemsRMSE and itemsScore
    itemRMSEsOrdByItemScores(user, filtItems, fullModel, origModel, itemsRMSE, 
        itemsScore, itemFreqScoresPair);
    
    //reset user specific vec to 0
    std::fill(uBucketNNZ.begin(), uBucketNNZ.end(), 0); 
    std::fill(uRMSEFreqScores.begin(), uRMSEFreqScores.end(), 0);
    updateBucketsArr(uRMSEFreqScores, uBucketNNZ, itemsRMSE, nBuckets);

    if (NULL != graphMat) {
      //get itemsRMSE and itemsScore
      itemRMSEsOrdByItemScores(user, filtItems, fullModel, origModel, itemsRMSE, 
          itemsScore, itemPPRScoresPair);
      
      //reset user specific vec to 0
      std::fill(uBucketNNZ.begin(), uBucketNNZ.end(), 0); 
      std::fill(uRMSEPPRScores.begin(), uRMSEPPRScores.end(), 0);
      updateBucketsArr(uRMSEPPRScores, uBucketNNZ, itemsRMSE, nBuckets);
    }

    //reset user specific vec to 0
    std::fill(uBucketNNZ.begin(), uBucketNNZ.end(), 0); 
    std::fill(uScores.begin(), uScores.end(), 0);
    updateBucketsArr(uScores, uBucketNNZ, itemsScore, nBuckets);
 
    //get itemsRMSE and itemsScore
    itemRMSEsOrdByItemScores(user, filtItems, fullModel, origModel, itemsRMSE, 
        itemsScore, itemOrigScoresPair);
    
    //reset user specific vec to 0
    std::fill(uBucketNNZ.begin(), uBucketNNZ.end(), 0); 
    std::fill(uRMSEGTScores.begin(), uRMSEGTScores.end(), 0);
    updateBucketsArr(uRMSEGTScores, uBucketNNZ, itemsRMSE, nBuckets);
    
    //reset user specific vec to 0
    std::fill(uBucketNNZ.begin(), uBucketNNZ.end(), 0); 
    std::fill(uScores.begin(), uScores.end(), 0);
    updateBucketsArr(uScores, uBucketNNZ, itemsScore, nBuckets);
   
    //write and update aggregated buckets
    for (int i = 0; i < nBuckets; i++) {
      bucketNNZ[i]     += uBucketNNZ[i];
      rmseGTScores[i]  += uRMSEGTScores[i];
      rmseSVDScores[i] += uRMSESVDScores[i];
      rmseFreqScores[i] += uRMSEFreqScores[i];
      rmsePPRScores[i] += uRMSEPPRScores[i];
      scores[i]        += uScores[i];
    }

    if (sampUsers.size() % PROGU == 0) {
      std::cout << "Done... " << sampUsers.size() << " :" << std::endl;
    }

  }

  predOpFile.close();
  svdOpFile.close();
  predInOrigSVDFile.close();
  predNotInOrigSVDFile.close();

  for (int i = 0; i < nBuckets; i++) {
    rmseGTScores[i] = sqrt(rmseGTScores[i]/bucketNNZ[i]);
    rmseSVDScores[i] = sqrt(rmseSVDScores[i]/bucketNNZ[i]);
    rmsePPRScores[i] = sqrt(rmsePPRScores[i]/bucketNNZ[i]);
    rmseFreqScores[i] = sqrt(rmseFreqScores[i]/bucketNNZ[i]);
    scores[i] = scores[i]/bucketNNZ[i];
  }

  std::cout << "GT RMSE buckets: ";
  dispVector(rmseGTScores); 
  std::cout << std::endl;

  std::cout << "SVD RMSE buckets: ";
  dispVector(rmseSVDScores);
  std::cout << std::endl;

  std::cout << "Freq RMSE buckets: ";
  dispVector(rmseFreqScores);
  std::cout << std::endl;

  std::cout << "PPR RMSE buckets: ";
  dispVector(rmsePPRScores);
  std::cout << std::endl;

  std::cout << "GT Score buckets: ";
  dispVector(scores);
  std::cout << std::endl;
  
  //std::string rmseScoreFName = prefix + "_rmseBuckets.txt";
  //writeVector(rmseScores, rmseScoreFName.c_str());

  //std::string avgScoreFName = prefix + "_avgScore.txt";
  //writeVector(scores, avgScoreFName.c_str());
  
  std::cout << "No. sample users: " << sampUsers.size() << std::endl;
  std::cout << "predOrigOverlap: " << predOrigOverlap/sampUsers.size() << std::endl;
  std::cout << "svdOrigOverlap: " << svdOrigOverlap/sampUsers.size() << std::endl;
  std::cout << "svdInPred: " << svdInPred/sampUsers.size() << std::endl;
  std::cout << "svdNotInPred: " << svdNotInPred/sampUsers.size() << std::endl;
  
  std::cout << "pprOrigOverlap: " << pprOrigOverlap/sampUsers.size() << std::endl;
  std::cout << "pprofPredInOrig: " << pprOfPredInOrig/sampUsers.size() << std::endl;
  std::cout << "pprOfPredNotInOrig: " << pprOfPredNotInOrig/sampUsers.size() << std::endl;

  std::cout << "svdPredOverlap: " << svdPredOverlap/sampUsers.size() << std::endl;
  
  std::cout << "svdOfPredInOrig: " << svdOfPredInOrig/sampUsers.size()
    << " svdOfPredNotInOrig: " << svdOfPredNotInOrig/sampUsers.size() << std::endl;
  
  std::cout << "svdVarOfPredInOrig: " << svdVarOfPredInOrig/sampUsers.size() 
    << " svdVarOfPredNotInOrig: " << svdVarOfPredNotInOrig/sampUsers.size() << std::endl;
 
  std::cout << "svdAboveAvgInOrig: " << svdAboveAvgInOrig/sampUsers.size() << std::endl; 

  std::cout << "svdOfMaxPredInorig: " << svdOfMaxPredInorig/sampUsers.size() << std::endl;
  std::cout << "svdOfMinPredInOrig: " << svdOfMinPredInOrig/sampUsers.size() << std::endl;
  std::cout << "svdOfTopPredInOrig: " << svdOfTopPredInOrig/sampUsers.size() << std::endl;
  std::cout << "svdOfMedPredInOrig: " << svdOfMedPredInOrig/sampUsers.size() << std::endl; 
  std::cout << "svdOfBotPredInOrig: " << svdOfBotPredInOrig/sampUsers.size() << std::endl;

  /* 
  std::cout << "aggregated Pred scores in orig: " <<  std::endl;
  for (auto&& score: aggPredScoresInOrig) {
    std::cout << score/sampUsers.size() << " ";
  }
  std::cout << std::endl;

  std::cout << "aggregated SVD scores of pred in orig: " << std::endl;
  for (auto&& score: aggSVDScoresInOrig) {
    std::cout << score/sampUsers.size() << " ";
  }
  std::cout << std::endl;

  std::cout << "aggregated SVD scores of pred not in orig: " << std::endl;
  for (auto&& score: aggSVDScoresNotInOrig) {
    std::cout << score/sampUsers.size() << " ";
  }
  std::cout << std::endl;
  */

  std::cout << "pprOfPredInOrig: " << pprOfPredInOrig/sampUsers.size()
    << " pprOfPredNotInOrig: " << pprOfPredNotInOrig/sampUsers.size() << std::endl;
  
  std::cout << "iterPredSVDOrigOverlap: " 
    << iterPredSVDOrigOverlap/sampUsers.size() << std::endl;

  std::cout << "predSVDAvgOverlap: " << predSVDAvgOverlap/sampUsers.size() << std::endl;

  int sumSVDBins = 0;
  int sumPPRBins = 0;
  int sumGTSVDBins = 0;
  int sumGTPPRBins = 0;
  for (int i = 0; i < 20; i++) {
    sumSVDBins += misPredSVDCountBins[i];
    sumPPRBins += misPredPPRCountBins[i];

    sumGTSVDBins += misGTSVDCountBins[i];
    sumGTPPRBins += misGTPPRCountBins[i];
  }

  std::cout << "Total SVD misPred: " << sumSVDBins << std::endl;
 
  std::cout << "Mispred SVD %: ";
  for (int i = 0; i < 20; i++) {
    std::cout << misPredSVDCountBins[i]/sumSVDBins << ",";
  }
  std::cout << std::endl;

  std::cout << "Mispred avg SVD: ";
  for (int i = 0; i < 20; i++) {
    std::cout << svdScoreBins[i]/misPredSVDCountBins[i] << ",";
  }
  std::cout << std::endl;
  
  std::cout << "Mispred avg Freq: ";
  for (int i = 0; i < 20; i++) {
    std::cout << freqScoreBins[i]/misPredFreqCountBins[i] << ",";
  }

  std::cout << std::endl;
  std::cout << "Total PPR misPred: " << sumPPRBins << std::endl;
  std::cout << "Mispred PPR %: ";
  for (int i = 0; i < 20; i++) {
    std::cout << misPredPPRCountBins[i]/sumPPRBins << ","; 
  }
  std::cout << std::endl;

  std::cout << "Mispred avg PPR: ";
  for (int i = 0; i < 20; i++) {
    std::cout << pprScoreBins[i]/misPredPPRCountBins[i] << ","; 
  }
  std::cout << std::endl;
 
  std::cout << "Total SVD misGT: " << sumGTSVDBins << std::endl;
  std::cout << "MisGT SVD %: ";
  for (int i = 0; i < 20; i++) {
    std::cout << misGTSVDCountBins[i]/sumGTSVDBins << ",";
  }
  std::cout << std::endl;

  std::cout << "MisGT avg SVD: ";
  for (int i =0; i < 20; i++) {
    std::cout << misGTSVDScoreBins[i]/misGTSVDCountBins[i] << ",";
  }
  std::cout << std::endl;

  std::cout << "MisGT avg Freq: ";
  for (int i =0; i < 20; i++) {
    std::cout << misGTFreqScoreBins[i]/misGTFreqCountBins[i] << ",";
  }
  std::cout << std::endl;

  std::cout << "MisGT avg AllRatings: ";
  for (int i =0; i < 20; i++) {
    std::cout << misGTAvgTrainScoreBins[i]/misGTAvgTrainCountBins[i] << ",";
  }
  std::cout << std::endl;

  std::cout << "MisGT avg GT: ";
  for (int i =0; i < 20; i++) {
    std::cout << misGTOrigScoreBins[i]/misGTOrigCountBins[i] << ",";
  }
  std::cout << std::endl;

  std::cout << "MisGT avg Pred: ";
  for (int i =0; i < 20; i++) {
    std::cout << misGTMFScoreBins[i]/misGTMFCountBins[i] << ",";
  }
  std::cout << std::endl;

  std::cout << std::endl;
  std::cout << "Total PPR misGT: " << sumGTSVDBins << std::endl;
  std::cout << "MisGT PPR %: ";
  for (int i = 0; i < 20; i++) {
    std::cout << misGTPPRCountBins[i]/sumGTPPRBins << ",";
  }
  std::cout << std::endl;

  std::cout << "MisGT avg PPR: ";
  for (int i =0; i < 20; i++) {
    std::cout << misGTPPRScoreBins[i]/misGTPPRCountBins[i] << ",";
  }
  std::cout << std::endl;

  std::cout << "svdOrderedPredBins: ";
  for (auto&& pc: svdOrderedPredBins) {
    std::cout << pc/sampUsers.size() << " ";
  }
  std::cout<<std::endl;

  std::cout << "mfOrderedPredBins: ";
  for (auto&& pc: mfOrderedPredBins) {
    std::cout << pc/sampUsers.size() << " ";
  }
  std::cout<<std::endl;

  std::cout << "avgTopSvd: " << avgTopSvd << std::endl;
  std::cout << "avgBottomSVD: " << avgBottomSVD << std::endl;

  /*
  int k = 0;
  std::cout << "svdPredOrigOverlaps: " << std::endl;
  for (auto&& alpha: alphas) {
    std::cout << "alpha: " << alpha << " " 
      << predSVDOrigOverlaps[k++]/sampUsers.size() << std::endl; 
  }
  */

  /*
  //generate stats for sampled users
  std::vector<int> users;
  for (auto&& user: sampUsers) {
    users.push_back(user);
  }
  std::string statsFName = prefix + "_userStats.txt";
  getUserStats(users, trainMat, invalItems, statsFName.c_str());
  */  
}


void predSampUsersRMSEProbPar(const Data& data, 
    int nUsers, int nItems, 
    Model& origModel, Model& fullModel, Model& svdModel, 
    std::unordered_set<int>& invalUsers, std::unordered_set<int>& invalItems, 
    std::unordered_set<int>& filtItems, 
    int nSampUsers, int seed, std::string prefix) {
  
  int nBuckets = 10;
  int nItemsPerBuck = nItems/nBuckets;

  gk_csr_t* trainMat = data.trainMat;
  gk_csr_t* graphMat = data.graphMat;

  std::vector<double> g_rmseGTScores(nBuckets, 0);
  std::vector<double> g_rmseFreqScores(nBuckets, 0);
  std::vector<double> g_rmsePPRScores(nBuckets, 0);
  std::vector<double> g_rmseSVDScores(nBuckets, 0);
  std::vector<double> g_rmseOPTScores(nBuckets, 0);
  std::vector<double> g_scores(nBuckets, 0);
  std::vector<double> g_bucketNNZ(nBuckets, 0);

  //initialize random engine
  std::mt19937 mt(seed);
  //user distribution to sample users
  std::uniform_int_distribution<int> uDist(0, nUsers-1);

  std::unordered_set<int> sampUsers;

  auto avgTrainRating = meanRating(trainMat);
  
  int predLowcount = 0, predHighcount = 0;
  int lowCount = 0, highCount = 0;
  
  auto rowColFreq = getRowColFreq(trainMat);
  auto userFreq = rowColFreq.first;
  auto itemFreq = rowColFreq.second;
  std::vector<std::pair<int, double>> itemFreqScoresPair;
  for (int i = 0; i < itemFreq.size(); i++) {
    itemFreqScoresPair.push_back(std::make_pair(i, itemFreq[i]));
  }
  //sort item frequency in decreasing order
  std::sort(itemFreqScoresPair.begin(), itemFreqScoresPair.end(), descComp);
  
  auto itemAvgRatings = meanItemRating(trainMat);

  int topBuckN = 0.05*nItems;

  std::cout << "\ntopBuckN: " << topBuckN << std::endl;

  std::vector<double> g_misPredSVDCountBins(20, 0);
  std::vector<double> g_misPredFreqCountBins(20, 0);
  std::vector<double> g_misPredPPRCountBins(20, 0);
 
  std::vector<double> g_svdScoreBins(20, 0);
  std::vector<double> g_freqScoreBins(20, 0);
  std::vector<double> g_pprScoreBins(20, 0);

  std::vector<double> g_misGTSVDCountBins(20, 0);
  std::vector<double> g_misGTSVDScoreBins(20, 0);
  std::vector<double> g_misGTOrigCountBins(20, 0);
  std::vector<double> g_misGTOrigScoreBins(20, 0);
  
  std::vector<double> g_misGTMFCountBins(20, 0);
  std::vector<double> g_misGTMFScoreBins(20, 0);
  std::vector<double> g_misGTFreqScoreBins(20, 0);
  std::vector<double> g_misGTFreqCountBins(20, 0);
  
  std::vector<double> g_misGTAvgTrainScoreBins(20, 0);
  std::vector<double> g_misGTAvgTrainCountBins(20, 0);
  std::vector<double> g_misGTPPRCountBins(20, 0);
  std::vector<double> g_misGTPPRScoreBins(20, 0);

  double predOrigOverlap = 0, svdOrigOverlap = 0;
  double svdPredOverlap = 0;
  double pprOrigOverlap = 0;
  double freqOrigOverlap = 0;
  
  double freqNOTinSVDPc = 0;
  double svdNOTinFreqPc = 0;
  double freqNOTinPPRPc = 0;
  double pprNOTinFreqPc = 0;
  double svdNOTinPPRPc = 0;
  double pprNOTinSVDPc = 0;

  double freqSVDTopOvrlap = 0;
  double pprFreqTopOvrlap = 0;
  double pprSVDTopOvrlap = 0;
  double freqTopRMSE = 0;
  double svdTopRMSE = 0;
  double pprTopRMSE = 0;

  double svdNotInPred = 0, svdInPred = 0;

  double svdOfPredInOrig = 0, svdOfPredNotInOrig = 0;
  double svdVarOfPredInOrig = 0, svdVarOfPredNotInOrig = 0;
  double svdOfMedPredInOrig = 0;
  double svdOfMaxPredInorig = 0, svdOfMinPredInOrig = 0;
  double svdOfTopPredInOrig = 0, svdOfBotPredInOrig = 0;
  
  double svdAboveAvgInOrig = 0;

  double pprOfPredInOrig = 0, pprOfPredNotInOrig = 0;
  double pprNotInPred = 0, pprInPred = 0;
  double predPPROrigOverlap = 0;
  double iterPredSVDOrigOverlap = 0;
  
  int totalSampUsers;
  
  if (NULL != graphMat) {
    while (sampUsers.size() < nSampUsers) {
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
    } 
  } else {
    for (int user = 0; user < nUsers; user++) {
      //skip if user is invalid
      auto search = invalUsers.find(user);
      if (search != invalUsers.end()) {
        //found n skip
        continue;
      }
      //insert the sampled user
      sampUsers.insert(user);
    } 
  }
  
  int sampUsersSz = sampUsers.size();
  std::vector<int> sampUsersVec = std::vector<int>(sampUsers.begin(), 
      sampUsers.end());
  
  std::cout << "nItemsPerBuck: " << nItemsPerBuck << std::endl;

#pragma omp parallel
{
  std::vector<double> misPredSVDCountBins(20, 0);
  std::vector<double> misPredFreqCountBins(20, 0);
  std::vector<double> misPredPPRCountBins(20, 0);
  std::vector<double> svdScoreBins(20, 0);
  std::vector<double> freqScoreBins(20, 0);
  std::vector<double> pprScoreBins(20, 0);

  std::vector<double> misGTSVDCountBins(20, 0);
  std::vector<double> misGTSVDScoreBins(20, 0);
  std::vector<double> misGTOrigCountBins(20, 0);
  std::vector<double> misGTOrigScoreBins(20, 0);
  std::vector<double> misGTMFCountBins(20, 0);
  std::vector<double> misGTMFScoreBins(20, 0);
  std::vector<double> misGTFreqScoreBins(20, 0);
  std::vector<double> misGTFreqCountBins(20, 0);
  std::vector<double> misGTAvgTrainScoreBins(20, 0);
  std::vector<double> misGTAvgTrainCountBins(20, 0);
  std::vector<double> misGTPPRCountBins(20, 0);
  std::vector<double> misGTPPRScoreBins(20, 0);

  std::vector<double> rmseGTScores(nBuckets, 0);
  std::vector<double> rmseOPTScores(nBuckets, 0);
  std::vector<double> rmseFreqScores(nBuckets, 0);
  std::vector<double> rmsePPRScores(nBuckets, 0);
  std::vector<double> rmseSVDScores(nBuckets, 0);
  std::vector<double> scores(nBuckets, 0);
  std::vector<double> bucketNNZ(nBuckets, 0);

#pragma omp for reduction(+ : freqTopRMSE, svdTopRMSE, pprTopRMSE, freqSVDTopOvrlap, pprFreqTopOvrlap, pprSVDTopOvrlap, freqNOTinSVDPc, svdNOTinFreqPc, freqNOTinPPRPc, pprNOTinFreqPc, svdNOTinPPRPc, pprNOTinSVDPc, freqOrigOverlap, predOrigOverlap, svdOrigOverlap, svdPredOverlap, pprOrigOverlap, svdNotInPred, svdInPred, svdOfPredInOrig, svdOfPredNotInOrig, svdVarOfPredInOrig, svdVarOfPredNotInOrig, svdOfMedPredInOrig, svdOfMaxPredInorig, svdOfMinPredInOrig, svdOfTopPredInOrig, svdOfBotPredInOrig, svdAboveAvgInOrig, pprOfPredInOrig, pprOfPredNotInOrig, pprNotInPred, pprInPred, predPPROrigOverlap, iterPredSVDOrigOverlap, totalSampUsers)  
  for (int uInd=0; uInd < sampUsersSz; uInd++) {
    int user = sampUsersVec[uInd];

    totalSampUsers++;

    std::vector<double> uRMSEGTScores(nBuckets, 0);
    std::vector<double> uRMSEOPTScores(nBuckets, 0);
    std::vector<double> uRMSEFreqScores(nBuckets, 0);
    std::vector<double> uRMSEPPRScores(nBuckets, 0);
    std::vector<double> uRMSESVDScores(nBuckets, 0);
    std::vector<double> uScores(nBuckets, 0);
    std::vector<double> uBucketNNZ(nBuckets, 0);
    std::vector<double> itemsRMSE;
    std::vector<double> itemsScore;

    //get item scores ordered by predicted scores in decreasing order
    //auto headItems = getHeadItems(trainMat, 0.2);

    auto itemPredScoresPair = itemPredScores(fullModel, user, trainMat, 
        nItems, invalItems);
    auto itemOrigScoresPair = itemOrigScores(origModel, user, trainMat, 
        nItems, invalItems);
    auto itemSVDScoresPair = itemSVDScores(svdModel, user, trainMat, nItems, 
        invalItems);
   
    auto itemPredTopN = std::vector<std::pair<int, double>>(
        itemPredScoresPair.begin(), itemPredScoresPair.begin()+topBuckN);
    auto itemOrigTopN = std::vector<std::pair<int, double>>(
        itemOrigScoresPair.begin(), itemOrigScoresPair.begin()+topBuckN);

    std::vector<std::pair<int, double>> itemPPRScoresPair;

    auto predInOrigTop = orderingOverlap(itemOrigScoresPair, 
        itemPredScoresPair, topBuckN);
    double svdScore = 0;
    double maxSvd = 0, minSvd = 1;
    
    std::vector<double> svdScores;
    std::vector<double> predScores;
    for (auto&& pairs: predInOrigTop) {
      int item = pairs.first;
      double score = svdModel.estRating(user, item);
      svdScores.push_back(score);
      predScores.push_back(pairs.second);
      svdScore += score;
      if (score > maxSvd) {
        maxSvd = score;
      }
      if (score < minSvd) {
        minSvd = score;
      }
    }

    svdOfMaxPredInorig += maxSvd;
    svdOfMinPredInOrig += minSvd;

    if (predInOrigTop.size()) {
      svdScore = svdScore/predInOrigTop.size();
      svdOfMedPredInOrig += svdModel.estRating(user, 
          predInOrigTop[predInOrigTop.size()/2].first);
      svdOfTopPredInOrig += svdModel.estRating(user,
          predInOrigTop[0].first);
      svdOfBotPredInOrig += svdModel.estRating(user,
          predInOrigTop[predInOrigTop.size()-1].first);
    }
    svdOfPredInOrig += svdScore;

    double svdVarPredInOrig = 0;
    for (auto&& pair: predInOrigTop) {
      int item = pair.first;
      svdVarPredInOrig += std::pow(
          svdModel.estRating(user, item) - svdScore, 2);      
    }
    if (predInOrigTop.size()) {
      svdVarPredInOrig = svdVarPredInOrig/predInOrigTop.size();
    }
    svdVarOfPredInOrig += svdVarPredInOrig;


    auto predNotInOrig = orderingDiff(itemOrigScoresPair,
        itemPredScoresPair, topBuckN);
    svdScore = 0;
    svdScores.clear();
    for (auto&& pairs: predNotInOrig) {
      int item = pairs.first;
      double score = svdModel.estRating(user, item);
      svdScore += score;
      svdScores.push_back(score);
    }
    if (predNotInOrig.size()) {
      svdScore = svdScore/predNotInOrig.size();
    }
    svdOfPredNotInOrig += svdScore;

    updateMisPredBins(misPredSVDCountBins, svdScoreBins, itemOrigScoresPair, 
        itemPredTopN, topBuckN, svdModel, user);
    
    updateMisPredFreqBins(misPredFreqCountBins, freqScoreBins, itemOrigScoresPair,
        itemPredTopN, topBuckN, itemFreq, user);

    auto gtNotInPred = orderingDiff(itemPredScoresPair, itemOrigScoresPair,
        topBuckN);
    updateMisPredBins(misGTSVDCountBins, misGTSVDScoreBins, itemPredScoresPair,
        itemOrigTopN, topBuckN, svdModel, user);
    
    updateMisPredFreqBins(misGTFreqCountBins, misGTFreqScoreBins, itemPredScoresPair,
        itemOrigTopN, topBuckN, itemFreq, user);

    updateMisPredFreqBins(misGTAvgTrainCountBins, misGTAvgTrainScoreBins, 
        itemPredScoresPair, itemOrigTopN, topBuckN, itemAvgRatings, user);
    
    updateMisPredBins(misGTOrigCountBins, misGTOrigScoreBins, itemPredScoresPair,
        itemOrigTopN, topBuckN, origModel, user);

    updateMisPredBins(misGTMFCountBins, misGTMFScoreBins, itemPredScoresPair,
        itemOrigTopN, topBuckN, fullModel, user);
   
    double svdVarPredNotInOrig = 0;
    for (auto&& pair: predNotInOrig) {
      int item = pair.first;
      svdVarPredNotInOrig += std::pow(
          svdModel.estRating(user, item) - svdScore, 2);
    }
    if (predNotInOrig.size()) {
      svdVarPredNotInOrig = svdVarPredNotInOrig/predNotInOrig.size();
    }
    svdVarOfPredNotInOrig += svdVarPredNotInOrig;

    auto freqInTopOrig = orderingOverlap(itemOrigScoresPair, itemFreqScoresPair, topBuckN);
    auto svdInTopOrig = orderingOverlap(itemOrigScoresPair, itemSVDScoresPair, topBuckN);
    auto freqSVDDiffPc = compDiffPc(freqInTopOrig, svdInTopOrig);
    freqNOTinSVDPc += freqSVDDiffPc.first;
    svdNOTinFreqPc += freqSVDDiffPc.second;
   
    freqSVDTopOvrlap += compOrderingOverlap(itemFreqScoresPair, itemSVDScoresPair, topBuckN);
    freqTopRMSE += getSE(user, itemFreqScoresPair, origModel, fullModel, topBuckN);
    svdTopRMSE += getSE(user, itemSVDScoresPair, origModel, fullModel, topBuckN);

    if (NULL != graphMat) {
      itemPPRScoresPair = itemGraphItemScores(user, graphMat, trainMat, 
        0.01, nUsers, nItems, invalItems, false);
      std::map<int, double> pprMap;
      for (auto&& itemScore: itemPPRScoresPair) {
        if (isnan(itemScore.second)) {
          std::cerr << "Found NaN: " << itemScore.first << " " 
            << itemScore.second << std::endl;
        } else {
          pprMap[itemScore.first] = itemScore.second;
        }
      }

      updateMisPredBins(misPredPPRCountBins, pprScoreBins, itemOrigScoresPair,
          itemPredTopN, topBuckN, pprMap, user);

      updateMisPredBins(misGTPPRCountBins, misGTPPRScoreBins, itemPredScoresPair,
          itemOrigTopN, topBuckN, pprMap, user);

      double pprScore = 0;
      for (auto&& pairs: predInOrigTop) {
        int item = pairs.first;
        pprScore += pprMap[item];
      }
      if (predInOrigTop.size()) {
        pprScore = pprScore/predInOrigTop.size();
      }
      pprOfPredInOrig += pprScore;
      
      pprScore = 0;
      for (auto&& pairs: predNotInOrig) {
        int item = pairs.first;
        pprScore += pprMap[item];
      }
      if (predNotInOrig.size()) {
        pprScore = pprScore/predNotInOrig.size();
      }
      pprOfPredNotInOrig += pprScore;
      
      pprOrigOverlap += compOrderingOverlap(itemOrigScoresPair,
          itemPPRScoresPair, topBuckN);
      
      auto pprInTopOrig = orderingOverlap(itemOrigScoresPair, itemPPRScoresPair, topBuckN);
      auto freqPPRDiffPc = compDiffPc(freqInTopOrig, pprInTopOrig);
      freqNOTinPPRPc += freqPPRDiffPc.first;
      pprNOTinFreqPc += freqPPRDiffPc.second;
      auto svdPPRDiffPc = compDiffPc(svdInTopOrig, pprInTopOrig);
      svdNOTinPPRPc += svdPPRDiffPc.first;
      pprNOTinSVDPc += svdPPRDiffPc.second;

      pprFreqTopOvrlap += compOrderingOverlap(itemPPRScoresPair, itemFreqScoresPair, topBuckN);
      pprSVDTopOvrlap += compOrderingOverlap(itemPPRScoresPair, itemSVDScoresPair, topBuckN);
      pprTopRMSE += getSE(user, itemPPRScoresPair, origModel, fullModel, topBuckN);
    }
    
    predOrigOverlap += compOrderingOverlap(itemOrigScoresPair, 
        itemPredScoresPair, topBuckN);
    svdOrigOverlap += compOrderingOverlap(itemOrigScoresPair,
        itemSVDScoresPair, topBuckN);
    freqOrigOverlap += compOrderingOverlap(itemOrigScoresPair, 
        itemFreqScoresPair, topBuckN);
    svdPredOverlap += compOrderingOverlap(itemSVDScoresPair,
        itemPredScoresPair, topBuckN);

    auto itemOptScoresPair = itemOptScores(origModel, fullModel, user, 
        data.trainMat, nItems, invalItems);
    itemRMSEsOrdByItemScores(user, filtItems, fullModel, origModel, itemsRMSE,
        itemsScore, itemOptScoresPair);
    //reset user specific vec to 0
    std::fill(uBucketNNZ.begin(), uBucketNNZ.end(), 0); 
    std::fill(uRMSEOPTScores.begin(), uRMSEOPTScores.end(), 0);
    updateBucketsArr(uRMSEOPTScores, uBucketNNZ, itemsRMSE, nBuckets);

    //get itemsRMSE and itemsScore
    itemRMSEsOrdByItemScores(user, filtItems, fullModel, origModel, itemsRMSE, 
        itemsScore, itemSVDScoresPair);
    
    //reset user specific vec to 0
    std::fill(uBucketNNZ.begin(), uBucketNNZ.end(), 0); 
    std::fill(uRMSESVDScores.begin(), uRMSESVDScores.end(), 0);
    updateBucketsArr(uRMSESVDScores, uBucketNNZ, itemsRMSE, nBuckets);

    //get itemsRMSE and itemsScore
    itemRMSEsOrdByItemScores(user, filtItems, fullModel, origModel, itemsRMSE, 
        itemsScore, itemFreqScoresPair);
    
    //reset user specific vec to 0
    std::fill(uBucketNNZ.begin(), uBucketNNZ.end(), 0); 
    std::fill(uRMSEFreqScores.begin(), uRMSEFreqScores.end(), 0);
    updateBucketsArr(uRMSEFreqScores, uBucketNNZ, itemsRMSE, nBuckets);

    if (NULL != graphMat) {
      //get itemsRMSE and itemsScore
      itemRMSEsOrdByItemScores(user, filtItems, fullModel, origModel, itemsRMSE, 
          itemsScore, itemPPRScoresPair);
      
      //reset user specific vec to 0
      std::fill(uBucketNNZ.begin(), uBucketNNZ.end(), 0); 
      std::fill(uRMSEPPRScores.begin(), uRMSEPPRScores.end(), 0);
      updateBucketsArr(uRMSEPPRScores, uBucketNNZ, itemsRMSE, nBuckets);
    }

    //reset user specific vec to 0
    std::fill(uBucketNNZ.begin(), uBucketNNZ.end(), 0); 
    std::fill(uScores.begin(), uScores.end(), 0);
    updateBucketsArr(uScores, uBucketNNZ, itemsScore, nBuckets);
 
    //get itemsRMSE and itemsScore
    itemRMSEsOrdByItemScores(user, filtItems, fullModel, origModel, itemsRMSE, 
        itemsScore, itemOrigScoresPair);
    
    //reset user specific vec to 0
    std::fill(uBucketNNZ.begin(), uBucketNNZ.end(), 0); 
    std::fill(uRMSEGTScores.begin(), uRMSEGTScores.end(), 0);
    updateBucketsArr(uRMSEGTScores, uBucketNNZ, itemsRMSE, nBuckets);
    
    //reset user specific vec to 0
    std::fill(uBucketNNZ.begin(), uBucketNNZ.end(), 0); 
    std::fill(uScores.begin(), uScores.end(), 0);
    updateBucketsArr(uScores, uBucketNNZ, itemsScore, nBuckets);
   
    //write and update aggregated buckets
    for (int i = 0; i < nBuckets; i++) {
      bucketNNZ[i]      += uBucketNNZ[i];
      rmseGTScores[i]   += uRMSEGTScores[i];
      rmseOPTScores[i]  += uRMSEOPTScores[i];
      rmseSVDScores[i]  += uRMSESVDScores[i];
      rmseFreqScores[i] += uRMSEFreqScores[i];
      rmsePPRScores[i]  += uRMSEPPRScores[i];
      scores[i]         += uScores[i];
    }
  } //end for user

#pragma omp critical
{
  for (int i = 0; i < nBuckets; i++) {
      g_bucketNNZ[i]      += bucketNNZ[i];
      g_rmseGTScores[i]   += rmseGTScores[i];
      g_rmseOPTScores[i]  += rmseOPTScores[i];
      g_rmseSVDScores[i]  += rmseSVDScores[i];
      g_rmseFreqScores[i] += rmseFreqScores[i];
      g_rmsePPRScores[i]  += rmsePPRScores[i];
      g_scores[i]         += scores[i];
  }

  for (int i = 0; i < 20; i++) {
    
    g_misPredSVDCountBins[i]  += misPredSVDCountBins[i];
    g_misPredFreqCountBins[i] += misPredFreqCountBins[i];
    g_misPredPPRCountBins[i]  += misPredPPRCountBins[i];
   
    g_svdScoreBins[i]  += svdScoreBins[i];
    g_freqScoreBins[i] += freqScoreBins[i];
    g_pprScoreBins[i]  += pprScoreBins[i];

    g_misGTSVDCountBins[i]  += misGTSVDCountBins[i];
    g_misGTSVDScoreBins[i]  += misGTSVDScoreBins[i];
    g_misGTOrigCountBins[i] += misGTOrigCountBins[i];
    g_misGTOrigScoreBins[i] += misGTOrigScoreBins[i];
    
    g_misGTMFCountBins[i]   += misGTMFCountBins[i];
    g_misGTMFScoreBins[i]   += misGTMFScoreBins[i];
    g_misGTFreqScoreBins[i] += misGTFreqScoreBins[i];
    g_misGTFreqCountBins[i] += misGTFreqCountBins[i];

    g_misGTAvgTrainCountBins[i] += misGTAvgTrainCountBins[i];
    g_misGTAvgTrainScoreBins[i] += misGTAvgTrainScoreBins[i];
    g_misGTPPRCountBins[i]      += misGTPPRCountBins[i];
    g_misGTPPRScoreBins[i]      += misGTPPRScoreBins[i];
  }

}

}

  for (int i = 0; i < nBuckets; i++) {
    g_rmseGTScores[i] = sqrt(g_rmseGTScores[i]/g_bucketNNZ[i]);
    g_rmseOPTScores[i] = sqrt(g_rmseOPTScores[i]/g_bucketNNZ[i]);
    g_rmseSVDScores[i] = sqrt(g_rmseSVDScores[i]/g_bucketNNZ[i]);
    g_rmsePPRScores[i] = sqrt(g_rmsePPRScores[i]/g_bucketNNZ[i]);
    g_rmseFreqScores[i] = sqrt(g_rmseFreqScores[i]/g_bucketNNZ[i]);
    g_scores[i] = g_scores[i]/g_bucketNNZ[i];
  }

  std::ofstream opFile;
  if (NULL != graphMat) {
    opFile.open(prefix + "samp.txt");
  } else {
    opFile.open(prefix + ".txt");
  }

  opFile << "GT RMSE buckets: ";
  writeVector(g_rmseGTScores, opFile); 
  opFile << std::endl;

  opFile << "SVD RMSE buckets: ";
  writeVector(g_rmseSVDScores, opFile);
  opFile << std::endl;

  opFile << "Freq RMSE buckets: ";
  writeVector(g_rmseFreqScores, opFile);
  opFile << std::endl;

  opFile << "OPT RMSE buckets: ";
  writeVector(g_rmseOPTScores, opFile);
  opFile << std::endl;

  opFile << "PPR RMSE buckets: ";
  writeVector(g_rmsePPRScores, opFile);
  opFile << std::endl;

  opFile << "GT Score buckets: ";
  writeVector(g_scores, opFile);
  opFile << std::endl;
  
  opFile << "No. sample users: " << totalSampUsers << std::endl;
  opFile << "predOrigOverlap: " << predOrigOverlap/totalSampUsers << std::endl;
  opFile << "svdOrigOverlap: " << svdOrigOverlap/totalSampUsers << std::endl;
  opFile << "freqOrigOverlap: " << freqOrigOverlap/totalSampUsers << std::endl;
  opFile << "pprOrigOverlap: " << pprOrigOverlap/totalSampUsers << std::endl;

  opFile << "freqNOTinSVDPc: " << freqNOTinSVDPc/totalSampUsers << std::endl;
  opFile << "svdNOTinFreqPc: " << svdNOTinFreqPc/totalSampUsers << std::endl;
  opFile << "freqNOTinPPRPc: " << freqNOTinPPRPc/totalSampUsers << std::endl;
  opFile << "pprNOTinFreqPc: " << pprNOTinFreqPc/totalSampUsers << std::endl;
  opFile << "svdNOTinPPRPc: " << svdNOTinPPRPc/totalSampUsers << std::endl;
  opFile << "pprNOTinSVDPc: " << pprNOTinSVDPc/totalSampUsers << std::endl;

  opFile << "freqSVDTopOvrlap: " << freqSVDTopOvrlap/totalSampUsers << std::endl;
  opFile << "pprFreqTopOvrlap: " << pprFreqTopOvrlap/totalSampUsers << std::endl;
  opFile << "pprSVDTopOvrlap: " << pprSVDTopOvrlap/totalSampUsers << std::endl;

  opFile << "freqTopRMSE: " << sqrt(freqTopRMSE/(totalSampUsers*topBuckN)) << std::endl;
  opFile << "pprTopRMSE: " << sqrt(pprTopRMSE/(totalSampUsers*topBuckN)) << std::endl;
  opFile << "svdTopRMSE: " << sqrt(svdTopRMSE/(totalSampUsers*topBuckN)) << std::endl;

  opFile << "svdInPred: " << svdInPred/totalSampUsers << std::endl;
  opFile << "svdNotInPred: " << svdNotInPred/totalSampUsers << std::endl;
  
  opFile << "pprofPredInOrig: " << pprOfPredInOrig/totalSampUsers << std::endl;
  opFile << "pprOfPredNotInOrig: " << pprOfPredNotInOrig/totalSampUsers << std::endl;

  opFile << "svdPredOverlap: " << svdPredOverlap/totalSampUsers << std::endl;
  
  opFile << "svdOfPredInOrig: " << svdOfPredInOrig/totalSampUsers << std::endl
    << "svdOfPredNotInOrig: " << svdOfPredNotInOrig/totalSampUsers << std::endl;
  
  opFile << "svdVarOfPredInOrig: " << svdVarOfPredInOrig/totalSampUsers << std::endl
    << "svdVarOfPredNotInOrig: " << svdVarOfPredNotInOrig/totalSampUsers << std::endl;
 
  opFile << "svdAboveAvgInOrig: " << svdAboveAvgInOrig/totalSampUsers << std::endl; 

  opFile << "svdOfMaxPredInorig: " << svdOfMaxPredInorig/totalSampUsers << std::endl;
  opFile << "svdOfMinPredInOrig: " << svdOfMinPredInOrig/totalSampUsers << std::endl;
  opFile << "svdOfTopPredInOrig: " << svdOfTopPredInOrig/totalSampUsers << std::endl;
  opFile << "svdOfMedPredInOrig: " << svdOfMedPredInOrig/totalSampUsers << std::endl; 
  opFile << "svdOfBotPredInOrig: " << svdOfBotPredInOrig/totalSampUsers << std::endl;

  opFile << "pprOfPredInOrig: " << pprOfPredInOrig/totalSampUsers << std::endl
    << "pprOfPredNotInOrig: " << pprOfPredNotInOrig/totalSampUsers << std::endl;
  
  opFile << "iterPredSVDOrigOverlap: " 
    << iterPredSVDOrigOverlap/totalSampUsers << std::endl;


  int sumSVDBins = 0;
  int sumPPRBins = 0;
  int sumGTSVDBins = 0;
  int sumGTPPRBins = 0;
  for (int i = 0; i < 20; i++) {
    sumSVDBins += g_misPredSVDCountBins[i];
    sumPPRBins += g_misPredPPRCountBins[i];

    sumGTSVDBins += g_misGTSVDCountBins[i];
    sumGTPPRBins += g_misGTPPRCountBins[i];
  }

  opFile << "Total SVD misPred: " << sumSVDBins << std::endl;
 
  opFile << "Mispred SVD %: ";
  for (int i = 0; i < 20; i++) {
    opFile << g_misPredSVDCountBins[i]/sumSVDBins << ",";
  }
  opFile << std::endl;

  opFile << "Mispred avg SVD: ";
  for (int i = 0; i < 20; i++) {
    opFile << g_svdScoreBins[i]/g_misPredSVDCountBins[i] << ",";
  }
  opFile << std::endl;
  
  opFile << "Mispred avg Freq: ";
  for (int i = 0; i < 20; i++) {
    opFile << g_freqScoreBins[i]/g_misPredFreqCountBins[i] << ",";
  }

  opFile << std::endl;
  opFile << "Total PPR misPred: " << sumPPRBins << std::endl;
  opFile << "Mispred PPR %: ";
  for (int i = 0; i < 20; i++) {
    opFile << g_misPredPPRCountBins[i]/sumPPRBins << ","; 
  }
  opFile << std::endl;

  opFile << "Mispred avg PPR: ";
  for (int i = 0; i < 20; i++) {
    opFile << g_pprScoreBins[i]/g_misPredPPRCountBins[i] << ","; 
  }
  opFile << std::endl;
 
  opFile << "Total SVD misGT: " << sumGTSVDBins << std::endl;
  opFile << "MisGT SVD %: ";
  for (int i = 0; i < 20; i++) {
    opFile << g_misGTSVDCountBins[i]/sumGTSVDBins << ",";
  }
  opFile << std::endl;

  opFile << "MisGT avg SVD: ";
  for (int i =0; i < 20; i++) {
    opFile << g_misGTSVDScoreBins[i]/g_misGTSVDCountBins[i] << ",";
  }
  opFile << std::endl;

  opFile << "MisGT avg Freq: ";
  for (int i =0; i < 20; i++) {
    opFile << g_misGTFreqScoreBins[i]/g_misGTFreqCountBins[i] << ",";
  }
  opFile << std::endl;

  opFile << "MisGT avg AllRatings: ";
  for (int i =0; i < 20; i++) {
    opFile << g_misGTAvgTrainScoreBins[i]/g_misGTAvgTrainCountBins[i] << ",";
  }
  opFile << std::endl;

  opFile << "MisGT avg GT: ";
  for (int i =0; i < 20; i++) {
    opFile << g_misGTOrigScoreBins[i]/g_misGTOrigCountBins[i] << ",";
  }
  opFile << std::endl;

  opFile << "MisGT avg Pred: ";
  for (int i =0; i < 20; i++) {
    opFile << g_misGTMFScoreBins[i]/g_misGTMFCountBins[i] << ",";
  }
  opFile << std::endl;

  opFile << std::endl;
  opFile << "Total PPR misGT: " << sumGTSVDBins << std::endl;
  opFile << "MisGT PPR %: ";
  for (int i = 0; i < 20; i++) {
    opFile << g_misGTPPRCountBins[i]/sumGTPPRBins << ",";
  }
  opFile << std::endl;

  opFile << "MisGT avg PPR: ";
  for (int i =0; i < 20; i++) {
    opFile << g_misGTPPRScoreBins[i]/g_misGTPPRCountBins[i] << ",";
  }
  opFile << std::endl;
  
  opFile << "Train RMSE: " 
    << fullModel.RMSE(trainMat, invalUsers, invalItems) << std::endl;
  opFile << "Test RMSE: " 
    << fullModel.RMSE(data.testMat, invalUsers, invalItems) << std::endl;
  opFile << "Val RMSE: " 
    << fullModel.RMSE(data.valMat, invalUsers, invalItems) << std::endl;
  opFile << "Full RMSE: " 
    << fullModel.fullLowRankErr(data, invalUsers, invalItems) << std::endl;

  opFile.close();
}


//return a map of item- users
std::map<int, std::vector<int>> pprSampTopNItemsUsers(gk_csr_t *graphMat, 
    gk_csr_t *trainMat,
    int nUsers, int nItems, Model& origModel, Model& fullModel,
    float lambda, int max_niter, std::unordered_set<int>& invalUsers, 
    std::unordered_set<int>& invalItems, std::unordered_set<int>& filtItems, 
    int nSampUsers, int seed, int N, std::string& prefix) {
  
  int nBuckets = 10;
  int nItemsPerBuck = N/nBuckets;

  std::vector<double> bucketScores(nBuckets, 0);
  std::vector<double> bucketNNZ(nBuckets, 0);

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
    
    itemScores = itemGraphItemScores(user, graphMat, trainMat, lambda, nUsers, 
        nItems, invalItems, true);
    
    std::vector<int> sortedItems;
    int i = 0;
    while (sortedItems.size() < N) {
      int item = itemScores[i++].first;
      
      /*
      auto search = filtItems.find(item);
      if (search != filtItems.end()) {
        //found in filtered items, skip
        continue;
      }
      */

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


  //write out bucket scores
  std::string scoreFName = prefix + "_" + std::to_string(lambda) + "_ppr_bucket.txt";
  writeVector(bucketScores, scoreFName.c_str());

  //write out bucket nnz
  std::string nnzFName = prefix + "_" + std::to_string(lambda) + "_nnz_bucket.txt";
  writeVector(bucketNNZ, nnzFName.c_str());

  return itemUsers;
}


std::map<int, std::vector<int>> svdSampTopNItemsUsers(gk_csr_t *trainMat,
    int nUsers, int nItems, Model& origModel, Model& fullModel, Model& svdModel,
    std::unordered_set<int>& invalUsers, std::unordered_set<int>& invalItems, 
    std::unordered_set<int>& filtItems, 
    int nSampUsers, int seed, int N, std::string& prefix) {
  
  int nBuckets = 10;
  int nItemsPerBuck = N/nBuckets;

  std::vector<double> bucketScores(nBuckets, 0);
  std::vector<double> bucketNNZ(nBuckets, 0);

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
  
    itemScores = itemSVDScores(svdModel, user, trainMat, nItems, invalItems);
    
    std::vector<int> sortedItems;
    int i = 0;
    while (sortedItems.size() < N) {
      int item = itemScores[i++].first;
      
      /*
      auto search = filtItems.find(item);
      if (search != filtItems.end()) {
        //found in filtered items, skip
        continue;
      }
      */

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
      std::cout << "Done... " << sampUsers.size() << " :" << std::endl;
    }

  }

  for (size_t i = 0; i < bucketScores.size(); i++) {
    bucketScores[i] = sqrt(bucketScores[i]/bucketNNZ[i]);
  }


  //write out bucket scores
  std::string scoreFName = prefix + "_svd_bucket.txt";
  writeVector(bucketScores, scoreFName.c_str());

  //write out bucket nnz
  std::string nnzFName = prefix +  "_nnz_bucket.txt";
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


void writeTopBuckRMSEs(Model& origModel, Model& fullModel, Model& svdModel,
    gk_csr_t* graphMat,
    gk_csr_t* trainMat,
    float lambda, int max_niter, std::unordered_set<int>& invalUsers, 
    std::unordered_set<int>& invalItems, std::unordered_set<int>& filtItems,
    int nSampUsers, int seed, int N, std::string prefix) {
  
  std::cout << "\nGetting itemUsers... : " << lambda << std::endl;

  /*
  auto itemUsers = pprSampTopNItemsUsers(graphMat, trainMat, fullModel.nUsers, 
      fullModel.nItems, origModel, fullModel, lambda, max_niter, invalUsers, 
      invalItems, filtItems, nSampUsers, seed, N, prefix);
  */

  auto itemUsers = svdSampTopNItemsUsers(trainMat, fullModel.nUsers, 
      fullModel.nItems, origModel, fullModel, svdModel, invalUsers, invalItems, 
      filtItems, nSampUsers, seed, N, prefix);

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
  
  std::string fname = prefix + "_svdtop_" + std::to_string(N) + "_freq_" 
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


