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


std::vector<std::pair<int, double>> prodScores(
    std::vector<std::pair<int, double>> pairScores1,
    std::vector<std::pair<int, double>> pairScores2) {
  
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
            mapScores1[key]*score));
    }
  }
  
  std::sort(resScores.begin(), resScores.end(), descComp);
  
  return resScores;
}


std::vector<std::pair<int, double>> itemGraphItemScores(int user, 
    gk_csr_t *graphMat, gk_csr_t *mat, float lambda, int nUsers, 
    int nItems, std::unordered_set<int>& invalItems) {

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
    //pr[item] = 1.0/nUserRat;
    sumRat += mat->rowval[ii];
  }
  
  for (int ii = mat->rowptr[user]; ii < mat->rowptr[user+1]; ii++) {
    int item = mat->rowind[ii];
    pr[item] = mat->rowval[ii]/sumRat; 
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
    Model& fullModel, Model& origModel, int user,
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
    itemScores.push_back(std::make_pair(item, predRating));
  }
  
  //sort items in decreasing order of score
  std::sort(itemScores.begin(), itemScores.end(), descComp);
  
  return itemScores;
}


std::vector<std::pair<int, double>> itemOrigScores( 
    Model& fullModel, Model& origModel, int user,
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


void itemRMSEsProb(int user, gk_csr_t *graphMat, gk_csr_t *trainMat, 
    float lambda, int nUsers, int nItems, std::unordered_set<int>& invalItems,
    std::unordered_set<int>& filtItems, Model& fullModel, Model& origModel,
    std::vector<double>& itemsRMSE, std::vector<double>& itemsProb) {
  
  std::vector<std::pair<int, double>> itemScores = itemGraphItemScores(user, 
      graphMat, trainMat, lambda, nUsers, nItems, invalItems);
  
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
std::vector<std::pair<int, double>> compOrderingDiff(
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

  int topBuckN = 0.1*nItems;

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

    auto itemPredScoresPair = itemPredScores(fullModel, origModel, user, trainMat, 
        nItems, invalItems);
    auto itemOrigScoresPair = itemOrigScores(fullModel, origModel, user, trainMat, 
        nItems, invalItems);
    auto itemSVDScoresPair = itemSVDScores(svdModel, user, trainMat, nItems, 
        invalItems);
    //auto itemPPRScoresPair = itemGraphItemScores(user, graphMat, trainMat, 
    //    0.01, nUsers, nItems, invalItems);

    auto itemFreqPair = itemFreqScores(fullModel, origModel, itemFreq,
        user, trainMat, nItems, invalItems);
    auto itemPredSVDScoresPair = prodScores(itemPredScoresPair, 
        itemSVDScoresPair);
    //auto itemPredPPRScoresPair = prodScores(itemPredScoresPair, 
    //    itemPPRScoresPair);
    auto svdItemPairsNotInPred = compOrderingDiff(itemPredScoresPair,
        itemSVDScoresPair, topBuckN);
    
    //auto pprItemPairsNotInPred = compOrderingDiff(itemPredScoresPair,
    //    itemPPRScoresPair, topBuckN);

    auto freqItemPairsNotInPred = compOrderingDiff(itemPredScoresPair,
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


void predSampUsersRMSEProb2(gk_csr_t *trainMat, gk_csr_t *graphMat, 
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

  auto avgTrainRating = meanRating(trainMat);
  
  int predLowcount = 0, predHighcount = 0;
  int lowCount = 0, highCount = 0;
  
  auto rowColFreq = getRowColFreq(trainMat);
  auto userFreq = rowColFreq.first;
  auto itemFreq = rowColFreq.second;

  int topBuckN = 0.05*nItems;

  std::cout << "\ntopBuckN: " << topBuckN << std::endl;

  int nHighPredUsers = 0;

  double predOrigOverlap = 0, svdOrigOverlap = 0;
  double svdPredOverlap = 0;
  double pprOrigOverlap = 0;
  double svdNotInPred = 0, svdInPred = 0;
  double svdOfPredInOrig = 0, svdOfPredNotInOrig = 0;
  double pprOfPredInOrig = 0, pprOfPredNotInOrig = 0;
  double pprNotInPred = 0, pprInPred = 0;
  double predPPROrigOverlap = 0;

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

    auto itemPredScoresPair = itemPredScores(fullModel, origModel, user, trainMat, 
        nItems, invalItems);
    auto itemOrigScoresPair = itemOrigScores(fullModel, origModel, user, trainMat, 
        nItems, invalItems);
    auto itemSVDScoresPair = itemSVDScores(svdModel, user, trainMat, nItems, 
        invalItems);
    std::vector<std::pair<int, double>> itemPPRScoresPair;


    auto predInOrigTop = orderingOverlap(itemOrigScoresPair, 
        itemPredScoresPair, topBuckN);
    double svdScore = 0;
    for (auto&& pairs: predInOrigTop) {
      int item = pairs.first;
      svdScore += svdModel.estRating(user, item);
    }
    if (predInOrigTop.size()) {
      svdScore = svdScore/predInOrigTop.size();
    }
    svdOfPredInOrig += svdScore;

    auto predNotInOrig = compOrderingDiff(itemOrigScoresPair,
        itemPredScoresPair, topBuckN);
    svdScore = 0;
    for (auto&& pairs: predNotInOrig) {
      int item = pairs.first;
      svdScore += svdModel.estRating(user, item);  
    }
    if (predNotInOrig.size()) {
      svdScore = svdScore/predNotInOrig.size();
    }
    svdOfPredNotInOrig += svdScore;

    if (NULL != graphMat) {
      itemPPRScoresPair = itemGraphItemScores(user, graphMat, trainMat, 
        0.01, nUsers, nItems, invalItems);
      std::map<int, double> pprMap;
      for (auto&& itemScore: itemPPRScoresPair) {
        pprMap[itemScore.first] = itemScore.second;
      }

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
    }

    //auto pprItemPairsNotInPred = compOrderingDiff(itemPredScoresPair,
    //    itemPPRScoresPair, topBuckN);

    predOrigOverlap += compOrderingOverlap(itemOrigScoresPair, 
        itemPredScoresPair, topBuckN);
    svdOrigOverlap += compOrderingOverlap(itemOrigScoresPair,
        itemSVDScoresPair, topBuckN);

    //pprOrigOverlap += compOrderingOverlap(itemOrigScoresPair,
    //    itemPPRScoresPair, topBuckN);
    
    svdPredOverlap += compOrderingOverlap(itemSVDScoresPair,
        itemPredScoresPair, topBuckN);

    //auto itemPredPPRScoresPair = prodScores(itemPredScoresPair,
    //    itemPPRScoresPair);

    auto itemScoresPair = itemOrigScoresPair;

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
    for (int i = 0; i < nBuckets; i++) {
      bucketNNZ[i]  += uBucketNNZ[i];
      rmseScores[i] += uRMSEScores[i];
      scores[i]     += uScores[i];
    }

    if (sampUsers.size() % PROGU == 0) {
      std::cout << "Done... " << sampUsers.size() << " :" << std::endl;
    }

  }
  
  for (int i = 0; i < nBuckets; i++) {
    rmseScores[i] = sqrt(rmseScores[i]/bucketNNZ[i]);
    scores[i] = scores[i]/bucketNNZ[i];
  }
  std::string rmseScoreFName = prefix + "_rmseBuckets.txt";
  writeVector(rmseScores, rmseScoreFName.c_str());

  std::string avgScoreFName = prefix + "_avgScore.txt";
  writeVector(scores, avgScoreFName.c_str());

  std::cout << "predOrigOverlap: " << predOrigOverlap/sampUsers.size() << std::endl;
  std::cout << "svdOrigOverlap: " << svdOrigOverlap/sampUsers.size() << std::endl;
  std::cout << "svdInPred: " << svdInPred/sampUsers.size() << std::endl;
  std::cout << "svdNotInPred: " << svdNotInPred/sampUsers.size() << std::endl;
  //std::cout << "pprOrigOverlap: " << pprOrigOverlap/sampUsers.size() << std::endl;

  //std::cout << "pprNotInPredOrigOverlap: " << pprNotInPredOrigOverlap/sampUsers.size() 
  //  << std::endl;
  std::cout << "svdPredOverlap: " << svdPredOverlap/sampUsers.size() << std::endl;
  
  std::cout << "svdOfPredInOrig: " << svdOfPredInOrig/sampUsers.size()
    << " svdOfPredNotInOrig: " << svdOfPredNotInOrig/sampUsers.size() << std::endl;
  
  std::cout << "pprOfPredInOrig: " << pprOfPredInOrig/sampUsers.size()
    << " pprOfPredNotInOrig: " << pprOfPredNotInOrig/sampUsers.size();

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
        nItems, invalItems);
    
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
    gk_csr_t* trainMat,
    float lambda, int max_niter, std::unordered_set<int>& invalUsers, 
    std::unordered_set<int>& invalItems, std::unordered_set<int>& filtItems,
    int nSampUsers, int seed, int N, std::string prefix) {
  
  std::cout << "\nGetting itemUsers... : " << lambda << std::endl;

  auto itemUsers = pprSampTopNItemsUsers(graphMat, trainMat, fullModel.nUsers, 
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


