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
   
    /*
    if (0 == user%1000) {
      std::cout << "\n buckScores u: " << user;
      dispVector(bucketScores);
    }
    */
  
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
  }
  
  free(pr);
  
  for (int i = 0; i < nBuckets; i++) {
    bucketScores[i] = sqrt(bucketScores[i]/bucketNNZ[i]);
  }
  return bucketScores;
}









