#include "longTail.h"


bool isModelHit(Model& model, std::unordered_set<int>& sampItems, int user, 
    int testItem, int N) {

  std::vector<std::pair<int, double>> itemRatings;
  for (auto && item: sampItems) {
    itemRatings.push_back(std::make_pair(item, model.estRating(user, item)));
  }
  itemRatings.push_back(std::make_pair(testItem, 
        model.estRating(user, testItem)));

  //sort itemRatings such that Nth rating is at its correct place in
  //decreasing order
  std::nth_element(itemRatings.begin(), itemRatings.begin()+N, 
      itemRatings.end(), descComp);

  for (int i = 0; i < N; i++) {
    if (itemRatings[i].first == testItem) {
      //hit
      return true;
    }
  }
  
  return false;
}


bool isModelModelHit(Model& model1, Model& model2, 
    std::unordered_set<int>& sampItems, int user, int testItem, int N) {

  std::vector<std::pair<int, double>> itemRatings;
  for (auto && item: sampItems) { 
    itemRatings.push_back(std::make_pair(item, 
          model1.estRating(user, item)*model2.estRating(user, item)));
  }
  itemRatings.push_back(std::make_pair(testItem, 
        model1.estRating(user, testItem)*model2.estRating(user, testItem)));

  //sort itemRatings such that Nth rating is at its correct place in
  //decreasing order
  std::nth_element(itemRatings.begin(), itemRatings.begin()+N, 
      itemRatings.end(), descComp);

  for (int i = 0; i < N; i++) { 
    if (itemRatings[i].first ==  testItem) {
      //hit
      return true;
    }
  }
  
  return false;
}


bool isModelLocalScoreHit(Model& model, std::unordered_set<int>& sampItems, int user, 
    int testItem, std::vector<double> itemScores, int N) {

  std::vector<std::pair<int, double>> itemRatings;
  for (auto && item: sampItems) {
    itemRatings.push_back(std::make_pair(item, 
          model.estRating(user, item)*itemScores[item]));
  } 
  itemRatings.push_back(std::make_pair(testItem, 
        model.estRating(user, testItem)*itemScores[testItem]));

  //sort itemRatings such that Nth rating is at its correct place in
  //decreasing order
  std::nth_element(itemRatings.begin(), itemRatings.begin()+N, 
      itemRatings.end(), descComp);

  for (int i = 0; i < N; i++) {
    if (itemRatings[i].first == testItem) {
      //hit
      return true;
    }
  }
  
  return false;
}


bool isLocalScoreHit(std::unordered_set<int>& sampItems, int user, 
    int testItem, std::vector<double> itemScores, int N) {

  std::vector<std::pair<int, double>> itemRatings;
  for (auto && item: sampItems) {
    itemRatings.push_back(std::make_pair(item, itemScores[item]));
  } 
  itemRatings.push_back(std::make_pair(testItem, itemScores[testItem]));

  //sort itemRatings such that Nth rating is at its correct place in
  //decreasing order
  std::nth_element(itemRatings.begin(), itemRatings.begin()+N, 
      itemRatings.end(), descComp);

  for (int i = 0; i < N; i++) {
    if (itemRatings[i].first == testItem) {
      //hit
      return true;
    }
  }
  
  return false;
}


bool isModelLocalIterHit(Model& model, std::unordered_set<int>& sampItems, int user, 
    int testItem, std::vector<double> itemScores, int N) {

  std::vector<std::pair<int, double>> itemRatings;
  for (auto && item:  sampItems) {
    itemRatings.push_back(std::make_pair(item, 
          model.estRating(user, item)));
  } 
  itemRatings.push_back(std::make_pair(testItem, 
        model.estRating(user, testItem)));

  //sort itemRatings such that 5*Nth rating is at its correct place in
  //decreasing order
  std::nth_element(itemRatings.begin(), itemRatings.begin()+(5*N), 
      itemRatings.end(), descComp);
  
  std::vector<std::pair<int, double>> itemLocalScore;
  for (int i = 0; i < 5*N; i++) {
    itemLocalScore.push_back(std::make_pair(itemRatings[i].first,
          itemScores[itemRatings[i].first]));
  }
  
  //sort itemLocal score such that Nth rating is in its correct place as per
  //local score
  std::nth_element(itemLocalScore.begin(), itemLocalScore.begin()+N, 
      itemLocalScore.end(), descComp);

  for (int i = 0; i < N; i++) { 
    if (itemLocalScore[i].first ==  testItem) {
      //hit
      return true;
    }
  }
  
  return false;
}


bool isModelLocalIterHitRev(Model& model, std::unordered_set<int>& sampItems, int user, 
    int testItem, std::vector<double> itemScores, int N) {

  std::vector<std::pair<int, double>> itemLocalScore;
  for (auto && item:  sampItems) {
    itemLocalScore.push_back(std::make_pair(item, itemScores[item]));
  }
  itemLocalScore.push_back(std::make_pair(testItem, itemScores[testItem]));
  
  std::nth_element(itemLocalScore.begin(), itemLocalScore.begin() + (10*N), 
      itemLocalScore.end(), descComp);

  std::vector<std::pair<int, double>> itemRatings;
  for (int i = 0; i < 10*N; i++) {
    itemRatings.push_back(std::make_pair(itemLocalScore[i].first, 
          model.estRating(user, itemLocalScore[i].first)));
  }

  //sort itemLocal score such that Nth rating is in its correct place as per
  //local score
  std::nth_element(itemRatings.begin(), itemRatings.begin()+N, 
      itemRatings.end(), descComp);

  for (int i = 0; i < N; i++) { 
    if (itemRatings[i].first ==  testItem) {
      //hit
      return true;
    }
  }
  
  return false;
}


void topNRec(Model& model, gk_csr_t *trainMat, gk_csr_t *testMat, 
    gk_csr_t *graphMat, float lambda,
    std::unordered_set<int>& invalidItems,
    std::unordered_set<int>& invalidUsers,
    std::unordered_set<int>& headItems,
    int N, int seed) {

  double rec = 0, localRec = 0, localWtRec = 0;
  double headRec = 0, headLocalRec = 0, headLocalWtRec = 0;
  double tailRec = 0, tailLocalRec = 0, tailLocalWtRec = 0;
  int nItems = trainMat->ncols;
  int nUsers = trainMat->nrows;
  int nTestItems = 0, nHeadItems = 0, nTailItems = 0;
  bool isHeadItem;

  auto rowColFreq = getRowColFreq(trainMat);
  auto userFreq = rowColFreq.first;
  auto itemFreq = rowColFreq.second;

  std::cout << "\nlambda: " << lambda << " N: " << N << " seed: " << seed 
    << std::endl;

  //initialize random engine
  std::mt19937 mt(seed);
  //item distribution to sample items
  std::uniform_int_distribution<int> itemDist(0, nItems-1);
  std::uniform_int_distribution<int> userDist(0, nUsers-1);

  auto compPairsIndAsc = [] (std::pair<int, double> a, std::pair<int, double> b) {
    return a.first < b.second;
  };
  std::vector<std::pair<int, double>> itemScorePairs; 
  std::vector<double> itemScores(nItems, 0); 
  std::vector<double> itemFreqWtScores(nItems, 0); 

  std::unordered_set<int> testUsers;
  while (testUsers.size() < 10000) {
    int u = userDist(mt);
    //check if invalid users
    auto search = invalidUsers.find(u);
    if (search != invalidUsers.end()) {
      //found n skip
      continue;
    }

    //check if atleast one test item exists
    if (testMat->rowptr[u+1] - testMat->rowptr[u] == 0) {
      continue;
    }
   
    //check if user already evaluated
    search = testUsers.find(u);
    if (search != testUsers.end()) {
      //found
      continue;
    }

    //add to test users
    testUsers.insert(u);

    std::vector<int> trainItems;
    for (int ii = trainMat->rowptr[u]; ii < trainMat->rowptr[u+1]; ii++) {
      int item = trainMat->rowind[ii];
      trainItems.push_back(item);
    }
    //sort the train items to make binary search efficient
    //std::sort(trainItems.begin(), trainItems.end())
   
    //run personalized RW on graph w.r.t. user
    itemScorePairs = itemGraphItemScores(u, 
        graphMat, trainMat, lambda, nUsers, nItems, invalidItems);
    std::sort(itemScorePairs.begin(), itemScorePairs.end(), compPairsIndAsc);
    std::fill(itemScores.begin(), itemScores.end(), 0);
    std::fill(itemFreqWtScores.begin(), itemFreqWtScores.end(), 0);
    
    //NOTE: following is assuming that items are in descending order
    for (int i = 0; i < itemScorePairs.size(); i++) {
      int item = itemScorePairs[i].first;
      //assign item index in decreasing order as its score
      itemScores[item] = itemScorePairs.size() - i;
      if (itemFreq[item] > 0) {
        itemFreqWtScores[item] = itemScores[item]/itemFreq[item];
      }
    }

    /*
    for (auto&& itemScore: itemScorePairs) {
      itemScores[itemScore.first] = itemScore.second;
      if (itemFreq[itemScore.first] > 0) {
        itemFreqWtScores[itemScore.first] = itemScore.second/itemFreq[itemScore.first];
      }
    }
    */

    std::unordered_set<int> sampItems;
    for (int ii = testMat->rowptr[u]; ii < testMat->rowptr[u+1]; ii++) {
      int testItem = testMat->rowind[ii];
    
      //check if in head items
      search = headItems.find(testItem);
      if (search != headItems.end()) {
        isHeadItem = true;
        nHeadItems++;
      } else {
        isHeadItem = false;
        nTailItems++;
      }
      
      //sample 1000 unrated items at random
      sampItems.clear();
      while (sampItems.size() < 1000) {
        int sampItem = itemDist(mt);
        
        if (sampItem == testItem) {
          continue;
        }

        //check if sample item is present in train
        if (std::binary_search(trainItems.begin(), trainItems.end(), sampItem)) {
          continue;
        }
        
        //check if sample item is invalid
        search = invalidItems.find(sampItem);
        if (search != invalidItems.end()) {
          //found n skip
          continue;
        }

        sampItems.insert(sampItem);
      }
      
      if (isModelHit(model, sampItems, u, testItem, N)) {
        rec += 1;
        if (isHeadItem) {
          headRec += 1;
        } else {
          tailRec += 1;
        }
      }

      if (isModelLocalIterHit(model, sampItems, u, testItem, itemScores, N)) {
      //if (isModelLocalScoreHit(model, sampItems, u, testItem, itemScores, N)) {
        localRec += 1;
        if (isHeadItem) {
          headLocalRec += 1;
        } else {
          tailLocalRec += 1;
        }
      }


      //if (isModelLocalScoreHit(model, sampItems, u, testItem, 
      if (isModelLocalIterHit(model, sampItems, u, testItem, 
            itemFreqWtScores, N)) {
        localWtRec += 1;
        if (isHeadItem) {
          headLocalWtRec += 1;
        } else {
          tailLocalWtRec += 1;
        }
      }

      nTestItems++;
    }
    
    if (testUsers.size() % 1000 == 0) {
      std::cout << "Done.. " << testUsers.size() << std::endl;
      
      std::cout << "nTestItems: " << nTestItems << " nHeadItems: " 
        << nHeadItems << " nTailItems: " << nTailItems << std::endl;

      std::cout << "Top-" << N << " model recall: " 
        << rec/nTestItems << std::endl;
      std::cout << "Top-" << N << " model local recall: " 
        << localRec/nTestItems << std::endl;
      std::cout << "Top-" << N << " model local wt recall: " 
        << localWtRec/nTestItems << std::endl;
  
      std::cout << "Top-" << N << " model head recall: " 
        << headRec/nHeadItems << std::endl;
      std::cout << "Top-" << N << " model head local recall: " 
        << headLocalRec/nHeadItems << std::endl;
      std::cout << "Top-" << N << " model head local wt recall: " 
        << headLocalWtRec/nHeadItems << std::endl;
  
      std::cout << "Top-" << N << " model tail recall: " 
        << tailRec/nTailItems << std::endl;
      std::cout << "Top-" << N << " model tail local recall: " 
        << tailLocalRec/nTailItems << std::endl;
      std::cout << "Top-" << N << " model tail local wt recall: " 
        << tailLocalWtRec/nTailItems << std::endl;
    }
  }
  
  rec          = rec/nTestItems;
  localRec     = localRec/nTestItems;
  headRec      = headRec/nHeadItems;
  headLocalRec = headLocalRec/nHeadItems;
  tailRec      = tailRec/nTailItems;
  tailLocalRec = tailLocalRec/nTailItems;

  std::cout << "nTestItems: " << nTestItems << " nHeadItems: " 
    << nHeadItems << " nTailItems: " << nTailItems << std::endl;
  
  std::cout << "Top-" << N << " model recall: " 
    << rec << std::endl;
  std::cout << "Top-" << N << " model local recall: " 
    << localRec << std::endl;
  std::cout << "Top-" << N << " model local wt recall: " 
    << localWtRec << std::endl;

  std::cout << "Top-" << N << " model head recall: " 
    << headRec << std::endl;
  std::cout << "Top-" << N << " model head local recall: " 
    << headLocalRec << std::endl;
  std::cout << "Top-" << N << " model head local wt recall: " 
    << headLocalWtRec << std::endl;

  std::cout << "Top-" << N << " model tail recall: " 
    << tailRec << std::endl;
  std::cout << "Top-" << N << " model tail local recall: " 
    << tailLocalRec << std::endl;
  std::cout << "Top-" << N << " model tail local wt recall: " 
    << tailLocalWtRec << std::endl;
}


void topNRecTail(Model& model, gk_csr_t *trainMat, gk_csr_t *testMat, 
    gk_csr_t *graphMat, float lambda,
    std::unordered_set<int>& invalidItems,
    std::unordered_set<int>& invalidUsers,
    float headPc,
    int N, int seed, std::string opFileName) {

  double rec = 0, localRec = 0, pprRec = 0;
  double modelLocalRMSE = 0, modelRMSE = 0, localRMSE = 0;
  float testPredRating = 0, testRating = 0;
  int nItems = trainMat->ncols;
  int nUsers = trainMat->nrows;

  std::unordered_set<int> headItems = getHeadItems(trainMat, headPc);
  
  std::ofstream opFile(opFileName);

  opFile << "\nNo. of head items: " << headItems.size() << " head items pc: " 
    << ((float)headItems.size()/(trainMat->ncols)) << std::endl; 
  opFile << "\nlambda: " << lambda << " N: " << N << " seed: " << seed 
    << std::endl;

  //initialize random engine
  std::mt19937 mt(seed);
  //item distribution to sample items
  std::uniform_int_distribution<int> itemDist(0, nItems-1);
  std::uniform_int_distribution<int> userDist(0, nUsers-1);

  std::vector<std::pair<int, double>> itemScorePairs; 
  std::vector<double> itemScores(nItems, 0); 

  std::vector<int> testUsers;
  for (int u = 0; u < testMat->nrows; u++) {
    //check if invalid users
    auto search = invalidUsers.find(u);
    if (search != invalidUsers.end()) {
      //found n skip
      continue;
    }
    for (int ii = testMat->rowptr[u]; ii < testMat->rowptr[u+1]; ii++) {
      int item = testMat->rowind[ii];
      auto search = headItems.find(item);
      if (search == headItems.end()) {
        //tail item
        testUsers.push_back(u);
        break;
      }
    }
  }

  std::cout << "\nNo. of test users: " << testUsers.size();

  //shuffle the user item rating indexes
  std::shuffle(testUsers.begin(), testUsers.end(), mt);
 
  //check if train matrix items are sorted
  if (!checkIfUISorted(trainMat)) {
    std::cout << "\nTrain matrix is not sorted"  << std::endl;
    exit(0);
  }

  int nTestItems = 0;
  int modelInterPPR = 0;
  double modelInterPPRRMSE = 0;
  
  int modelInterModelPPR = 0;
  double modelInterModelPPRRMSE = 0;
  
  int pprInterModelPPR = 0;
  double pprInterModelPPRRMSE = 0;

  for (int k = 0; 
      k < 5000 && nTestItems < 5000 && k < testUsers.size(); k++) {
    int u = testUsers[k];
    std::vector<int> trainItems;
    for (int ii = trainMat->rowptr[u]; ii < trainMat->rowptr[u+1]; ii++) {
      int item = trainMat->rowind[ii];
      trainItems.push_back(item);
    }
    //sort the train items for binary search 
    //std::sort(trainItems.begin(), trainItems.end())
   
    //run personalized RW on graph w.r.t. user
    itemScorePairs = itemGraphItemScores(u, 
        graphMat, trainMat, lambda, nUsers, nItems, invalidItems);
    //std::sort(itemScorePairs.begin(), itemScorePairs.end(), compPairsIndAsc);
    std::fill(itemScores.begin(), itemScores.end(), 0);
    
    for (auto&& itemScore: itemScorePairs) {
      itemScores[itemScore.first] = itemScore.second;
    }

    std::unordered_set<int> sampItems;
    for (int ii = testMat->rowptr[u]; ii < testMat->rowptr[u+1]; ii++) {
      int testItem = testMat->rowind[ii];
      
      testPredRating = model.estRating(u, testItem);
      testRating = testMat->rowval[ii];
      double se = (testPredRating - testRating)*(testPredRating - testRating);

      //check if in head items
      auto search = headItems.find(testItem);
      if (search != headItems.end()) {
        continue;
      } 
     
      //sample 1000 unrated tail items at random
      sampItems.clear();
      while (sampItems.size() < 1000) {
        int sampItem = itemDist(mt);
        
        if (sampItem == testItem) {
          continue;
        }

        //check if sample item is present in train
        if (std::binary_search(trainItems.begin(), trainItems.end(), sampItem)) {
          continue;
        }
        
        //check if sample item is invalid
        search = invalidItems.find(sampItem);
        if (search != invalidItems.end()) {
          //found n skip
          continue;
        }

        //check if sample item is head
        search = headItems.find(sampItem);
        if (search != headItems.end()) {
          //found n skip
          continue;
        }

        sampItems.insert(sampItem);
      }


      bool isModHit = false;
      if (isModelHit(model, sampItems, u, testItem, N)) {
        rec += 1;
        modelRMSE += se;
        isModHit = true;
      }
      
      bool isModPPRHit = false;
      if (isModelLocalScoreHit(model, sampItems, u, testItem, itemScores, N)) {
        localRec += 1;
        modelLocalRMSE += se;
        isModPPRHit = true;
      }
      
      bool isPPRHit = false;
      if (isLocalScoreHit(sampItems, u, testItem, itemScores, N)) {
        pprRec += 1;
        localRMSE += se;
        isPPRHit = true;
      }

      if (isModHit && isPPRHit) {
        modelInterPPR++;
        modelInterPPRRMSE += se;
      }
      
      if (isModHit && isModPPRHit) {
        modelInterModelPPR++;
        modelInterModelPPRRMSE += se;
      }

      if (isPPRHit && isModPPRHit) {
        pprInterModelPPR++;
        pprInterModelPPRRMSE += se;
      }
  
      nTestItems++;
    }
    
    if (k % 500 == 0 || nTestItems % 500 == 0) {
      opFile << "Done.. " << k << std::endl;
      
      opFile << "nTestItems: " << nTestItems << std::endl;
      
      opFile << "Model hits: " << rec << std::endl;
      opFile << "Model RMSE: " << sqrt(modelRMSE/rec) << std::endl;

      opFile << "Model+PPR hits: " << localRec << std::endl;
      opFile << "Model PPR RMSE: " << sqrt(modelLocalRMSE/localRec) << std::endl;
      
      opFile << "PPR hits: " << pprRec << std::endl;
      opFile << "PPR RMSE: " << sqrt(localRMSE/pprRec) << std::endl;
      
      opFile << "Model inter PPR count: " << modelInterPPR << std::endl;
      opFile << "Model inter PPR RMSE: " 
        << sqrt(modelInterPPRRMSE/modelInterPPR) << std::endl;
     
      opFile << "Model inter Model+PPR count: " << modelInterModelPPR << std::endl;
      opFile << "Model inter Model+PPR RMSE: " 
        << sqrt(modelInterModelPPRRMSE/modelInterModelPPR) << std::endl;

      opFile << "PPR inter Model+PPR count: " << pprInterModelPPR << std::endl;
      opFile << "PPR inter Model+PPR RMSE: " 
        << sqrt(pprInterModelPPRRMSE/pprInterModelPPR) << std::endl;

      opFile << "Top-" << N << " Model recall: " 
        << rec/nTestItems << std::endl;
      opFile << "Top-" << N << " Model+PPR recall: " 
        << localRec/nTestItems << std::endl;
      opFile << "Top-" << N << " PPR recall: " 
        << pprRec/nTestItems << std::endl;
    } 
  }
  
  opFile << "nTestItems: " << nTestItems << std::endl;
      
  opFile << "Model hits: " << rec << std::endl;
  opFile << "Model RMSE: " << sqrt(modelRMSE/rec) << std::endl;

  opFile << "Model+PPR hits: " << localRec << std::endl;
  opFile << "Model PPR RMSE: " << sqrt(modelLocalRMSE/localRec) << std::endl;
  
  opFile << "PPR hits: " << pprRec << std::endl;
  opFile << "PPR RMSE: " << sqrt(localRMSE/pprRec) << std::endl;
  
  opFile << "Model inter PPR count: " << modelInterPPR << std::endl;
  opFile << "Model inter PPR RMSE: " 
    << sqrt(modelInterPPRRMSE/modelInterPPR) << std::endl;
 
  opFile << "Model inter Model+PPR count: " << modelInterModelPPR << std::endl;
  opFile << "Model inter Model+PPR RMSE: " 
    << sqrt(modelInterModelPPRRMSE/modelInterModelPPR) << std::endl;

  opFile << "PPR inter Model+PPR count: " << pprInterModelPPR << std::endl;
  opFile << "PPR inter Model+PPR RMSE: " 
    << sqrt(pprInterModelPPRRMSE/pprInterModelPPR) << std::endl;

  rec          = rec/nTestItems;
  localRec     = localRec/nTestItems;
  pprRec       = pprRec/nTestItems;
  
  opFile << "Top-" << N << " model recall: " 
    << rec << std::endl;
  opFile << "Top-" << N << " model local recall: " 
    << localRec << std::endl;
  opFile << "Top-" << N << " ppr recall: " 
    << pprRec << std::endl;

  opFile.close();
}


void writeTestMat(std::vector<std::tuple<int, int, float>>& testUIRatings, 
    std::unordered_set<int>& headItems) {
  //sort triplets by user
  auto compareTriplets = [] (std::tuple<int, int, float> a, 
      std::tuple<int, int, float> b) {
    return std::get<0>(a) < std::get<0>(b);
  };

  std::sort(testUIRatings.begin(), testUIRatings.end(), compareTriplets);

  std::ofstream opFile("test.slim.0.2.csr");

  int uStart = 0;
  int lastU = 0;

  for (auto&& uiRating : testUIRatings) {
    
    int u        = std::get<0>(uiRating);
    int item     = std::get<1>(uiRating);
    float rating = std::get<2>(uiRating);

    while (uStart < u) {
      opFile << std::endl;
      uStart++;
    }

    opFile << item << " "  << rating << " ";
    
    lastU = u;
  }
  
  opFile << std::endl;

  std::cout << "\nlast User: " << lastU << std::endl;

  opFile.close();
}


void topNRecTailWSVD(Model& model, Model& svdModel, gk_csr_t *trainMat, 
    gk_csr_t *testMat, gk_csr_t *graphMat, float lambda,
    std::unordered_set<int>& invalidItems, std::unordered_set<int>& invalidUsers,
    float headPc, int N, int seed, std::string opFileName) {

  enum {MF, PPR, SVD, MFPPR, SVDPPR, MFSVD};
  const int NMETH = 6;
  const int MAXTESTITEMS = 5000;
  const int MAXTESTUSERS = 5000;
  const int SAMPITEMSZ = 1000;

  float testPredModelRating = 0, testRating = 0, testPredSVDRating = 0;
  float testPredModelMean = 0, testPredSVDMean = 0;
  int nItems = trainMat->ncols;
  int nUsers = trainMat->nrows;
  int nSampItems = SAMPITEMSZ;
  int nTailItems;

  std::vector<bool> hitFlags(NMETH, false);
  std::vector<std::vector<double>> counts(NMETH, std::vector<double>(NMETH, 0));
  std::vector<std::vector<double>> rmses(NMETH, std::vector<double>(NMETH, 0.0));

  std::unordered_set<int> headItems = getHeadItems(trainMat, headPc);
  
  std::ofstream opFile(opFileName);
  
  nTailItems = nItems - invalidItems.size() - headItems.size();
  if (SAMPITEMSZ > nTailItems) {
    nSampItems = nTailItems;
  }

  opFile << "\nNo. of head items: " << headItems.size() << " head items pc: " 
    << ((float)headItems.size()/(trainMat->ncols)) << std::endl;
  opFile << "No. of tail items: " << nTailItems << std::endl;
  opFile << "nSampItems: " << nSampItems << std::endl; 

  opFile << "lambda: " << lambda << " N: " << N << " seed: " << seed 
    << std::endl;

  //initialize random engine
  std::mt19937 mt(seed);
  //item distribution to sample items
  std::uniform_int_distribution<int> itemDist(0, nItems-1);
  std::uniform_int_distribution<int> userDist(0, nUsers-1);

  std::vector<std::pair<int, double>> itemScorePairs; 
  std::vector<double> itemScores(nItems, 0); 

  std::vector<int> testUsers;
  for (int u = 0; u < testMat->nrows; u++) {
    //check if invalid users
    auto search = invalidUsers.find(u);
    if (search != invalidUsers.end()) {
      //found n skip
      continue;
    }
    for (int ii = testMat->rowptr[u]; ii < testMat->rowptr[u+1]; ii++) {
      int item = testMat->rowind[ii];
      auto search = headItems.find(item);
      if (search == headItems.end()) {
        //tail item
        testUsers.push_back(u);
        break;
      }
    }
  }

  opFile << "No. of test users: " << testUsers.size() << std::endl;
  
  //shuffle the user item rating indexes
  std::shuffle(testUsers.begin(), testUsers.end(), mt);
 
  //check if train matrix items are sorted
  if (!checkIfUISorted(trainMat)) {
    std::cout << "\nTrain matrix is not sorted"  << std::endl;
    exit(0);
  }

  int nTestItems = 0;
  double testRMSE = 0.0;  
  std::vector<std::tuple<int, int, float>> testUIRatings;

  for (int k = 0; 
      k < MAXTESTUSERS && nTestItems < MAXTESTITEMS && k < testUsers.size(); k++) {
    int u = testUsers[k];
    std::vector<int> trainItems;
    for (int ii = trainMat->rowptr[u]; ii < trainMat->rowptr[u+1]; ii++) {
      int item = trainMat->rowind[ii];
      trainItems.push_back(item);
    }
    //sort the train items for binary search 
    //std::sort(trainItems.begin(), trainItems.end())
    
    
    //run personalized RW on graph w.r.t. user
    itemScorePairs = itemGraphItemScores(u, 
        graphMat, trainMat, lambda, nUsers, nItems, invalidItems);
    //std::sort(itemScorePairs.begin(), itemScorePairs.end(), compPairsIndAsc);
    std::fill(itemScores.begin(), itemScores.end(), 0);
    
    for (auto&& itemScore: itemScorePairs) {
      itemScores[itemScore.first] = itemScore.second;
    }
    

    std::unordered_set<int> sampItems;
    for (int ii = testMat->rowptr[u]; 
        ii < testMat->rowptr[u+1] && nTestItems < MAXTESTITEMS; ii++) {
      int testItem = testMat->rowind[ii];
      
      testPredModelRating = model.estRating(u, testItem);
      testPredModelMean += testPredModelRating;
      testPredSVDRating = svdModel.estRating(u, testItem);
      testPredSVDMean += testPredSVDRating;
      testRating = testMat->rowval[ii];
      double se = (testPredModelRating - testRating)*(testPredModelRating - testRating);
      
      testRMSE += se;

      //check if in head items
      auto search = headItems.find(testItem);
      if (search != headItems.end()) {
        continue;
      } 
      
      testUIRatings.push_back(std::make_tuple(u, testItem, testRating));
      
      //sample unrated tail items at random
      sampItems.clear();
      int insItem = 0;
      while (sampItems.size() < nSampItems && insItem < nItems) {
        int sampItem;
        if (nSampItems < SAMPITEMSZ) {
          sampItem = insItem++;
        } else {
          sampItem = itemDist(mt); 
        }
        
        if (sampItem == testItem) {
          continue;
        }

        //check if sample item is present in train
        if (std::binary_search(trainItems.begin(), trainItems.end(), sampItem)) {
          continue;
        }
        
        //check if sample item is invalid
        search = invalidItems.find(sampItem);
        if (search != invalidItems.end()) {
          //found n skip
          continue;
        }
        
        //check if sample item is head
        search = headItems.find(sampItem);
        if (search != headItems.end()) {
          //found n skip
          continue;
        }
        
        sampItems.insert(sampItem);
      }

      std::fill(hitFlags.begin(), hitFlags.end(), false);

      if (isModelHit(model, sampItems, u, testItem, N)) { 
        hitFlags[MF] = true;
      }
      
      if (isModelHit(svdModel, sampItems, u, testItem, N)) {
        hitFlags[SVD] = true;
      }

      if (isModelModelHit(model, svdModel, sampItems, u, testItem, N)) {
        hitFlags[MFSVD] = true;
      }
       
      
      if (isLocalScoreHit(sampItems, u, testItem, itemScores, N)) {
        hitFlags[PPR] = true;
      }

      if (isModelLocalScoreHit(model, sampItems, u, testItem, itemScores, N)) {
        hitFlags[MFPPR] = true;
      }
      
      if (isModelLocalScoreHit(svdModel, sampItems, u, testItem, itemScores, N)) {
        hitFlags[SVDPPR] = true;
      }
       
      
      for (int i = 0; i < NMETH; i++) {
        
        if (hitFlags[i]) {
          counts[i][i] += 1;
          rmses[i][i] += se;
        }

        for (int j = i+1; j < NMETH; j++) {
          if (hitFlags[i] && hitFlags[j]) {
            counts[i][j] += 1;
            counts[j][i] = counts[i][j];
            rmses[i][j] += se;
            rmses[j][i] = rmses[i][j];
          }
        }
      }
       
      nTestItems++;
    }
    
    if (k % 500 == 0 || nTestItems % 500 == 0) {
      opFile << "Done.. " << k << std::endl;
      
      opFile << "nTestItems: " << nTestItems << std::endl;
      opFile << "testRMSE: " << sqrt(testRMSE/nTestItems) << std::endl;

      //write counts
      opFile << "counts: " << std::endl;
      for (int i = 0; i < NMETH; i++) {
        for (int j = 0; j < NMETH; j++) {
          opFile << counts[i][j] << " ";
        }
        opFile << std::endl;
      }

      //write RMSE
      opFile << "RMSEs: " << std::endl;
      for (int i = 0; i < NMETH; i++) {
        for (int j = 0; j < NMETH; j++) {
          opFile << sqrt(rmses[i][j]/counts[i][j]) << " ";
        }
        opFile << std::endl;
      }
      
      //write recall
      opFile << "MF Recall: " << counts[MF][MF]/nTestItems << std::endl;
      opFile << "PPR Recall: " << counts[PPR][PPR]/nTestItems << std::endl;
      opFile << "SVD Recall: " << counts[SVD][SVD]/nTestItems << std::endl;
      opFile << "MF+PPR Recall: " << counts[MFPPR][MFPPR]/nTestItems << std::endl;
      opFile << "SVD+PPR Recall: " << counts[SVDPPR][SVDPPR]/nTestItems << std::endl;
      opFile << "MF+SVD Recall: " << counts[MFSVD][MFSVD]/nTestItems << std::endl;
    }
    
  }
     
  //writeTestMat(testUIRatings, headItems);

  opFile << "nTestItems: " << nTestItems << std::endl;
  opFile << "testRMSE: " << sqrt(testRMSE/nTestItems) << std::endl;
    
  //write counts
  opFile << "counts: " << std::endl;
  for (int i = 0; i < NMETH; i++) {
    for (int j = 0; j < NMETH; j++) {
      opFile << counts[i][j] << " ";
    }
    opFile << std::endl;
  }

  //write RMSE
  opFile << "RMSEs: " << std::endl;
  for (int i = 0; i < NMETH; i++) {
    for (int j = 0; j < NMETH; j++) {
      opFile << sqrt(rmses[i][j]/counts[i][j]) << " ";
    }
    opFile << std::endl;
  }
  
  //write recall
  opFile << "MF Recall: " << counts[MF][MF]/nTestItems << std::endl;
  opFile << "PPR Recall: " << counts[PPR][PPR]/nTestItems << std::endl;
  opFile << "SVD Recall: " << counts[SVD][SVD]/nTestItems << std::endl;
  opFile << "MF+PPR Recall: " << counts[MFPPR][MFPPR]/nTestItems << std::endl;
  opFile << "SVD+PPR Recall: " << counts[SVDPPR][SVDPPR]/nTestItems << std::endl;
  opFile << "MF+SVD Recall: " << counts[MFSVD][MFSVD]/nTestItems << std::endl;
  
  //write RMSEs of predictions
  opFile << "MF Test Mean: " << testPredModelMean/nTestItems << std::endl;
  opFile << "SVD Test Mean: " << testPredSVDMean/nTestItems << std::endl;
    
  opFile.close();
} 




