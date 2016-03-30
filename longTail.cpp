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


bool isLocalScoreHit(Model& model, std::unordered_set<int>& sampItems, int user, 
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
    std::unordered_set<int>& headItems,
    int N, int seed, std::string opFileName) {

  double rec = 0, localRec = 0, localWtRec = 0, pprRec = 0;
  int nItems = trainMat->ncols;
  int nUsers = trainMat->nrows;

  auto rowColFreq = getRowColFreq(trainMat);
  auto userFreq = rowColFreq.first;
  auto itemFreq = rowColFreq.second;

  std::ofstream opFile(opFileName);

  opFile << "\nlambda: " << lambda << " N: " << N << " seed: " << seed 
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
  
  //shuffle the user item rating indexes
  std::shuffle(testUsers.begin(), testUsers.end(), mt);
  
  int nTestItems = 0;
  for (int k = 0; k < 5000 && nTestItems < 5000; k++) {
    int u = testUsers[k];
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
    //std::sort(itemScorePairs.begin(), itemScorePairs.end(), compPairsIndAsc);
    std::fill(itemScores.begin(), itemScores.end(), 0);
    std::fill(itemFreqWtScores.begin(), itemFreqWtScores.end(), 0);
    
    for (auto&& itemScore: itemScorePairs) {
      itemScores[itemScore.first] = itemScore.second;
      if (itemFreq[itemScore.first] > 0) {
        itemFreqWtScores[itemScore.first] = itemScore.second/itemFreq[itemScore.first];
      }
    }

    std::unordered_set<int> sampItems;
    for (int ii = testMat->rowptr[u]; ii < testMat->rowptr[u+1]; ii++) {
      int testItem = testMat->rowind[ii];
    
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
      
      if (isModelHit(model, sampItems, u, testItem, N)) {
        rec += 1;
      }

      //if (isModelLocalIterHit(model, sampItems, u, testItem, itemScores, N)) {
      if (isModelLocalScoreHit(model, sampItems, u, testItem, itemScores, N)) {
        localRec += 1;
      }


      if (isModelLocalScoreHit(model, sampItems, u, testItem, 
      //if (isModelLocalIterHit(model, sampItems, u, testItem, 
            itemFreqWtScores, N)) {
        localWtRec += 1;
      }

      if (isLocalScoreHit(model, sampItems, u, testItem, itemScores, N)) {
        pprRec += 1;
      }

      nTestItems++;
    }
    
    if (k % 500 == 0 || nTestItems % 500 == 0) {
      opFile << "Done.. " << k << std::endl;
      
      opFile << "nTestItems: " << nTestItems << std::endl;

      opFile << "Top-" << N << " model recall: " 
        << rec/nTestItems << std::endl;
      opFile << "Top-" << N << " model local recall: " 
        << localRec/nTestItems << std::endl;
      opFile << "Top-" << N << " model local wt recall: " 
        << localWtRec/nTestItems << std::endl;
      opFile << "Top-" << N << " ppr recall: " 
        << pprRec/nTestItems << std::endl;
    } 
  }
  
  rec          = rec/nTestItems;
  localRec     = localRec/nTestItems;
  localWtRec   = localWtRec/nTestItems;
  pprRec       = pprRec/nTestItems;

  opFile << "nTestItems: " << nTestItems << std::endl;
  
  opFile << "Top-" << N << " model recall: " 
    << rec << std::endl;
  opFile << "Top-" << N << " model local recall: " 
    << localRec << std::endl;
  opFile << "Top-" << N << " model local wt recall: " 
    << localWtRec << std::endl;
  opFile << "Top-" << N << " ppr recall: " 
    << pprRec << std::endl;

  opFile.close();
}







