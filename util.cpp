#include "util.h"


std::unordered_set<int> getHeadItems(gk_csr_t *mat) {
  
  //get number of ratings per user and item, i.e. frequency
  auto rowColFreq = getRowColFreq(mat);
  auto userFreq = rowColFreq.first;
  auto itemFreq = rowColFreq.second;
  double nRatings  = 0;

  std::vector<std::pair<int, double>> itemFreqPairs;
  for (int i = 0; i < itemFreq.size(); i++) {
    itemFreqPairs.push_back(std::make_pair(i, itemFreq[i]));
    nRatings += itemFreq[i];
  }
  std::sort(itemFreqPairs.begin(), itemFreqPairs.end(), descComp);
  
  //find head/popular items responsible for 20% of ratings
  double headRatings = 0;
  std::unordered_set<int> headItems;
  for (int i = 0; i < itemFreqPairs.size(); i++) {
    if (headRatings/nRatings >= 0.2) {
      break;
    }
    headRatings += itemFreqPairs[i].second;
    headItems.insert(itemFreqPairs[i].first);
  }
  
  std::cout << "\nHead ratings: " << headRatings << " nRatings: " << nRatings 
    << std::endl;

  return headItems;
}


double compRecall(std::vector<int> order1, std::vector<int> order2, int N) {

  //change N to smaller of the list, in case N is larger than the either one
  N = N < order1.size()? N : order1.size();
  N = N < order2.size()? N : order2.size();

  std::unordered_set<int> set1;
  for (int i  = 0; i < N; i++) {
    set1.insert(order1[i]);
  }

  std::unordered_set<int> set2;
  for (int i  = 0; i < N; i++) {
    set2.insert(order2[i]);
  }

  //compute overlap
  double overlap = 0;
  for (int item: set1) {
    auto search = set2.find(item);
    if (search != set2.end()) {
      //found item
      overlap += 1;
    }
  }
  
  return overlap/N;
}


double meanRating(gk_csr_t* mat) {
  int u, ii, nnz;
  double avg = 0;
  nnz = 0;
  for (u = 0; u < mat->nrows; u++) {
    for (ii = mat->rowptr[u]; ii < mat->rowptr[u+1]; ii++) {
      avg += mat->rowval[ii];
      nnz++;
    }
  }
  avg = avg/nnz;
  return avg;
}


std::pair<double, double> getMeanVar(std::vector<std::vector<double>> uFac,
    std::vector<std::vector<double>> iFac, int facDim, int nUsers, int nItems) {
  
  double mean = 0, var = 0, diff = 0;
  
  for (int u = 0; u < nUsers; u++) {
    for (int item = 0; item < nItems; item++) {
      mean += dotProd(uFac[u], iFac[item], facDim);
    }
  }
  mean = mean/(nItems*nUsers);

  for(int u = 0; u < nUsers; u++) {
    for (int item = 0; item < nItems; item++) {
      diff = dotProd(uFac[u], iFac[item], facDim) - mean;
      var += diff*diff;
    }
  }
  var = var/((nItems*nUsers) - 1);

  return std::make_pair(mean, var);
}


//include start exclude end
int nnzSubMat(gk_csr_t *mat, int uStart, int uEnd, int iStart, int iEnd) {
  
  int u, ii, item;
  int nnz = 0;

  for (u = uStart; u < uEnd; u++) {
    for (ii = mat->rowptr[u]; ii < mat->rowptr[u+1]; ii++) {
      item = mat->rowind[ii];
      if (item >= iStart && item < iEnd) {
        nnz++;
       }
     }
  } 

  return nnz;
}


int getNNZ(gk_csr_t *mat) {
  int nnz = 0;
  for (int u = 0; u < mat->nrows; u++) {
    nnz += mat->rowptr[u+1] - mat->rowptr[u];
  }
  return nnz;
}


//check if (u, item) is present inside the passed block
//includes start but exclude end
bool isInsideBlock(int u, int item, int uStart, int uEnd, int iStart, 
    int iEnd) {
  if ((u >= uStart && u < uEnd) && (item >= iStart && item < iEnd)) {
    return true;
  } else {
    return false;
  }
}


//compute standard deviation in vector
double stddev(std::vector<double> v) {
  
  double sum = 0;
  for (int i = 0; i < v.size(); i++) {
    sum += v[i];
  }
  double mean = sum / v.size();
  
  double sq_sum = 0;
  //subtract mean, compute square sum
  for (int i = 0; i < v.size(); i++) {
    sq_sum += (v[i] - mean)*(v[i] - mean);
  }

  double stdev = sqrt(sq_sum / v.size());
  return stdev;
}


//return a min element of a vector
template<typename T>
T minVec(std::vector<T> v) {
  typename std::vector<T>::iterator result = std::min_element(std::begin(v), std::end(v));
  int minInd = std::distance(std::begin(v), result); 
  return v[minInd];
}


template<typename T>
T maxVec(std::vector<T> v) {
  typename std::vector<T>::iterator result = std::max_element(std::begin(v), std::end(v));
  int maxInd = std::distance(std::begin(v), result); 
  return v[maxInd];
}


//compute min ratngs per user/ item, also can pass set of ignored user-item
//pairs
void genStats(gk_csr_t *mat, 
    std::vector<std::unordered_set<int>> uISetIgnore,
    std::string opPrefix) {
  
  int nUsers = mat->nrows;
  int nItems = mat->ncols;
  int nnz = 0, igNNZ = 0;
  std::vector<int> uItemCount(nUsers, 0);
  std::vector<int> uItemIgCount(nUsers, 0);
  std::vector<int> iUserCount(nItems, 0);
  std::vector<int> iUserIgCount(nItems, 0);

  for  (int u = 0; u < mat->nrows; u++) {
    for (int ii = mat->rowptr[u]; ii < mat->rowptr[u+1]; ii++) {
      int item = mat->rowind[ii];
      uItemCount[u] += 1;
      iUserCount[item] += 1;
      nnz++;
      //check if given user item pair should be ignore
      auto search = uISetIgnore[u].find(item);
      if (search == uISetIgnore[u].end()) {
        //not found in ignore set
        uItemIgCount[u] += 1;
        iUserIgCount[item] += 1;
      } else {
        igNNZ++;
       }
    } 
  }
  
  int minUserRatCount = minVec(uItemCount);
  int maxUserRatCount = maxVec(uItemCount);
  int minItemRatCount = minVec(iUserCount);
  int maxItemRatCount = maxVec(iUserCount);

  std::cout << "\nnUsers: " << nUsers;
  std::cout << "\nnItems: " << nItems;
  std::cout << "\nNNZ: " << nnz;
  std::cout << "\nmin nratings per user: " << minUserRatCount 
    << " opPrefix: " << opPrefix;
  std::cout << "\nmax nratings per user: " << maxUserRatCount 
    << " opPrefix: " << opPrefix;
  std::cout << "\nmin nratings per item: " << minItemRatCount 
    << " opPrefix: " << opPrefix;
  std::cout << "\nmax nratings per item: " << maxItemRatCount 
    << " opPrefix: " << opPrefix;

  int minUserIgRatCount = minVec(uItemIgCount);
  int maxUserIgRatCount = maxVec(uItemIgCount);
  int minItemIgRatCount = minVec(iUserIgCount);
  int maxItemIgRatCount = maxVec(iUserIgCount);

  std::cout << "\nmin nratings per user after ig: " << minUserIgRatCount 
    << " opPrefix: " << opPrefix;
  std::cout << "\nmax nratings per user after ig: " << maxUserIgRatCount 
    << " opPrefix: " << opPrefix;
  std::cout << "\nmin nratings per item after ig: " << minItemIgRatCount 
    << " opPrefix: " << opPrefix;
  std::cout << "\nmax nratings per item after ig: " << maxItemIgRatCount 
    << " opPrefix: " << opPrefix;
  
  int nUsersWithMinRatcount = 0;
  int nUsersWithMaxRatcount = 0;
  for (int i = 0; i < uItemIgCount.size(); i++) {
    if (uItemIgCount[i] == minUserIgRatCount) {
      nUsersWithMinRatcount++;
    }
    if (uItemIgCount[i] == maxUserIgRatCount) {
      nUsersWithMaxRatcount++;
    }
  }

  std::cout << "\nnUsers with minRatcount(" << minUserIgRatCount << "): " 
    << nUsersWithMinRatcount << " opPrefix: " << opPrefix;
  std::cout << "\nnUsers with maxRatcount(" << maxUserIgRatCount << "): " 
    << nUsersWithMaxRatcount << " opPrefix: " << opPrefix;
  
  int nItemsWithMinRatCount = 0;
  int nItemsWithMaxRatcount = 0;
  for (int i = 0; i < iUserIgCount.size(); i++) {
    if (iUserIgCount[i] == minItemIgRatCount) {
      nItemsWithMinRatCount++;
    }
    if (iUserIgCount[i] == maxItemIgRatCount) {
      nItemsWithMaxRatcount++;
    }
  }

  std::cout << "\nnItems with minRatCount(" << minItemIgRatCount << "): "
    << nItemsWithMinRatCount << " opPrefix: " << opPrefix;
  std::cout << "\nnItems with maxRatCount(" << maxItemIgRatCount << "): "
    << nItemsWithMaxRatcount << " opPrefix: " << opPrefix;
}


void getUserStats(std::vector<int>& users, gk_csr_t* mat, 
    std::unordered_set<int>& filtItems, const char* opFName) {

  auto rowColFreq = getRowColFreq(mat);
  auto userFreq = rowColFreq.first;
  auto itemFreq = rowColFreq.second;

  std::vector<std::pair<int, double>> itemFreqP;
  for (int i = 0; i < itemFreq.size(); i++) {
    itemFreqP.push_back(std::make_pair(i, itemFreq[i]));
  }
  std::sort(itemFreqP.begin(), itemFreqP.end(), descComp);

  std::unordered_set<int> top500Items;
  for (int i = 0; i < 500; i++) {
    top500Items.insert(itemFreqP[i].first);
  }

  std::ofstream opFile(opFName); 
  
  for (auto&& user: users) {
    
    int nUserItems = 0;
    std::unordered_set<int>  items;
    for (int ii = mat->rowptr[user]; 
        ii < mat->rowptr[user+1]; ii++) {
      int item = mat->rowind[ii];
      auto search = filtItems.find(item);
      if (search != filtItems.end()) {
        //found n skip
        continue;
      }
      items.insert(item);
      nUserItems += 1;
    }
   
    //no. of users that rated the items rated by user
    std::unordered_set<int> adjUsers;
    //mean freq of items
    float meanFreq = 0;
    int top500Count = 0;
    for (auto&& item: items) {
      for (int jj = mat->colptr[item]; jj < mat->colptr[item+1];
          jj++) {
        int u = mat->colind[jj];
        adjUsers.insert(u);
      }
      meanFreq += itemFreq[item];
      auto search = top500Items.find(item);
      if (search != top500Items.end()) {
        //found in top 500 items
        top500Count++;
      }
    }
    meanFreq = meanFreq/items.size();

    //analyze adjUsers
    double meanAdjUItems = 0;
    double meanAdjUItemsFreq = 0;
    int nMeanAdjUItems = 0;
    int adjTop500Count = 0;
    for (auto&& adjUser: adjUsers) {
      meanAdjUItems += userFreq[adjUser];
      for (int ii = mat->rowptr[adjUser]; ii < mat->rowptr[adjUser+1]; ii++) {
        int item = mat->rowind[ii];
        auto search = top500Items.find(item);
        if (search != top500Items.end()) {
          adjTop500Count++;
        }
        meanAdjUItemsFreq += itemFreq[item];
        nMeanAdjUItems++;
      }
    }
    meanAdjUItems = meanAdjUItems/adjUsers.size();
    meanAdjUItemsFreq = meanAdjUItemsFreq/nMeanAdjUItems;

    opFile << user << " " << nUserItems << " " << adjUsers.size() << " " 
      << meanFreq << " " << top500Count  << " " << meanAdjUItems  << " "
      << meanAdjUItemsFreq << " " << adjTop500Count/adjUsers.size() 
      << std::endl;
  }

  opFile.close();
}


std::vector<int> getInvalidUsers(gk_csr_t *mat) {
  std::vector<int> invalUsers;
  for (int u = 0; u < mat->nrows; u++) {
    if (mat->rowptr[u+1] - mat->rowptr[u] == 0) {
      invalUsers.push_back(u);
    }
  }
  return invalUsers;
}


void getInvalidUsersItems(gk_csr_t *mat, 
    std::vector<std::unordered_set<int>>& uISetIgnore,
    std::unordered_set<int>& invalidUsers,
    std::unordered_set<int>& invalidItems) {
  
  std::vector<int> uItemCount (mat->nrows, 0);
  std::vector<int> iUserCount (mat->ncols, 0);
  for (int u = 0; u < mat->nrows; u++) {
    for (int ii = mat->rowptr[u]; ii < mat->rowptr[u+1]; ii++) {
      int item = mat->rowind[ii];
      //check if not in ignore u, item pair
      auto search = uISetIgnore[u].find(item);
      if (search == uISetIgnore[u].end()) {
        //not found in ignored pairs
        uItemCount[u] += 1;
        iUserCount[item] += 1;
      }
    }
  }

  //find the users with no ratings
  for (int u = 0; u < mat->nrows; u++) {
    if (0 == uItemCount[u]) {
      invalidUsers.insert(u);
    }
  }

  //find the items with no ratings
  for (int item = 0; item < mat->ncols; item++) {
    if (0 == iUserCount[item]) {
      invalidItems.insert(item);
    }
  }
}


std::pair<std::vector<double>, std::vector<double>> getRowColFreq(gk_csr_t *mat) {
  
  std::vector<double> rowFreq(mat->nrows, 0);
  std::vector<double> colFreq(mat->ncols, 0);

  for (int u = 0; u < mat->nrows; u++) {
    for (int ii = mat->rowptr[u]; ii < mat->rowptr[u+1]; ii++) {
      int item = mat->rowind[ii];
      rowFreq[u] += 1;
      colFreq[item] += 1;
    }
  }
  
  return std::make_pair(rowFreq, colFreq);
}


std::vector<std::pair<int, int>> getUIPairs(gk_csr_t *mat, 
    std::unordered_set<int>& invalidUsers, 
    std::unordered_set<int>& invalidItems) {

  std::vector<std::pair<int, int>> uiPairs;
  for (int u = 0; u < mat->nrows; u++) {
    //check if invalid user
    auto search = invalidUsers.find(u);
    if (search != invalidUsers.end()) {
      //found n skip
      continue;
    }
    
    for (int ii = mat->rowptr[u]; ii < mat->rowptr[u+1]; ii++) {
      int item = mat->rowind[ii];
      search = invalidItems.find(item);
      if (search != invalidItems.end()) {
        //found n skip
        continue;
      }
      uiPairs.push_back(std::make_pair(u, item)); 
    }
  }

  return uiPairs;
}


std::vector<std::tuple<int, int, float>> getUIRatings(gk_csr_t* mat) {
  std::vector<std::tuple<int, int, float>> uiRatings;
  for (int u = 0; u < mat->nrows; u++) {
    for (int ii = mat->rowptr[u]; ii < mat->rowptr[u+1]; ii++) {
      int item = mat->rowind[ii];
      float rating = mat->rowval[ii];
      uiRatings.push_back(std::make_tuple(u, item, rating));
    }
  }
  return uiRatings;
}


std::vector<std::tuple<int, int, float>> getUIRatings(gk_csr_t* mat, 
    std::unordered_set<int>& invalidUsers, 
    std::unordered_set<int>& invalidItems) {
  std::vector<std::tuple<int, int, float>> uiRatings;
  for (int u = 0; u < mat->nrows; u++) {
    //skip if in invalid users
    auto search = invalidUsers.find(u);
    if (search != invalidUsers.end()) {
      //found and skip
      continue;
    }
    
    for (int ii = mat->rowptr[u]; ii < mat->rowptr[u+1]; ii++) {
      int item = mat->rowind[ii];
      //skip if in invalid items
      search = invalidItems.find(item);
      if (search != invalidItems.end()) {
        //found and skip
        continue;
      }
      float rating = mat->rowval[ii];
      uiRatings.push_back(std::make_tuple(u, item, rating));
    }
  }
  return uiRatings;
}


double normVec(std::vector<double>& vec) {
  double norm = 0;
  for (auto v: vec) {
    norm += v*v;
  }
  norm = sqrt(norm);
  return norm;
}


bool descComp(std::pair<int, double>& a, std::pair<int, double>& b) {
  return a.second > b.second;
}


float sparseRowDotProd(gk_csr_t* mat1, int i, gk_csr_t* mat2, int j) {
  float sim = 0;
  for (int ii = mat1->rowptr[i]; ii < mat1->rowptr[i+1]; ii++) {
    
    int ind1   = mat1->rowind[ii];
    float val1 = mat1->rowval[ii];

    for (int ii2 = mat2->rowptr[j]; ii2 < mat2->rowptr[j+1]; ii2++) {
      
      int ind2   = mat2->rowind[ii2];
      float val2 = mat2->rowval[ii2];
      
      if (ind1 == ind2 ) {
        sim += val1*val2;
        break;
      }

    }
  } 
  return sim;
}


float sparseColDotProd(gk_csr_t* mat1, int i, gk_csr_t* mat2, int j) {
  float sim = 0;
  for (int jj1 = mat1->colptr[i]; jj1 < mat1->colptr[i+1]; jj1++) {
    
    int ind1   = mat1->colind[jj1];
    float val1 = mat1->colval[jj1];

    for (int jj2 = mat2->colptr[j]; jj2 < mat2->colptr[j+1]; jj2++) {
      
      int ind2   = mat2->colind[jj2];
      float val2 = mat2->colval[jj2];
      
      if (ind1 == ind2 ) {
        sim += val1*val2;
        break;
      }

    }
  } 
  return sim;
}


//return 1 for atleast one corated user
int sparseBinColDotProd(gk_csr_t* mat1, int i, gk_csr_t* mat2, int j) {
  for (int jj1 = mat1->colptr[i]; jj1 < mat1->colptr[i+1]; jj1++) {
    int ind1   = mat1->colind[jj1];
    for (int jj2 = mat2->colptr[j]; jj2 < mat2->colptr[j+1]; jj2++) {
      int ind2   = mat2->colind[jj2];
      if (ind1 == ind2 ) {
        return 1;
      }
    }
  } 
  return 0;
}


//return no. of co-rated users for the items
int sparseCoRatedUsers(gk_csr_t* mat, int i, int j) {
  int coUsers = 0;
  for (int jj1 = mat->colptr[i]; jj1 < mat->colptr[i+1]; jj1++) {
    int ind1   = mat->colind[jj1];
    for (int jj2 = mat->colptr[j]; jj2 < mat->colptr[j+1]; jj2++) {
      int ind2   = mat->colind[jj2];
      if (ind1 == ind2) {
        coUsers++;
        break;
      }
    }
  } 
  return coUsers;
}


int binSearch(int *sortedArr, int key, int ub, int lb) {
  
  int ind = -1;
  
  while (ub >= lb) {
    int midP = (ub + lb) / 2;
    if (sortedArr[midP] == key) {
      ind = midP;
      break;
    } else if (sortedArr[midP] < key) {
      lb = midP + 1;
    } else {
      ub = midP - 1;
    }
  }

  return ind;
}


//return no. of co-rated users for the items
int coRatedUsersFrmSortedMat(gk_csr_t* mat, int i, int j) {
  int coUsers = 0;
  
  //exit if either of item don't have any ratings
  if (mat->colptr[i+1] - mat->colptr[i] == 0 || 
      mat->colptr[j+1] - mat->colptr[j] == 0) {
    return coUsers;
  }

  for (int jj = mat->colptr[i]; jj < mat->colptr[i+1]; jj++) {
    int user   = mat->colind[jj];
    int lb = mat->colptr[j];
    int ub = mat->colptr[j+1];
    if (binSearch(mat->colind, user, ub, lb) != -1) {
      coUsers++;
    }
  } 
  return coUsers;
}


int coRatedUsersFrmSortedMatLinMerge(gk_csr_t* mat, int i, int j) {
  int coUsers = 0;
  
  //exit if either of item don't have any ratings
  if (mat->colptr[i+1] - mat->colptr[i] == 0 || 
      mat->colptr[j+1] - mat->colptr[j] == 0) {
    return coUsers;
  }

  int jj1 = mat->colptr[i]; 
  int jj2 = mat->colptr[j];
  
  while(jj1 < mat->colptr[i+1] && jj2 < mat->colptr[j+1]) {
    int user1 = mat->colind[jj1];
    int user2 = mat->colind[jj2];
    if (user1 == user2) {
      coUsers++;
      jj1++;
      jj2++;
    } else if (user1 < user2) {
      jj1++;
    } else {
      jj2++;
    }
  }

  return coUsers;
}



int checkIfUISorted(gk_csr_t* mat) {
  
  for (int u = 0; u < mat->nrows; u++) {
    int prevItem = mat->rowind[mat->rowptr[u]];
    for (int ii = mat->rowptr[u]; ii < mat->rowptr[u+1]; ii++) {
      int item = mat->rowind[ii];
      if (item < prevItem) {
        std::cout << "\nitem < prevItem: " << u << " " << item << " " 
          << prevItem << std::endl;
        return 0;
      }
      prevItem = item;
    }
  }
  
  for (int item = 0; item < mat->ncols; item++) {
    int prevUser = mat->colind[mat->colptr[item]];
    for (int jj = mat->colptr[item]; jj < mat->colptr[item+1]; jj++) {
      int u = mat->colind[jj];
      if (u < prevUser) {
        std::cout << "\nu < prevUser: " << item << " " << u << " " 
          << prevUser << std::endl;
        return 0;
      }
      prevUser = u;
    }
  }
  
  return 1;
}







