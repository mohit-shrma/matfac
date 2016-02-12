#include "util.h"


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



void getInvalidUsersItems(gk_csr_t *mat, 
    std::vector<std::unordered_set<int>>& uISetIgnore,
    std::unordered_set<int>& invalidUsers,
    std::unordered_set<int>& invalidItems) {
  
  std::vector<int> uItemCount (mat->nrows, 0);
  std::vector<int> iUserCount (mat->ncols, 0);
  for (int u = 0; u < mat->nrows; u++) {
    for (int ii = 0; ii < mat->rowptr[u]; ii++) {
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


