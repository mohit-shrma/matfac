#include <vector>
#include <tuple>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <algorithm>
#include "GKlib.h"



std::vector<std::pair<int,float>> getSortedItems(int u, gk_csr_t *mat, 
    float lambda, int max_niter, int nUsers, int nItems) {
 
  //custom compare function to sort vector of pairs in decreasing order
  auto comparePair = [](std::pair<int, float> a, std::pair<int, float> b) { 
    return a.second > b.second; 
  };

  float *pr = (float*)malloc(sizeof(float)*mat->nrows);
  memset(pr, 0, sizeof(float)*mat->nrows);
  pr[u] = 1.0;
  
  //run personalized page rank on the graph w.r.t. u
  gk_rw_PageRank(mat, lambda, 0.0001, max_niter, pr);
 
  //get pr score of items
  std::vector<std::pair<int, float>> itemScores;
  for (int i = nUsers; i < nUsers + nItems; i++) {
    itemScores.push_back(std::make_pair(i, pr[i]));
  }
  
  //sort items by their pr scores in decreasing order
  std::sort(itemScores.begin(), itemScores.end(), comparePair);

  free(pr);

  return itemScores;
}


void writeSortedItemsByPR(gk_csr_t *mat, int nUsers, int nItems, float lambda,
    int max_niter) {
  std::ofstream opFile("prItems.txt");
  if (opFile.is_open()) {
    for (int u = 0; u < nUsers; u++) {
      std::vector<std::pair<int, float>> itemScores = getSortedItems(u, mat,
          lambda, max_niter, nUsers, nItems);
      if (itemScores.size() != nItems) {
        std::cerr << "\nscore not computed for suff items." << std::endl;
      }
      for (int i = 0; i < nItems; i++) {
        opFile << itemScores[i].first << " " << itemScores[i].second << " ";
      }
      opFile << std::endl;
      if (u % 500 == 0) {
        std::cout << "\nDone users " << u << "..." << std::endl;
      }
    }
    opFile.close();
  }
}


int main(int argc, char* argv[]) {

  if (argc < 6) {
    std::cerr << "\nInsufficient arguments" << std::endl;
    exit(0);
  }

  char* ipCSRAdj = argv[1];
  int nUsers     = atoi(argv[2]);
  int nItems     = atoi(argv[3]);
  float lambda   = atof(argv[4]);
  int max_niter  = atoi(argv[5]);

  gk_csr_t *graphMat = gk_csr_Read(ipCSRAdj, GK_CSR_FMT_CSR, 0, 0);

  std::cout << "\nnrows: " << graphMat->nrows;
  std::cout << "\nncols: " << graphMat->ncols;
  std::cout << "\nnUsers: " << nUsers;
  std::cout << "\nnItems: " << nItems;

  writeSortedItemsByPR(graphMat, nUsers, nItems, lambda, max_niter);

  gk_csr_Free(&graphMat);
  return 0;
}




