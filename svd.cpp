#include "svd.h" 

void svdFrmCSR(gk_csr_t *mat, int rank, std::vector<std::vector<double>>& uFac,
                std::vector<std::vector<double>>& iFac) {

  int item;
  float rat;
  //create dense matrix from sparse
  Eigen::MatrixXf denseMat(mat->nrows, mat->ncols);
  
  for (int u = 0; u < mat->nrows; u++) {
    for (int ii = mat->rowptr[u]; ii < mat->rowptr[u+1]; ii++) {
      item = mat->rowind[ii];
      rat = mat->rowval[ii];
      denseMat(u, item) = rat;
    }
  }
  
  //compute thin svd
  Eigen::JacobiSVD<Eigen::MatrixXf> svd(denseMat, Eigen::ComputeThinU|Eigen::ComputeThinV);
  std::cout <<"\nRank using svd: " << svd.rank();

  //copy top-rank left singular vectors to uFac
  /*
  auto thinU = svd.matrixU(); 
  for (int u = 0; u < mat->nrows; u++) {
    for (int k = 0; k < rank; k++) {
      uFac[u][k] = thinU(u, k);
    }
  }

  //copy top-rank right singular vectors to iFac
  auto thinV = svd.matrixV();
  for (int item = 0; item < mat->ncols; item++) {
    for (int k = 0; k < rank; k++) {
      iFac[item][k] = thinV(item, k);
    }
  }
  */
}




