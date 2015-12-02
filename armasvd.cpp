#include "armasvd.h"

void svdFrmCSR(gk_csr_t *mat, int rank, std::vector<std::vector<double>>& uFac,
                std::vector<std::vector<double>>& iFac) {
 
  int u, item;
  int nnz = 1;
  //int nrows = mat->nrows;
  //int ncols = mat->ncols;
  for (int u = 0; u < mat->nrows; u++) {
    nnz += mat->rowptr[u+1] - mat->rowptr[u] + 1;
  }

  std::cout << "\nNNZ = " << nnz;

  arma::uvec rowind(nnz); //of size nnz
  arma::uvec colptr(mat->ncols + 1); //of size n_cols + 1
  arma::vec  values(nnz); //values in the matrix
  
  for (item = 0; item < mat->ncols; item++) {
    colptr[item] = mat->colptr[item];
    for (int jj = mat->colptr[item]; jj < mat->colptr[item+1]; jj++) {
      rowind[jj] = mat->colind[jj];
      values[jj] = mat->colval[jj];
    }
  }
  colptr[item] = mat->colptr[item];

  arma::sp_mat X(rowind, colptr, values, mat->nrows, mat->ncols);
  arma::mat U(mat->nrows, rank, arma::fill::zeros); 
  arma::mat V(mat->ncols, rank, arma::fill::zeros);
  arma::vec s(rank);
  
  //singular value decomposition
  arma::svds(U, s, V, X, rank);
  
  for (u = 0; u < mat->nrows; u++) {
    for (int j = 0; j < rank; j++) {
      uFac[u][j] = U(u,j);
    }
  }
  std::cout << "\nsize of U: " << size(U) << " V: " << size(V) << std::endl;
  
  for (item = 0; item < mat->ncols; item++) {
    for (int j = 0; j < rank; j++) {
      iFac[item][j] = V(item,j);
    }
  }
  
}
