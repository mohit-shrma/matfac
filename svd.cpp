#include "svd.h" 


double computeDiff(Eigen::MatrixXf &denseMat, std::vector<std::vector<double>>& uFac,
    std::vector<std::vector<double>>& iFac, int facDim) {
  double rmse = 0.0, diff;
  for (int u = 0; u < denseMat.rows(); u++) {
    for (int item = 0; item < denseMat.cols(); item++) {
      diff = dotProd(uFac[u], iFac[item], facDim) - denseMat(u,item);
      rmse += diff*diff;
    }
  }
  return rmse;
}


void svdFrmCSR(gk_csr_t *mat, int rank, std::vector<std::vector<double>>& uFac,
                std::vector<std::vector<double>>& iFac) {

  int item;
  float rat;
  //create dense matrix from sparse
  Eigen::MatrixXf denseMat = Eigen::MatrixXf::Zero(mat->nrows, mat->ncols);
  
  for (int u = 0; u < mat->nrows; u++) {
    for (int ii = mat->rowptr[u]; ii < mat->rowptr[u+1]; ii++) {
      item = mat->rowind[ii];
      rat = mat->rowval[ii];
      denseMat(u, item) = rat;
    }
  }
  std::cout << "\nmat nrows: " << mat->nrows << " ncols: " << mat->ncols; 
  std::cout << "\nDense mat nrows: " << denseMat.rows() << " ncols: " << denseMat.cols();
  std::cout << "\nrmse with known fac before svd: " << computeDiff(denseMat, 
      uFac, iFac, rank) << std::endl;

  //compute thin svd
  Eigen::JacobiSVD<Eigen::MatrixXf> svd(denseMat, Eigen::ComputeThinU|Eigen::ComputeThinV);
  std::cout <<"\nRank using svd: " << svd.rank();
  
  std::cout << "\nSingular values: ";
  auto singVals = svd.singularValues();
  for (int i = 0; i < rank; i++) {
    std::cout << singVals[i] << " ";
  }

  //copy top-rank left singular vectors to uFac
  auto thinU = svd.matrixU(); 
  std::cout << "\n U nrows: " << thinU.rows() << " ncols: " << thinU.cols();
  for (int u = 0; u < mat->nrows; u++) {
    for (int k = 0; k < rank; k++) {
      uFac[u][k] = thinU(u, k);
    }
  }

  //copy top-rank right singular vectors to iFac
  auto thinV = svd.matrixV();
  std::cout << "\n v nrows: " << thinV.rows() << " ncols: " << thinV.cols();
  for (int item = 0; item < mat->ncols; item++) {
    for (int k = 0; k < rank; k++) {
      iFac[item][k] = thinV(item, k);
    }
  }
  
  std::cout << "\nrmse with known fac after svd: " << computeDiff(denseMat, 
      uFac, iFac, rank) << std::endl;
  
}



//compute svd of matrix by replacing 0 with the column average or the user
//average
void svdFrmCSRColAvg(gk_csr_t *mat, int rank, std::vector<std::vector<double>>& uFac,
                std::vector<std::vector<double>>& iFac) {

  int item;
  float rat, uAvg;
  
  //create dense matrix from sparse
  Eigen::MatrixXf denseMat = Eigen::MatrixXf::Zero(mat->nrows, mat->ncols);
  
  for (int u = 0; u < mat->nrows; u++) {
    uAvg = 0;
    for (int ii = mat->rowptr[u]; ii < mat->rowptr[u+1]; ii++) {
      item = mat->rowind[ii];
      rat = mat->rowval[ii];
      denseMat(u, item) = rat;
      uAvg += rat;
    }
    uAvg = uAvg/(mat->rowptr[u+1] - mat->rowptr[u]);
    for (item = 0; item < mat->ncols; item++) {
      if (denseMat(u, item) == 0.0) {
        denseMat(u, item) = uAvg;
      }
    }
  }
  std::cout << "\nSVD row avg";
  std::cout << "\nmat nrows: " << mat->nrows << " ncols: " << mat->ncols; 
  std::cout << "\nDense mat nrows: " << denseMat.rows() << " ncols: " << denseMat.cols();
  std::cout << "\nrmse with known fac before svd: " << computeDiff(denseMat, 
      uFac, iFac, rank) << std::endl;

  //compute thin svd
  Eigen::JacobiSVD<Eigen::MatrixXf> svd(denseMat, Eigen::ComputeThinU|Eigen::ComputeThinV);
  std::cout <<"\nRank using svd: " << svd.rank();

  //copy top-rank left singular vectors to uFac
  
  auto thinU = svd.matrixU(); 
  std::cout << "\n U nrows: " << thinU.rows() << " ncols: " << thinU.cols();
  for (int u = 0; u < mat->nrows; u++) {
    for (int k = 0; k < rank; k++) {
      uFac[u][k] = thinU(u, k);
    }
  }

  //copy top-rank right singular vectors to iFac
  auto thinV = svd.matrixV();
  std::cout << "\n v nrows: " << thinV.rows() << " ncols: " << thinV.cols();
  for (int item = 0; item < mat->ncols; item++) {
    for (int k = 0; k < rank; k++) {
      iFac[item][k] = thinV(item, k);
    }
  }
  
  std::cout << "\nrmse with known fac after svd: " << computeDiff(denseMat, 
      uFac, iFac, rank) << std::endl;
  
}

