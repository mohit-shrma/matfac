#include "spectrasvd.h"

void computeTopEig(Eigen::MatrixXd M, int nev, int ncv, 
    std::vector<std::vector<double>> &fac) {
  
  //Construct matrix operation object using the wrapper class DenseGenMatProd
  Spectra::DenseGenMatProd<double> op(M);

  //Construct eigen solver object, requesting the largest three eigenvalues
  Spectra::SymEigsSolver< double, Spectra::LARGEST_ALGE, Spectra::DenseGenMatProd<double> > eigs(&op, nev, ncv);

  //Initialize and compute
  eigs.init();
  int nconv = eigs.compute();

  //retrieve results
  Eigen::VectorXd evalues;
  if(nconv > 0) {
    //evalues = eigs.eigenvalues();
    auto evec = eigs.eigenvectors(nev);
    for (int i = 0; i < evec.rows(); ++i) {
      for (int j = 0; j < nev; j++) {
        fac[i][j] = evec(i, j);
      }
    }
    //copy eigen vectors to fac
    //std::cout  << "fac: " << fac.rows() << " " << fac.cols() << std::endl;
    std::cout << "\nEig vec: " << evec.rows() << " " << evec.cols() << std::endl;
  }

}





void spectraSvdFrmCSR(gk_csr_t *mat, int rank, std::vector<std::vector<double>>& uFac,
                std::vector<std::vector<double>>& iFac) {
  int item;
  float rat;
  //create dense matrix from sparse
  Eigen::MatrixXd A = Eigen::MatrixXd::Zero(mat->nrows, mat->ncols);
  
  for (int u = 0; u < mat->nrows; u++) {
    for (int ii = mat->rowptr[u]; ii < mat->rowptr[u+1]; ii++) {
      item = mat->rowind[ii];
      rat = mat->rowval[ii];
      A(u, item) = rat;
    }
  }
  std::cout << "\nmat nrows: " << mat->nrows << " ncols: " << mat->ncols; 
  std::cout << "\nDense mat nrows: " << A.rows() << " ncols: " << A.cols();

  //compute right singular vectors
  Eigen::MatrixXd ATA = A.transpose() * A;
  computeTopEig(ATA, rank, 2.0*rank, iFac);
  
  //compute left singular vectors
  Eigen::MatrixXd AAT = A*A.transpose();
  computeTopEig(AAT, rank, 2.0*rank, uFac);

}


void spectraSvdFrmCSRColAvg(gk_csr_t *mat, int rank, std::vector<std::vector<double>>& uFac,
                std::vector<std::vector<double>>& iFac) {
  int item;
  float rat, uAvg;
  //create dense matrix from sparse
  Eigen::MatrixXd A = Eigen::MatrixXd::Zero(mat->nrows, mat->ncols);
  
  for (int u = 0; u < mat->nrows; u++) {
    uAvg = 0;
    for (int ii = mat->rowptr[u]; ii < mat->rowptr[u+1]; ii++) {
      item = mat->rowind[ii];
      rat = mat->rowval[ii];
      A(u, item) = rat;
      uAvg += rat;
    }
    uAvg = uAvg/(mat->rowptr[u+1] - mat->rowptr[u]);
    for (item = 0; item < mat->ncols; item++) {
      if (A(u, item) == 0.0) {
        A(u, item) = uAvg;
      }
    }
  }
  std::cout << "\nmat nrows: " << mat->nrows << " ncols: " << mat->ncols; 
  std::cout << "\nDense mat nrows: " << A.rows() << " ncols: " << A.cols();

  //compute right singular vectors
  Eigen::MatrixXd ATA = A.transpose() * A;
  computeTopEig(ATA, rank, 2.0*rank, iFac);
  
  //compute left singular vectors
  Eigen::MatrixXd AAT = A*A.transpose();
  computeTopEig(AAT, rank, 2.0*rank, uFac);

}


