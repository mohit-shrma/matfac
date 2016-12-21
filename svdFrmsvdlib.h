#ifndef _SVD_FRM_SVDLIB_H_
#define _SVD_FRM_SVDLIB_H_

#include <iostream>
#include <vector>
#include <memory>
#include <Eigen/Dense>
extern "C" {
  #include "svdlib.h"
}

#include "GKlib.h"

void svdFrmSvdlibCSR(gk_csr_t *mat, int rank, std::vector<std::vector<double>>& uFac,
                std::vector<std::vector<double>>& iFac, bool pureSVD);
void svdFrmSvdlibCSRSparsity(gk_csr_t *mat, int rank, std::vector<std::vector<double>>& uFac,
                std::vector<std::vector<double>>& iFac, bool pureSVD);
void svdFrmSvdlibCSREig(gk_csr_t *mat, int rank, Eigen::MatrixXf& uFac,
                Eigen::MatrixXf& iFac, bool pureSVD);
void svdFrmSvdlibCSRSparsityEig(gk_csr_t *mat, int rank, Eigen::MatrixXf& uFac,
                Eigen::MatrixXf& iFac, bool pureSVD);

#endif

