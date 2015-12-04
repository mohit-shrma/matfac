#ifndef _SPECTRA_SVD_H_
#define _SPECTRA_SVD_H_

#include <Eigen/Core>
#include <Eigen/Dense>
#include <iostream>
#include "GKlib.h"
#include <SymEigsSolver.h>

void spectraSvdFrmCSR(gk_csr_t *mat, int rank, std::vector<std::vector<double>>& uFac,
                std::vector<std::vector<double>>& iFac);

void spectraSvdFrmCSRColAvg(gk_csr_t *mat, int rank, std::vector<std::vector<double>>& uFac,
                std::vector<std::vector<double>>& iFac);
#endif
