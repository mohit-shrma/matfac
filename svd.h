#ifndef _SVD_H_
#define _SVD_H_
#include <Eigen/Dense>
#include <iostream>
#include "GKlib.h"

void svdFrmCSR(gk_csr_t *mat, int rank, std::vector<std::vector<double>>& uFac,
                std::vector<std::vector<double>>& iFac);
#endif
