#ifndef _ARMA_SVD_H_
#define _ARMA_SVD_H_

#include <armadillo>
#include <vector>
#include <iostream>
#include "GKlib.h"



void svdFrmCSR(gk_csr_t *mat, int rank, std::vector<std::vector<double>>& uFac,
                std::vector<std::vector<double>>& iFac) ;
 

#endif
