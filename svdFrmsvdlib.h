#ifndef _SVD_FRM_SVDLIB_H_
#define _SVD_FRM_SVDLIB_H_

#include <iostream>
#include <vector>
#include <memory>
extern "C" {
  #include "svdlib.h"
}

#include "GKlib.h"

void svdFrmSvdlibCSR(gk_csr_t *mat, int rank, std::vector<std::vector<double>>& uFac,
                std::vector<std::vector<double>>& iFac, bool pureSVD);

#endif

