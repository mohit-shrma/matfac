#ifndef _ANALYZE_MODELS_H_
#define _ANALYZE_MODELS_H_

#include "modelMF.h"
#include "datastruct.h"
#include <iomanip>

void compareModels(Data& data, Params& params);
void averageModels(Data& data, Params& params);
void compJaccSimAccu(Data& data, Params& params);
void compJaccSimAccuMeth(Data& data, Params& params);
void compJaccSimAccuSingleOrigModel(Data& data, Params& params);
void analyzeAccuracy(Data& data, Params& params);
void analyzeAccuracySingleOrigModel(Data& data, Params& params);
void meanAndVarSameGroundAllUsers(Data& data, Params& params) ;
void meanAndVarSameGroundSampUsers(Data& data, Params& params) ;

#endif

