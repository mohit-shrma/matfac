#ifndef _MODELMF_LOC_H_
#define _MODELMF_LOC_H_

#include <iostream>
#include <vector>
#include <set>
#include <utility>
#include <unordered_set>
#include <string>
#include "io.h"
#include "model.h"

class ModelMFLoc: public Model {
  public:
    std::unordered_set<int> headItems;
    std::unordered_set<int> headUsers;
    ModelMFLoc(const Params& params, 
        std::unordered_set<int>& pHeadItems,
        std::unordered_set<int>& pHeadUsers) : Model(params) {
      headItems = pHeadItems;
      headUsers = pHeadUsers;
    }
    ModelMFLoc(const Params& params, int seed, 
        std::unordered_set<int>& pHeadItems,
        std::unordered_set<int>& pHeadUsers) : Model(params, seed) {
      headItems = pHeadItems;
      headUsers = pHeadUsers;
    }
    ModelMFLoc(const Params& params, const char*uFacName, const char* iFacName, 
        int seed, std::unordered_set<int>& pHeadItems,
        std::unordered_set<int>& pHeadUsers) : Model(params, uFacName,
          iFacName, seed) {
      headItems = pHeadItems;
      headUsers = pHeadUsers;
    }
    
    virtual void train(const Data& data, Model& bestModel,
        std::unordered_set<int>& invalidUsers,
        std::unordered_set<int>& invalidItems);
    void zeroedTailItemFacs(std::unordered_set<int>& headItems);
    void zeroedTailUserFacs(std::unordered_set<int>& headUsers);

};

#endif
