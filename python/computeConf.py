import sys
import numpy as np

def computeConf(u, item, uFacs, iFacs):
  nModels = len(uFacs)
  predRat = []
  for m in range(nModels):
    uFac = uFacs[m]
    iFac = iFacs[m]
    predRat.append(np.dot(uFac[u], iFac[item]))
  std = np.std(predRat)
  confScore = -1
  if 0 != std:
    confScore = 1.0/std
  return confScore


def infCheck(x):
  for i in range(len(x)):
    if np.inf == x[i]:
      x[i] = -1
  return x


def computeConfBuckRMSE(confScoreFName, uFac, iFac, origUFac, origIFac, 
    nUsers, nItems, nBuckets):
  nItemsPerBuck = nItems/nBuckets
  bucketScores = [0 for i in range(nBuckets)]
  bucketNNZ = [0 for i in range(nBuckets)]
  with open(confScoreFName, 'r') as f:
    u = 0
    for line in f:
      cols = line.strip().split()
      itemConfScores = map(float, cols)
      #check for inf and conv it to -1 if found
      itemConfScores = infCheck(itemConfScores)
      itemScores = zip(itemConfScores, range(nItems))
      itemScores.sort(reverse=True)
      for bInd in range(nBuckets):
        start = bInd*nItemsPerBuck
        end = (bInd+1)*nItemsPerBuck
        if bInd == nBuckets-1 or end > nItems:
          end = nItems
        #print bInd, start, end
        for i in range(start, end):
          item =  itemScores[i][1]
          rui_est = np.dot(uFac[u], iFac[item])
          rui = np.dot(origUFac[u], origIFac[item])
          se = (rui - rui_est)*(rui - rui_est)
          bucketScores[bInd] += se
          bucketNNZ[bInd] += 1
      if u%1000 == 0:
        print 'Done...', u, bucketScores
        #print 'Top item confscore:', itemScores[:10]
      u += 1

  print 'bucketScores: ', bucketScores
  print 'bucketNNZ: ', bucketNNZ
  
  for i in range(nBuckets):
    bucketScores[i] = np.sqrt(bucketScores[i]/bucketNNZ[i])
  
  print 'bucketScores: ', bucketScores


def computeConfBuckRMSEFrmModels(uFacs, iFacs, origUFac, origIFac, 
    uFac, iFac, nUsers, nItems, nBuckets):
  
  nItemsPerBuck = nItems/nBuckets
  bucketScores = [0 for i in range(nBuckets)]
  bucketNNZ = [0 for i in range(nBuckets)]
  
  for u in range(nUsers):
    
    itemScores = []
    for item in range(nItems):
      #TODO: check if std = 0
      confScore = computeConf(u, item, uFacs, iFacs)
      itemScores.append((confScore, item))
    itemScores.sort(reverse=True)
    
    for bInd in range(nBuckets):
      start = bInd*nItemsPerBuck
      end = (bInd+1)*nItemsPerBuck
      if bInd == nBuckets-1 or end > nItems:
        end = nItems
      for i in range(start, end):
        item =  itemScores[i][1]
        rui_est = np.dot(uFac[u], iFac[item])
        rui = np.dot(origUFac[u], origIFac[item])
        se = (rui - rui_est)*(rui - rui_est)
        bucketScores[bInd] += se
        bucketNNZ[bInd] += 1

    if u%1000 == 0:
      print 'Done...', u, bucketScores
      print 'Top item scores: ', itemScores[:10]
  for i in range(nBuckets):
    bucketScores[i] = np.sqrt(bucketScores[i]/bucketNNZ[i])
  
  print 'bucketScores: ', bucketScores

#compute confidence score for (user, item)
def computeConfs(uFacs, iFacs, nUsers, nItems, opPrefix):
  nFac = len(uFacs)
  with open(opPrefix + '_conf.txt', 'w') as g:
    for u in range(nUsers):
      for item in range(nItems):
        #compute std dev across models for (u, item)
        predRats = []
        for i in range(nFac):
          uFac = uFacs[i]
          iFac = iFacs[i]
          rating = np.dot(uFac[u], iFac[item])
          predRats.append(rating)
        #std dev of the predicted rating
        stdRat = np.std(predRats)
        conf = -1
        if 0 != stdRat:
          conf = 1.0/stdRat
        g.write(str(conf) + ' ')
      g.write('\n')
      if u % 100 == 0:
        print "Done... ", u


def computeConfsWPR(uFacs, iFacs, nUsers, nItems, prFName, nBuckets, 
    opPrefix):
  nFac = len(uFacs)
  nItemsPerBuck = nItems/nBuckets
  print 'nItemsPerBuck: ', nItemsPerBuck
  with open(opPrefix + '.buckOverlap', 'w') as g, open(prFName, 'r') as f:
    u = 0
    for line in f:
      cols = line.strip().split()
      uPrItems = []
      uConfItems = []
      uConfSortItems = []
      for i in range(0, len(cols), 2):
        item = int(cols[i]) - nUsers
        uPrItems.append(item)
        #compute std dev across models for (u, item)
        predRats = []
        for k in range(nFac):
          uFac = uFacs[k]
          iFac = iFacs[k]
          rating = np.dot(uFac[u], iFac[item])
          predRats.append(rating)
        #std dev of the predicted rating
        stdRat = np.std(predRats)
        if 0 == stdRat:
          continue
          #print u, item, predRats
        conf = 1.0/stdRat
        uConfItems.append((conf,item))
      #sort in decreasing order
      uConfItems.sort(reverse=True)
      for (confScore, item) in uConfItems:
        uConfSortItems.append(item)

      #compute overlap among buckets
      for i in range(nBuckets):
        startItem = i*nItemsPerBuck
        endItem = (i+1)*nItemsPerBuck
        if i==nBuckets-1 or endItem > nItems:
          endItem = nItems
        prItemSet = set(uPrItems[startItem: endItem])
        confItemSet = set(uConfSortItems[startItem: endItem])
        overlapCount = len(prItemSet & confItemSet)
        g.write(str(overlapCount) + ' ')
      g.write('\n')
      g.flush()

      if (u%100) == 0:
        print 'Done...', u

      u += 1


def getFacs(facNames):
  facs = []
  for facName in facNames:
    fac = np.loadtxt(facName)
    facs.append(fac)
  return facs


def main():
  nUsers         = int(sys.argv[1])
  nItems         = int(sys.argv[2])
  uFacName       = sys.argv[3]
  iFacName       = sys.argv[4]
  origUFacName   = sys.argv[5]
  origIFacName   = sys.argv[6]
  confScoreFName = sys.argv[7]

  #prFName = sys.argv[3]
  #opPrefix = sys.argv[4]

  uFacNames = ['conf_0_uFac_20000_5_0.001000.mat', 
      'conf_1_uFac_20000_5_0.001000.mat',
      'conf6_uFac_20000_5_0.001000.mat',
      'conf7_uFac_20000_5_0.001000.mat',
      'conf8_uFac_20000_5_0.001000.mat']
  iFacNames = ['conf_0_iFac_17764_5_0.001000.mat',
      'conf_1_iFac_17764_5_0.001000.mat',
      'conf6_iFac_17764_5_0.001000.mat',
      'conf7_iFac_17764_5_0.001000.mat',
      'conf8_iFac_17764_5_0.001000.mat']
  """
  uFacNames = ['conf_0_uFac_20000_5_0.001000.mat',
      'conf_1_uFac_20000_5_0.001000.mat',
      'conf_2_uFac_20000_5_0.001000.mat',
      'conf6_uFac_20000_5_0.001000.mat',
      'conf7_uFac_20000_5_0.001000.mat',
      'conf8_uFac_20000_5_0.001000.mat']
  iFacNames = ['conf_0_iFac_20000_5_0.001000.mat',
      'conf_1_iFac_20000_5_0.001000.mat',
      'conf_2_iFac_20000_5_0.001000.mat',
      'conf6_iFac_20000_5_0.001000.mat',
      'conf7_iFac_20000_5_0.001000.mat',
      'conf8_iFac_20000_5_0.001000.mat']
  """
  uFacs = getFacs(uFacNames)
  iFacs = getFacs(iFacNames)

  uFac = np.loadtxt(uFacName)
  iFac = np.loadtxt(iFacName)
  origUFac = np.loadtxt(origUFacName)
  origIFac = np.loadtxt(origIFacName)

  #computeConfs(uFacs, iFacs, nUsers, nItems, opPrefix)
  #computeConfsWPR(uFacs, iFacs, nUsers, nItems, prFName, 10, opPrefix)
  computeConfBuckRMSE(confScoreFName, uFac, iFac, origUFac, origIFac, nUsers,
      nItems, 10)
  #computeConfBuckRMSEFrmModels(uFacs, iFacs, origUFac, origIFac, uFac, iFac,
  #    nUsers, nItems, 10)


if __name__ == '__main__':
  main()



