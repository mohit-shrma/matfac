import sys
import numpy as np


def computeBucketsRMSE(pprFName, uFac, iFac, origUFac, origIFac, 
    nBuckets):
  nItems = iFac.shape[0]
  nUsers = uFac.shape[0]
  u = 0
  itemsPerBuck = nItems/nBuckets
  print 'nItems: ', nItems, 
  print 'itemsPerBuck: ', itemsPerBuck

  bucketsScore = [0 for i in range(nBuckets+1)]
  bucketsNNZ = [0 for i in range(nBuckets+1)]
  usersBucketProp = {}

  with open(pprFName, 'r') as f:
    for line in f:
      cols = line.strip().split()
      bucketId = 0
      for i in range(0, len(cols), itemsPerBuck*2):
        
        nnz = 0
        se = 0.0
        
        first50PcItems = 0.0
        second50PcItems = 0.0

        for j in range(i, i+(itemsPerBuck*2), 2):
          
          if j >= len(cols)-1:
            #print 'bucket boundry'
            break
          
          item    = int(cols[j]) - nUsers
          
          if item < 10000:
            first50PcItems += 1
          else:
            second50PcItems += 1

          origRat = np.dot(origUFac[u], origIFac[item])
          estRat  = np.dot(uFac[u], iFac[item])

          se += (origRat - estRat)*(origRat - estRat)
          nnz += 1
          
        rmse = np.sqrt(se/nnz)
        #print 'i:', i, 'bucketId: ', bucketId
        bucketsScore[bucketId] += se
        bucketsNNZ[bucketId] += nnz

        if u not in usersBucketProp:
          usersBucketProp[u] = []
        usersBucketProp[u].append((bucketId,first50PcItems/nnz,
          second50PcItems/nnz))
        bucketId += 1

      if 0 == u%1000:
        print u, ' done...'
        print 'buckets score: ', bucketsScore
        print 'bucketsNNZ: ', bucketsNNZ
        print 'bucketsProp: ', usersBucketProp[u]

      u += 1

  for i in range(nBuckets+1):
    if bucketsNNZ[i] > 0:
      bucketsScore[i] = np.sqrt(bucketsScore[i]/bucketsNNZ[i])
  print 'Final buckets score: ', bucketsScore

  with open(pprFName+'uBucketsProp.txt', 'w') as g:
    for u in range(nUsers):
      for x in usersBucketProp[u]:
        g.write(str(x[0]) + ':' + str(x[1]) + ':' + str(x[2]) + ' ')
      g.write('\n')

  return bucketsScore


def computeBucketsRMSE4mGlobal(pprFName, uFac, iFac, origUFac, origIFac, 
    nBuckets):
  
  nItems = iFac.shape[0]
  nUsers = uFac.shape[0]
  itemsPerBuck = nItems/nBuckets
  print 'nItems: ', nItems, 
  print 'itemsPerBuck: ', itemsPerBuck

  bucketsScore = []
  prItemsRMSEs = []
  prItemsSEs = []

  with open(pprFName, 'r') as f:
    for line in f:
      cols = line.strip().split()
      item = int(cols[0]) - nUsers
      #accumulate square error for the item
      se = 0
      for u in range(nUsers):
        origRat = np.dot(origUFac[u], origIFac[item])
        estRat = np.dot(uFac[u], iFac[item])
        se += (origRat - estRat)*(origRat - estRat)
      prItemsSEs.append(se)
      prItemsRMSEs.append(np.sqrt(se/nUsers))

  for i in range(nBuckets+1):
    startItem = i*itemsPerBuck
    endItem = (i+1)*itemsPerBuck
    if endItem > nItems:
      endItem = nItems
    bucketSE = 0
    for j in range(startItem, endItem):
      bucketSE += prItemsSEs[j]
    if endItem-startItem > 0:
      bucketRMSE = np.sqrt(bucketSE/(nUsers*(endItem-startItem)))
      bucketsScore.append(bucketRMSE)

  print 'bucketsScore: ', bucketsScore

  with open(pprFName + 'gprItemsRMSE.txt', 'w') as g:
    for iRMSE in prItemsRMSEs:
      g.write(str(iRMSE) + '\n')

  with open(pprFName + 'gBuckRMSE.txt', 'w') as g:
    for buckScore in bucketsScore:
      g.write(str(buckScore) + '\n')
  
  return (prItemsRMSEs, bucketsScore)


def main():
  uFacFName    = sys.argv[1]
  iFacFName    = sys.argv[2]
  origUFacName = sys.argv[3]
  origIFacName = sys.argv[4]
  pprFName     = sys.argv[5]
  nBuckets     = int(sys.argv[6])

  uFac = np.loadtxt(uFacFName)
  origUFac = np.loadtxt(origUFacName)
  print 'uFac shape: ', uFac.shape
  nUsers = uFac.shape[0]
  print 'nUsers: ', nUsers

  iFac = np.loadtxt(iFacFName)
  origIFac = np.loadtxt(origIFacName)
  print 'iFac shape: ', iFac.shape
  nItems = iFac.shape[0]
  print 'nItems: ', nItems
  print 'nBuckets: ', nBuckets

  #computeBucketsRMSE(pprFName, uFac, iFac, origUFac, origIFac, nBuckets)
  computeBucketsRMSE4mGlobal(pprFName, uFac, iFac, origUFac, origIFac, nBuckets)

if __name__ == '__main__':
  main()


