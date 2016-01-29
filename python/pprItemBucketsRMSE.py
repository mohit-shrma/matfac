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

  with open(pprFName, 'r') as f:
    for line in f:
      cols = line.strip().split()
      bucketId = 0
      for i in range(0, len(cols), itemsPerBuck*2):
        
        nnz = 0
        se = 0.0

        for j in range(i, i+(itemsPerBuck*2), 2):
          
          if j >= len(cols)-1:
            #print 'bucket boundry'
            break
          
          item    = int(cols[j]) - nUsers
          origRat = np.dot(origUFac[u], origIFac[item])
          estRat  = np.dot(uFac[u], iFac[item])

          se += (origRat - estRat)*(origRat - estRat)
          nnz += 1
          
        rmse = np.sqrt(se/nnz)
        #print 'i:', i, 'bucketId: ', bucketId
        bucketsScore[bucketId] += se
        bucketsNNZ[bucketId] += nnz
        bucketId += 1

      if 0 == u%1000:
        print u, ' done...'
        print 'buckets score: ', bucketsScore
        print 'bucketsNNZ: ', bucketsNNZ

      u += 1

  for i in range(nBuckets):
    bucketsScore[i] = np.sqrt(bucketsScore[i]/bucketsNNZ[i])
  print 'Final buckets score: ', bucketsScore
  return bucketsScore


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

  computeBucketsRMSE(pprFName, uFac, iFac, origUFac, origIFac, nBuckets)


if __name__ == '__main__':
  main()


