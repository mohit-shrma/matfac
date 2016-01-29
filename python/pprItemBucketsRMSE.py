import sys
import numpy as np



def computeBucketsRMSE(pprFName, uFac, iFac, origUFac, origIFac, 
    nBuckets):
  nItems = iFac.shape[1]
  nUsers = uFac.shape[1]
  u = 0
  itemsPerBuck = nItems/nBuckets

  print 'itemsPerBuck: ', itemsPerBuck

  bucketsScore = [0 for i in range(nBuckets)]

  with open(pprFName, 'r') as f:
    for line in f:
      cols = line.strip().split()
      for i in range(0, len(cols), itemsPerBuck*2):
        nnz = 0
        se = 0.0
        for j in range(i, i+(itemsPerBuck*2), 2):
          if j >= len(cols)-1:
            break
          item = int(cols[j]) - nUsers
          origRat = np.dot(origUFac[u], origIFac[item])
          estRat = np.dot(uFac[u], iFac[item])
          se += (origRat - estRat)*(origRat - estRat)
          nnz += 1
        rmse = np.sqrt(se/nnz)
        bucketsScore[i%(itemsPerBuck*2)] += rmse
      u += 1
  for i in range(nBuckets):
    bucketsScore[i] = bucketsScore[i]/u
  print 'buckets score: ', bucketsScore
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
  
  computeBucketsRMSE(pprFName, uFac, iFac, origUFac, origIFac, nBuckets)


if __name__ == '__main__':
  main()


