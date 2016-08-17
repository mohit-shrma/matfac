import sys
import os
import os.path

LEARN_RATE = 0.005

def getRandLatFacs(ipDir):
  allFiles = [f for f in os.listdir(ipDir) if os.path.isfile(os.path.join(ipDir, f))]
  uFacs    = [f for f in allFiles if f.startswith('uFac')]
  iFacs    = [f for f in allFiles if f.startswith('iFac')]
  uFacs.sort()
  iFacs.sort()
  latFacs = zip(uFacs, iFacs)
  return latFacs


def getMats(ipDir, matSuff = 'syn.rand.ind.csr'):
  allFiles  = [f for f in os.listdir(ipDir) if os.path.isfile(os.path.join(ipDir, f))]
  allMats   = [f for f in allFiles if f.endswith(matSuff)]
  trainMats = [f for f in allMats if 'train' in f]
  testMats  = [f for f in allMats if 'test' in f]
  valMats   = [f for f in allMats if 'val' in f]
  trainMats.sort()
  testMats.sort()
  valMats.sort()
  randMats = zip(trainMats, testMats, valMats)
  return randMats


def getGraphMats(ipDir, matSuff = 'metis'):
  allFiles  = [f for f in os.listdir(ipDir) if os.path.isfile(os.path.join(ipDir, f))]
  allMats   = [f for f in allFiles if f.endswith(matSuff)]
  return allMats


def genJobs(prog, mats, latfacs, ipDir, graphMats=['null1'], suff='mf'):
  for latfac in latfacs:
    cols = latfac[0].split('_')
    if len(cols) <= 3:
      continue
    ind = cols[3].strip('.txt')
    dim = cols[2]
    nUsers = cols[1]
    cols = latfac[1].split('_')
    nItems = cols[1]
    for mat in mats:
      for graphMat in graphMats:
        trainMat = os.path.join(ipDir, mat[0])
        testMat = os.path.join(ipDir, mat[1])
        valMat = os.path.join(ipDir, mat[2])
        uFac = os.path.join(ipDir, latfac[0])
        iFac = os.path.join(ipDir, latfac[1])
        if graphMat != 'null1':
          opLog = os.path.join(ipDir, suff + '_' + str(ind) + '_' + str(dim) + '_samplog') 
        else:
          opLog = os.path.join(ipDir, suff + '_' + str(ind) + '_' + str(dim) + '_log') 
        print 'cd ' + ipDir + ' && ', prog, nUsers, nItems, dim, 50000, dim, dim, 1, 0.01, 0.01, \
        LEARN_RATE, 0.0, 0.0, trainMat, testMat, valMat, graphMat, uFac, iFac, \
            'null1', 'null2', suff + '_' + str(dim) + '_' + str(ind) , ' > ', opLog  


def main():
  ipDir  = sys.argv[1]
  prog   = sys.argv[2]

  randFacs = getRandLatFacs(ipDir)
  #print 'Latent factors: ', len(randFacs)
  #print randFacs

  randMats = getMats(ipDir, 'syn.rand.ind.csr')
  #print 'Random matrices: ', len(randMats)
  #print randMats
  
  graphMats = getGraphMats(ipDir, 'metis')
  if len(graphMats) == 0:
    graphMats = ['null1']
  #print 'graphMats: ', graphMats

  realMats = getMats(ipDir, 'syn.ind.csr')
  #print 'Real matrices: ', len(realMats)
  #print realMats
  #print 'Generating jobs...' 
  genJobs(prog, realMats, randFacs, ipDir, graphMats)
  #genJobs(prog, randMats, randFacs, ipDir, 'mfrand')
  

if __name__ == '__main__':
  main()


