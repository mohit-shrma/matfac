import sys
import os
import os.path

LEARN_RATE = 0.005

def checkIsFacDim(facStr, dims):
  cols = facStr.split('_')
  facDim = int(cols[2])
  for dim in dims:
    if dim == facDim:
      return True
  return False


def getRandLatFacs(ipDir):
  allFiles = [f for f in os.listdir(ipDir) if os.path.isfile(os.path.join(ipDir, f))]
  uFacs    = [f for f in allFiles if f.startswith('uFac') and checkIsFacDim(f, [5, 10, 20])]
  iFacs    = [f for f in allFiles if f.startswith('iFac') and checkIsFacDim(f, [5, 10, 20])]
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
    for i in range(len(mats)):
      mat = mats[i]
      for graphMat in graphMats:
        trainMat = os.path.join(ipDir, mat[0])
        testMat = os.path.join(ipDir, mat[1])
        valMat = os.path.join(ipDir, mat[2])
        uFac = os.path.join(ipDir, latfac[0])
        iFac = os.path.join(ipDir, latfac[1])
        pref = suff + '_' + str(ind) + '_' + str(i) + '_' + str(dim) 
        if graphMat != 'null1':
          opLog = os.path.join(ipDir, pref + '_samplog') 
        else:
          opLog = os.path.join(ipDir, pref + '_log') 
        print 'cd ' + ipDir + ' && ', prog, nUsers, nItems, dim, 50000, dim, dim, 1, 0.01, 0.01, \
        LEARN_RATE, 0.0, 0.0, trainMat, testMat, valMat, graphMat, uFac, iFac, \
            'null1', 'null2',  pref, ' > ', opLog  


def genJobsUsingGraph(prog, mats, latfacs, ipDir, suff='mf'):
  for latfac in latfacs:
    cols = latfac[0].split('_')
    if len(cols) <= 3:
      continue
    ind = cols[3].strip('.txt')
    dim = cols[2]
    nUsers = cols[1]
    cols = latfac[1].split('_')
    nItems = cols[1]
    for i in range(len(mats)):
      mat = mats[i]
      trainMat = os.path.join(ipDir, mat[0])
      testMat = os.path.join(ipDir, mat[1])
      valMat = os.path.join(ipDir, mat[2])
      cols = trainMat.split('.')
      randInd = cols[2]
      graphMat = 'mf_' + str(randInd) + '.train.jacSim.metis'
      uFac = os.path.join(ipDir, latfac[0])
      iFac = os.path.join(ipDir, latfac[1])
      pref = suff + '_' + str(ind) + '_' + str(i) + '_' + str(dim) 
      opLog = os.path.join(ipDir, pref + '_samplog') 
      print 'cd ' + ipDir + ' && ', prog, nUsers, nItems, dim, 50000, dim, dim, 1, 0.01, 0.01, \
      LEARN_RATE, 0.0, 0.0, trainMat, testMat, valMat, graphMat, uFac, iFac, \
          'null1', 'null2',  pref, ' > ', opLog  


def genGraphJobs(prog, mats, latfacs, ipDir, suff='mf'):
  for latfac in latfacs[:1]:
    cols = latfac[0].split('_')
    if len(cols) <= 3:
      continue
    ind = cols[3].strip('.txt')
    dim = cols[2]
    nUsers = cols[1]
    cols = latfac[1].split('_')
    nItems = cols[1]
    for i in range(len(mats)):
      mat = mats[i]
      trainMat = os.path.join(ipDir, mat[0])
      testMat = os.path.join(ipDir, mat[1])
      valMat = os.path.join(ipDir, mat[2])
      cols = trainMat.split('.')
      randInd = cols[2]
      uFac = os.path.join(ipDir, latfac[0])
      iFac = os.path.join(ipDir, latfac[1])
      pref = suff + '_' +  randInd  
      opLog = os.path.join(ipDir, pref + '_graph_log') 
      print 'cd ' + ipDir + ' && ', prog, nUsers, nItems, dim, 50000, dim, dim, 1, 0.01, 0.01, \
      LEARN_RATE, 0.0, 0.0, trainMat, testMat, valMat, 'null1', uFac, iFac, \
          'null1', 'null2',  pref, ' > ', opLog  



def main():
  ipDir  = sys.argv[1]
  prog   = sys.argv[2]

  randFacs = getRandLatFacs(ipDir)
  #print 'Latent factors: ', len(randFacs)
  #print randFacs

  randMats = getMats(ipDir, 'syn.rand.ind.csr')
  #print 'Random matrices: ', len(randMats)
  #print randMats
  
  graphMats = []#getGraphMats(ipDir, 'metis')
  if len(graphMats) == 0:
    graphMats = ['null1']
  #print 'graphMats: ', graphMats

  realMats = getMats(ipDir, 'syn.ind.csr')
  #print 'Real matrices: ', len(realMats)
  #print realMats
  #print 'Generating jobs...' 
  #genJobs(prog, realMats, randFacs, ipDir, graphMats)
  #genJobs(prog, randMats, randFacs, ipDir, graphMats=['null1'], suff='mfrand')
  genJobsUsingGraph(prog, randMats, randFacs, ipDir, suff='mfrand') 
  #genGraphJobs(prog, realMats, randFacs, ipDir, suff='mf') 

if __name__ == '__main__':
  main()


