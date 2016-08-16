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


def genJobs(prog, mats, latfacs, suff='mf'):
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
      print prog, nUsers, nItems, dim, 50000, dim, dim, 1, 0.01, 0.01, \
      LEARN_RATE, 0.0, 0.0, mat[0], mat[1], mat[2], 'null1', latfac[0], latfac[1], \
          'null1', 'null2', suff + str(ind) 


def main():
  ipDir  = sys.argv[1]
  prog   = sys.argv[2]

  randFacs = getRandLatFacs(ipDir)
  #print 'Latent factors: '
  #print randFacs

  randMats = getMats(ipDir, 'syn.rand.ind.csr')
  #print 'Random matrices: '
  #print randMats

  realMats = getMats(ipDir, 'syn.ind.csr')
  #print 'Real matrices: '
  #print realMats
  
  #genJobs(prog, realMats, randFacs)
  genJobs(prog, randMats, randFacs, 'mfrand')
  

if __name__ == '__main__':
  main()


