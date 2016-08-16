import sys
import numpy as np


def genFacs(nUsers, nItems, dim, nFacs = 10):
  for i in range(nFacs):
    uFac     = np.random.rand(nUsers, dim)
    iFac     = np.random.rand(nItems, dim)
    uFacName = 'uFac_' + str(nUsers) + '_' + str(dim) + '_' + str(i) + '.txt'
    iFacName = 'iFac_' + str(nItems) + '_' + str(dim) + '_' + str(i) + '.txt'
    np.savetxt(uFacName, uFac)
    np.savetxt(iFacName, iFac)


def main():
  nUsers   = int(sys.argv[1])
  nItems   = int(sys.argv[2])
  dim      = int(sys.argv[3])
  randSeed = 1
  np.random.seed(randSeed)
  genFacs(nUsers, nItems, dim)


if __name__ == '__main__':
  main()

