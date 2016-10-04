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


def genScaledFacs(nUsers, nItems, dim, scale = 80000, nFacs = 5):
  for i in range(nFacs):
    A = np.random.rand(nUsers, dim)
    B = np.random.rand(nItems, dim)
    [ua, sa, va] = np.linalg.svd(A, full_matrices=0)
    [ub, sb, vb] = np.linalg.svd(B, full_matrices=0)
    S = np.identity(dim)*np.sqrt(scale)
    uFac = np.dot(ua, S)
    iFac = np.dot(ub, S)
    print 'uFac Norm: ', np.linalg.norm(uFac)
    print 'iFac Norm: ', np.linalg.norm(iFac)
    X = np.dot(uFac[np.random.randint(nUsers, size=500), :],
        iFac[np.random.randint(nItems, size=500), :].T)
    print 'avg: ', np.average(X)
    print 'min: ', np.min(X)
    print 'max: ', np.max(X)
    uFacName = 'uFac_' + str(nUsers) + '_' + str(dim) + '_' + str(i) + '.txt'
    iFacName = 'iFac_' + str(nItems) + '_' + str(dim) + '_' + str(i) + '.txt'
    np.savetxt(uFacName, uFac)
    np.savetxt(iFacName, iFac)


def main():
  nUsers   = int(sys.argv[1])
  nItems   = int(sys.argv[2])
  dim      = int(sys.argv[3])
  scale    = int(sys.argv[4])
  randSeed = 1
  np.random.seed(randSeed)
  genScaledFacs(nUsers, nItems, dim, scale)


if __name__ == '__main__':
  main()

