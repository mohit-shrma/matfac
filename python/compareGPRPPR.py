import sys


def findBetterPPR(pprFName, gprFName):
  
  pprF = open(pprFName, 'r')
  gprF = open(gprFName, 'r')
  u = 0
  pprBetter = 0
  betterUs = []
  for pprLine in pprF:
    gprLine = gprF.readline()
    
    pprs = map(float, pprLine.strip().split())
    gprs = map(float, gprLine.strip().split())
    
    #compare first bucket
    if gprs[0] - pprs[0] > 0.0:
      #print 'user: ', u
      #print 'ppr: ', pprs
      #print 'gpr: ', gprs
      pprBetter += 1
      betterUs.append(u)
    u += 1

  #print 'Nusers: ', u
  #print 'No. ppr better users: ', pprBetter

  pprF.close()
  gprF.close()
  return betterUs


def findAvgItemPerU(csrName, betterUs):
  u = 0
  betterUSet = set(betterUs)
  with open(csrName, 'r') as f:
    for line in f:
      cols = line.strip().split()
      nItems = len(cols)/2
      if u not in betterUSet:
        print u, nItems
      u += 1


def main():
  pprFName = sys.argv[1]
  gprFName = sys.argv[2]
  csrName = sys.argv[3]
  betterUs = findBetterPPR(pprFName, gprFName)
  findAvgItemPerU(csrName, betterUs)

if __name__ == '__main__':
  main()


