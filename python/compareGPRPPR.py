import sys


def findBetterPPR(pprFName, gprFName):
  
  pprF = open(pprFName, 'r')
  gprF = open(gprFName, 'r')
  u = 0
  
  for pprLine in pprF:
    gprLine = gprF.readline()
    
    pprs = map(float, pprLine.strip().split())
    gprs = map(float, gprLine.strip().split())
    
    #compare first bucket
    if gprs[0] - pprs[0] < 0.01:
      print u
      print pprs
      print gprs

    u += 1


  pprF.close()
  gprF.close()


def main():
  pprFName = sys.argv[1]
  gprFName = sys.argv[2]
  findBetterPPR(pprFName, gprFName)


if __name__ == '__main__':
  main()


