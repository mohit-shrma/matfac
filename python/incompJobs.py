import sys
import os

def checkIfIncompOld(jobStr):
  ind = jobStr.index('>')
  prefix = jobStr[ind-100:ind].split()[-1]
  cols = prefix.split('_')
  topFName = prefix + '_top_.txt'
  #topFName = prefix + '_top_samp.txt'
  logFile = jobStr[ind+1:].strip() 
  if not os.path.isfile(logFile):
    print 'Log file not found: '+ logFile
    return (False, logFile)
  ipDir = os.path.dirname(logFile)
  topFile = os.path.join(ipDir, topFName)
  if not os.path.isfile(topFile):
    print 'Top file not found: '+ topFile
    return (False, logFile)
  return (True, logFile)


def checkComp(jobsListFName):
  with open(jobsListFName, 'r') as f:
    for line in f:
      status = checkIfIncompOld(line)
      if status[0]:
        print 'COMP: '+ status[1]
      else:
        print 'NOT_COMP: '+ status[1]

def main():
  jobsFName = sys.argv[1]
  checkComp(jobsFName)


if __name__ == '__main__':
  main()

