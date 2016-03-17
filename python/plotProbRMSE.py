import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import sys

def getUserStats(userStatFName):
  uStats = {}
  with open(userStatFName, 'r') as f:
    for line in f:
      cols = line.strip().split()
      user = int(cols[0])
      uRatings = int(cols[1])
      nUsers = int(cols[2])
      uStats[user] = (uRatings, nUsers)
  return uStats


def getURMSEProbs(probFName, rmseFName):
  
  uProbs = {} 
  uRMSE = {}

  with open(probFName, 'r') as f:
    for line in f:
      cols = line.strip().split()
      user = int(cols[0])
      probs = map(float, cols[1:])
      uProbs[user] = probs
  
  with open(rmseFName, 'r') as f:
    for line in f:
      cols = line.strip().split()
      user = int(cols[0])
      rmse = map(float, cols[1:])
      uRMSE[user] = rmse

  return (uProbs, uRMSE)


def savePlotsToPDF(uRMSE, uProbs, uStats, pdfFName): 
  
  #get max RMSE
  maxRMSE = 0
  for user, rmses in uRMSE.iteritems():
    if max(rmses) > maxRMSE:
      maxRMSE = max(rmses)

  #get max probs
  maxProb = 0
  for user, probs in uProbs.iteritems():
    if max(probs) > maxProb:
      maxProb = max(probs)

  pp = PdfPages(pdfFName)
  for user, rmses in uRMSE.iteritems():
    
    plt.figure()
    plt.clf()
    
    plt.subplot(1, 2, 1)
    plt.plot(range(len(rmses)), rmses)
    axes = plt.gca()
    axes.set_ylim([0, maxRMSE])
    plt.xlabel('buckets')
    plt.ylabel('RMSE')
    #plt.title('user: ' + str(user))
    plt.suptitle('user: ' + str(user) + ' ratings: ' 
        + str(uStats[user][0]) + ' no. of adjacent users: ' 
        + str(uStats[user][1]))
  
   
    probs = uProbs[user]
    #print probs[0] 
    plt.subplot(1, 2, 2)
    plt.plot(range(len(probs)), probs)
    axes = plt.gca()
    axes.set_ylim([0, maxProb])
    plt.xlabel('buckets')
    plt.ylabel('average steady state probabilities')
   
    plt.tight_layout()
    plt.subplots_adjust(top = 0.90)

    pp.attach_note('user: ' + str(user) + ' ratings: ' 
        + str(uStats[user][0]) + ' no. of adjacent users: ' 
        + str(uStats[user][1]))
    pp.savefig() 
    
    plt.close()

  pp.close()


def main():
  rmseFName = sys.argv[1]
  probFName = sys.argv[2]
  statsFName = sys.argv[3]
  pdfFName = sys.argv[4]

  (uProbs, uRMSE) = getURMSEProbs(probFName, rmseFName)
  uStats = getUserStats(statsFName)

  savePlotsToPDF(uRMSE, uProbs, uStats, pdfFName)


if __name__ == "__main__":
  main()

