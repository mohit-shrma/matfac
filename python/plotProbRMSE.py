import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import sys

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


def savePlotsToPDF(uRMSE, uProbs, pdfFName): 
  
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
    plt.title('user: ' + str(user))
  
   
    probs = uProbs[user]
    #print probs[0] 
    plt.subplot(1, 2, 2)
    plt.plot(range(len(probs)), probs)
    axes = plt.gca()
    axes.set_ylim([0, maxProb])
    plt.xlabel('buckets')
    plt.ylabel('average steady state probabilities')
   
    plt.tight_layout()
    
    pp.savefig()
    
    plt.close()

  pp.close()



def main():
  rmseFName = sys.argv[1]
  probFName = sys.argv[2]
  pdfFName = sys.argv[3]

  (uProbs, uRMSE) = getURMSEProbs(probFName, rmseFName)
  savePlotsToPDF(uRMSE, uProbs, pdfFName)

if __name__ == "__main__":
  main()

