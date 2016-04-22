import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import sys
import os.path

def getUserStats(userStatFName):
  uStats = {}
  with open(userStatFName, 'r') as f:
    for line in f:
      cols = line.strip().split()
      user = int(cols[0])
      uRatings = int(cols[1])
      nUsers = int(cols[2])
      meanFreq = float(cols[3])
      top500Count = int(cols[4])
      uStats[user] = (uRatings, nUsers, meanFreq, top500Count)
  return uStats


def getURMSEProbs(probFName, rmseFName):
  
  uProbs = {} 
  uRMSE = {}

  with open(probFName, 'r') as f:
    for line in f:
      cols = line.strip().split()
      user = int(cols[0])
      probs = map(float, cols[1:])
      probs = map(np.log10, probs)
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
  minProb = 0
  maxProb = -100
  for user, probs in uProbs.iteritems():
    if min(probs) < minProb:
      minProb = min(probs)
    if max(probs) > maxProb:
      maxProb = max(probs)

  pp = PdfPages(pdfFName)
  
  users = uRMSE.keys()
  users.sort()

  nRatingsUser = []
  for user in users:
    nRatingsUser.append((uStats[user][0], user))
  nRatingsUser.sort()

  for (nRat, user) in nRatingsUser:
    rmses  = uRMSE[user]
    
    plt.figure()
    plt.clf()
    
    plt.subplot(1, 2, 1)
    plt.plot(range(len(rmses)), rmses)
    axes = plt.gca()
    axes.set_ylim([0, maxRMSE])
    plt.xlabel('buckets')
    plt.xticks(range(len(rmses)))
    plt.ylabel('RMSE')
    plt.grid()
    #plt.title('user: ' + str(user))
    plt.suptitle('user: ' + str(user) + ' ratings: ' 
        + str(uStats[user][0]) + ' 2Hop users: ' 
        + str(uStats[user][1]) + ' meanIFreq: '
        + str(uStats[user][2]) + ' topItems: '
        + str(uStats[user][3]))
  
   
    probs =  uProbs[user]
    #print probs[0] 
    plt.subplot(1, 2, 2)
    plt.plot(range(len(probs)), probs)
    axes = plt.gca()
    axes.set_ylim([minProb, maxProb])
    plt.xlabel('buckets')
    plt.xticks(range(len(probs)))
    plt.ylabel('average steady state probabilities (log10)')
   
    plt.tight_layout()
    plt.subplots_adjust(top = 0.90)

    pp.attach_note('user: ' + str(user) + ' ratings: ' 
        + str(uStats[user][0]) + ' twoHop users: ' 
        + str(uStats[user][1]) + ' mean itemFreq: ' 
        + str(uStats[user][2]) + ' top500Items: '
        + str(uStats[user][3]))
    pp.savefig() 
    
    plt.close()

  pp.close()


def getLambdasRMSEProb():
  lambdas = ['0.01', '0.25', '0.50', '0.75', '0.99']
  uLambdaDic = {}
  for lamda in lambdas:
    uLambdaDic[lamda] = {}
    probFName = "mf_50_100_" + str(lamda)  
    
    if len(lamda) == 5:
      probFName = probFName + "000"
    else:
      probFName = probFName + "0000"
    
    rmseFName = probFName + "_uRMSE.txt" 
    uStatFName = probFName + "_userStats.txt"
    probFName = probFName + "_uProbs.txt" 

    if (not os.path.isfile(probFName)):
      print 'Not found: ', probFName
    if (not os.path.isfile(rmseFName)):
      print 'Not found: ", rmseFName'
    (uProbs, uRMSE) = getURMSEProbs(probFName, rmseFName)
    for user, probs in uProbs.iteritems():
      if user not in uLambdaDic:
        uLambdaDic[lamda][user] = [probs]
    for user, rmses in uRMSE.iteritems():
      uLambdaDic[lamda][user].append(rmses)
    uStats = getUserStats(uStatFName)
  return (uLambdaDic, uStats)


def savePlotsLambdas(uLambdaDic, uStats, pdfFName):
  #get max RMSE, probs
  maxRMSE = 0
  minProb = 0
  maxProb = -100
  for lamda, lamdaDic in uLambdaDic.iteritems():
    for user, uData in lamdaDic.iteritems():
      rmses = uData[1]
      if max(rmses) > maxRMSE:
        maxRMSE = max(rmses)
      probs = uData[0]
      if min(probs) < minProb:
        minProb = min(probs)
      if max(probs) > maxProb:
        maxProb = max(probs)
 
  print 'maxRMSE: ', maxRMSE
  print 'minProb: ', minProb
  print 'maxProb: ', maxProb

  nRatingsUser = []
  for user, stats in uStats.iteritems():
    nRatingsUser.append((stats[0], user))
  nRatingsUser.sort()

  pp = PdfPages(pdfFName)
  
  lambdas = uLambdaDic.keys()
  lambdas.sort()

  for (nRat, user) in nRatingsUser:
    
    print 'user: ', user

    plt.figure()
    plt.clf()
    
    plt.subplot(1, 2, 1)
    for lamda in lambdas:
      uData = uLambdaDic[lamda][user]
      rmses = uData[1]
      plt.plot(range(len(rmses)), rmses)
    plt.legend(labels=lambdas)
    axes = plt.gca()
    #axes.set_ylim([0, maxRMSE])
    plt.xlabel('buckets')
    plt.xticks(range(len(rmses)))
    plt.ylabel('RMSE')
    plt.suptitle('user: ' + str(user) + ' ratings: ' 
        + str(uStats[user][0]) + ' 2Hop users: ' 
        + str(uStats[user][1]) + ' meanIFreq: '
        + str(uStats[user][2]) + ' topItems: '
        + str(uStats[user][3]))
 
    
    plt.subplot(1, 2, 2)
    for lamda in lambdas:
      uData = uLambdaDic[lamda][user]
      probs = uData[0]
      plt.plot(range(len(probs)), probs)
    plt.legend(labels=lambdas)
    axes = plt.gca()
    #axes.set_ylim([minProb, maxProb])
    plt.xlabel('buckets')
    plt.xticks(range(len(probs)))
    plt.ylabel('average steady state probabilities (log10)')

    plt.tight_layout()
    plt.subplots_adjust(top = 0.90)

    pp.savefig()
    plt.close()
    print 'Done...'
  pp.close()


def main():
  #rmseFName = sys.argv[1]
  #probFName = sys.argv[2]
  #statsFName = sys.argv[3]
  pdfFName = sys.argv[1]

  #(uProbs, uRMSE) = getURMSEProbs(probFName, rmseFName)
  #uStats = getUserStats(statsFName)
  (uLambdaDic, uStats) = getLambdasRMSEProb()
  savePlotsLambdas(uLambdaDic, uStats, pdfFName)

  #savePlotsToPDF(uRMSE, uProbs, uStats, pdfFName)


if __name__ == "__main__":
  main()

