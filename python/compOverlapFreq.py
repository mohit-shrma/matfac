import sys
import numpy as np
import scipy.stats as stats

""" http://nemates.org/MA/progs/representation.stats.html """
def getRepFacNProb(overlapSz, szSet1, szSet2, szPop):
  #compute expected overlap
  expOverlap = float(szSet1*szSet2)/szPop
  #compute representation factor
  #repFac > 1: more overlap than expected of 2 independent groups
  #repFac < 1: less overlap than expected of 2 independent groups
  repFac = float(overlapSz)/expOverlap
  
  #compute hypergeometric probability, such that by random chance we get the
  #same or more overlap among the sets
  prob = stats.hypergeom.sf(overlapSz+1, szPop, szSet1, szSet2)
  print 'Expected overlap: ', expOverlap, 'Representation factor: ', repFac,\
      'Prob: ', prob

  return (repFac, prob)


def findItems(ipFName, minAccu, maxFreq, maxAccu = -1):
  
  itemSet = set([])
  avgFreq = 0.0
  avgAccu = 0.0
  nItems  = 0.0
  accuPreds = []
  freqs = []
  freqPopSz = 0.0

  with open(ipFName, 'r') as f:
    for line in f:
      cols = line.strip().split()
      item = int(cols[0])
      freq = float(cols[1])
      avgFreq += freq
      nItems += 1
      secFreq = float(cols[2])
      accuPred = float(cols[3])
      accuPreds.append(accuPred)
      avgAccu += accuPred
      freqs.append(freq)
      if freq < maxFreq:
        freqPopSz += 1
      if -1 == maxAccu:
        if freq < maxFreq and accuPred > minAccu:
          itemSet.add(item)
      else:
        if freq < maxFreq and accuPred > minAccu and accuPred < maxAccu:
          itemSet.add(item)
  print 'freqPopSz: ', freqPopSz
  print 'avgFreq: ', np.mean(freqs)
  print 'stdFreq: ', np.std(freqs)
  print 'avgAccu: ', np.mean(accuPreds)
  print 'stdAccu: ', np.std(accuPreds)
  return (itemSet, freqPopSz)


def compOverlapStats(itemSets, szPop):
  nSets = len(itemSets)
  avgRepFac = 0.0
  avgProb = 0.0
  count = 0.0
  print 'population sz: ', szPop
  for i in range(nSets-1):
    set1 = itemSets[i]
    print 'set: ', i, ' sz: ', len(set1)
    for j in range(i+1, nSets):
      set2 = itemSets[j]
      print 'set: ', j, ' sz: ', len(set2)
      if len(set1) > 0 and len(set2) > 0:
        overlapSz = len(set1.intersection(set2))
        print 'overlap: ', overlapSz
        (repFac, prob) = getRepFacNProb(overlapSz, len(set1), len(set2), szPop)
        avgRepFac += repFac
        avgProb += prob
        count += 1
  print 'avgRepFac: ', avgRepFac/count
  print 'avgProb: ', avgProb/count


def compOverlapPc(itemSets):
  nSets = len(itemSets)
  print 'nSets: ', nSets
  pairwiseOverlap = 0.0
  pairwiseOverlapCt = 0
  allInters = set([])
  allInters = allInters.union(itemSets[0])
  avgSetSz = 0.0
  for i in range(nSets):
    allInters = allInters & itemSets[i]
    avgSetSz += len(itemSets[i])
    for j in range(i+1, nSets):
      s1 = itemSets[i]
      s2 = itemSets[j]
      if len(s1) > 0 and len(s2) > 0:
        inters = len(s1.intersection(s2))
        s1Pc = float(inters)/len(s1)
        s2Pc = float(inters)/len(s2)
        pairwiseOverlapCt += 2
        pairwiseOverlap += s1Pc
        pairwiseOverlap += s2Pc
        print i, j, len(s1), s1Pc, len(s2), s2Pc
  avgSetSz = avgSetSz/nSets
  print "average set sz: ", avgSetSz
  print "average pw: ", (pairwiseOverlap/pairwiseOverlapCt)*avgSetSz
  print "average pw %: ", pairwiseOverlap/pairwiseOverlapCt
  print "allInters: ", len(allInters), float(len(allInters))/(avgSetSz)


def getItemSets(ipFiles, minAccu, maxFreq, maxAccu=-1):
  itemSets = []
  szPop = 0
  for ipFName in ipFiles:
    (itemSet, freqPopSz) = findItems(ipFName, minAccu, maxFreq, maxAccu) 
    itemSets.append(itemSet)
    szPop = freqPopSz
  print 'no. of itemsets: ', len(itemSets)
  return (itemSets, szPop)

def main():
  minAccu = float(sys.argv[1])
  maxAccu = float(sys.argv[2])
  maxFreq = float(sys.argv[3])
  ipFiles = sys.argv[4:]

  print 'ipFiles: ', ipFiles
  print 'minAccu: ', minAccu
  print 'maxAccu: ', maxAccu
  print 'maxFreq: ', maxFreq
  (itemSets, szPop) = getItemSets(ipFiles, minAccu, maxFreq, maxAccu)
  compOverlapPc(itemSets)
  compOverlapStats(itemSets, szPop)

if __name__ == '__main__':
  main()


