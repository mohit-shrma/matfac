import sys
import os

def updateArrDic(dic, k, line):
  cols = line.strip().split()[2:]
  vals = []
  for i in range(1, len(cols), 2):
    vals.append(float(cols[i]))
  if k not in dic:
    dic[k] = [[0.0 for i in range(len(vals))],  0.0]
  for i in range(len(vals)):
    dic[k][0][i] += vals[i]
  dic[k][1] += 1


def averageArrDic(dic):
  for k, v in dic.iteritems():
    vals = v[0]
    count = v[1]
    for i in range(len(vals)):
      vals[i] = vals[i]/count
    dic[k] = [vals, count]


def updateDic(dic, k, line):
  cols = line.strip().split()
  val = float(cols[-1])
  if k not in dic:
    dic[k] = [0.0, 0.0]
  dic[k][0] += val
  dic[k][1] += 1


def averageDic(dic):
  for k, v in dic.iteritems():
    dic[k][0] = dic[k][0]/dic[k][1]


def getRMSEs(ipFiles):

  ds = []
  keys = set([])

  valRMSEDic     = {}
  ds.append(valRMSEDic)
  testRMSEDic    = {}
  ds.append(testRMSEDic)
  trainRMSEDic   = {}
  ds.append(trainRMSEDic)

  arrayDs = []
  quartItemD = {}
  arrayDs.append(quartItemD)
  quartUserD = {}
  arrayDs.append(quartUserD)

  topTestItemPairDic = {}
  fileCount = 0
  ipF = open(ipFiles, 'r')
  for line in ipF:
    fName = line.strip()
    fileCount += 1

    ##
    splitInd = fName.split('_')[-2]
    #path = "em/60_20_20_splits/"+splitInd+"/"+fName
    #fName = path
    
    #print fName

    if not os.path.isfile(fName):
      continue
    with open(fName, 'r') as h:
      bName = os.path.basename(fName)
      bk = bName.strip('.log').split('_') [:-2]
      bk = ' '.join(bk)
      keys.add(bk)
      for fLine in h:
        if fLine.startswith('Val'):
          updateDic(valRMSEDic, bk, fLine)
        if fLine.startswith('Test RMSE'):
          updateDic(testRMSEDic, bk, fLine)
        if fLine.startswith('Train RMSE'):
          updateDic(trainRMSEDic, bk, fLine)
        if fLine.startswith('Items P'):
          updateArrDic(quartItemD, bk, fLine)
        if fLine.startswith('Users P'):
          updateArrDic(quartUserD, bk, fLine)
  ipF.close()

  for d in ds:
    averageDic(d)
  
  for d in arrayDs:
    averageArrDic(d)

  notFoundK = set([])

  topTestItemPairDicNF = 0
  
  for d in ds:
    for k in keys:
      if k not in d:
        print 'Not found: ', k 
        notFoundK.add(k)
  for k in keys:
    if k not in topTestItemPairDic:
      topTestItemPairDicNF += 1
  
  if len(notFoundK) > 0:
    print 'not found Count: ', len(notFoundK)
    print 'topTestK: ', topTestItemPairDicNF
    for nf in notFoundK:
      print nf
    return 

  for k in keys:
    if k in notFoundK:
      continue
    tempL = [k]
    tempL += [trainRMSEDic[k][0], valRMSEDic[k][0], testRMSEDic[k][0]]
    tempL += quartItemD[k][0][::-1]
    tempL += quartUserD[k][0][::-1]
    tempL += [testRMSEDic[k][1]]
    tempL += [quartUserD[k][1]]
    print ' '.join(map(str, tempL))
     
  


def main():
  
  ipFiles = sys.argv[1]
  getRMSEs(ipFiles) 
 

if __name__ == '__main__':
  main()



