import sys

fList = sys.stdin.readlines()

freqCountDic = {}
freqRMSEDic = {}

inFreqCountDic = {}
inFreqRMSEDic = {}

varDic = {}

for fName in fList:
  fName = fName.strip()
  with open (fName, 'r') as f:
    for line in f:
      if line.startswith('FreqVar'):
        cols        = line.strip().split()
        var         = cols[1]
        maxFreq     = cols[3]
        freqCount   = float(cols[4])
        freqRMSE    = float(cols[5])
        inFreqCount = float(cols[6])
        inFreqRMSE  = float(cols[7])
        variance    = float(cols[8])
        key = var + '_' + maxFreq
        
        if key not in freqCountDic:
          freqCountDic[key] = 0.0
        freqCountDic[key] += freqCount
        
        if key not in inFreqCountDic:
          inFreqCountDic[key] = 0.0
        inFreqCountDic[key] += inFreqCount

        if key not in freqRMSEDic:
          freqRMSEDic[key] = 0.0
        freqRMSEDic[key] += freqRMSE

        if key not in inFreqRMSEDic:
          inFreqRMSEDic[key] = 0.0
        inFreqRMSEDic[key] += inFreqRMSE
        
        if key not in varDic:
          varDic[key] = 0.0
        varDic[key] += variance

for k in freqCountDic.keys():
  var  = k.split('_')[0]
  freq = k.split('_')[1]
  count = len(fList)
  print var, freq, freqCountDic[k]/count, freqRMSEDic[k]/count, \
  inFreqCountDic[k]/count, inFreqRMSEDic[k]/count, varDic[k]/count, count

