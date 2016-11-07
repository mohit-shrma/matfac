import sys

fList = sys.stdin.readlines()

freqCountDic = {}
freqRMSEDic = {}

inFreqCountDic = {}
inFreqRMSEDic = {}

for fName in fList:
  fName = fName.strip()
  with open (fName, 'r') as f:
    for line in f:
      if line.startswith('MaxFreq'):
        cols = line.strip().split()
        freq = int(cols[1])
        freqCount   = float(cols[2])
        freqRMSE    = float(cols[3])
        inFreqCount = float(cols[4])
        inFreqRMSE  = float(cols[5])
       
        if freq not in freqCountDic:
          freqCountDic[freq] = 0.0
        freqCountDic[freq] += freqCount

        if freq not in inFreqCountDic:
          inFreqCountDic[freq] = 0.0
        inFreqCountDic[freq] += inFreqCount

        if freq not in freqRMSEDic:
          freqRMSEDic[freq] = 0.0
        freqRMSEDic[freq] += freqRMSE

        if freq not in inFreqRMSEDic:
          inFreqRMSEDic[freq] = 0.0
        inFreqRMSEDic[freq] += inFreqRMSE

freqs = freqCountDic.keys()
freqs.sort()
count = len(fList)

for freq in freqs:
  print freq, freqCountDic[freq]/count, freqRMSEDic[freq]/count, \
    inFreqCountDic[freq]/count, inFreqRMSEDic[freq]/count, count




