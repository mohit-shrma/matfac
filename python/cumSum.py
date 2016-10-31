import sys

#stdin of floats
ipList = map(lambda x: float(x), sys.stdin.readlines())

ipSum = [0 for x in ipList]
ipSum[0] = ipList[0]
for i in range(1, len(ipList)):
  ipSum[i] = ipSum[i-1] + ipList[i]

ipStrSums = map(str, ['{0:.3g}'.format(100*ipSum[0]), '{0:.3g}'.format(100*ipSum[1]),
'{0:.3g}'.format(100*ipSum[2]), '{0:.3g}'.format(100*ipSum[4]),
'{0:.3g}'.format(100*ipSum[9])] ) 
print ' & '.join(ipStrSums)



