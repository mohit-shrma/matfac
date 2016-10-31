import sys

#stdin of floats
ipList = map(lambda x: float(x), sys.stdin.readlines())

ipSum = [0 for x in ipList]
ipSum[0] = ipList[0]
for i in range(1, len(ipList)):
  ipSum[i] = ipSum[i-1] + ipList[i]

ipStrSums = map(str, ipSum)
print ' & '.join(ipStrSums)



