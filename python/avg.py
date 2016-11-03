import sys

#stdin of floats
ipList = map(lambda x: float(x), sys.stdin.readlines())
sum = 0.0
for x in ipList:
  sum += x
print sum/len(ipList)


