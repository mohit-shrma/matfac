import sys
import numpy as np
import types

def updateMetricDic(ipFName, metricDic):
  with open(ipFName, 'r') as f:
    for line in f:
      line = line.strip()
      if ':' in line:
        if line.count(':') == 1:
          ind = line.index(':')
          key = line[:ind]
          cols = line[ind+1:].strip(',').split(',')
          nVals = len(cols)
          filtNans = filter(lambda x: 'nan' not in x, cols)
          fVals = map(float, filtNans)
          nz = nVals - len(fVals)
          for i in range(nz):
            fVals.append(0)
          if key not in metricDic:
            metricDic[key] = [np.zeros(nVals), 0]
          metricDic[key][0] += np.asarray(fVals)
          metricDic[key][1] += 1
        else:
          kvs = line.split()
          k1 = kvs[0].strip(':')
          v1 = float(kvs[1])
          k2 = kvs[2].strip(':')
          v2 = float(kvs[3])
          if k1 not in metricDic:
            metricDic[k1] = [0, 0]
          metricDic[k1][0] += v1
          metricDic[k1][1] += 1
          if k2 not in metricDic:
            metricDic[k2] = [0, 0]
          metricDic[k2][0] += v2
          metricDic[k2][1] += 1 
    

def getIpFileList(ipF):
  ipFileList = []
  with open(ipF, 'r') as f:
    for line in f:
      ipFileList.append(line.strip())
  return ipFileList


def getAvgMetric(ipFileList):
  metricDic = {}
  for ipFName in ipFileList:
    updateMetricDic(ipFName, metricDic)
  print 'No. of files: ', len(ipFileList)
  for k, v in metricDic.iteritems():
    if isinstance(v[0], types.FloatType):
      print k+':', v[0]/v[1]
    else:
      print k+':', ','.join(map(str, list(v[0]/v[1])))


def main():
  if len(sys.argv) == 2:
    ipF = sys.argv[1]
    ipFileList = getIpFileList(ipF)
  elif len(sys.argv) > 2:
    ipFileList = sys.argv[1:]
  else:
    ipFileList = map(lambda x: x.strip('\n'), sys.stdin.readlines())
    
  

  getAvgMetric(ipFileList)


if __name__ == '__main__':
  main()


