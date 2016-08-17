import sys
import numpy as np

def updateMetricDic(ipFName, metricDic):
  with open(ipFName, 'r') as f:
    for line in f:
      line = line.strip()
      if ':' in line:
        ind = line.index(':')
        key = line[:ind]
        cols = line[ind+1:].strip(',').split(',')
        nVals = len(cols)
        filtNans = filter(lambda x: 'nan' not in x, cols)
        fVals = map(float, fVals)
        if key not in metricDic:
          metricDic[key] = [np.zeros(nVals), 0]
        metricDic[key][0] += fVals
        metricDic[key][1] += 1


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

  for k, v in metricDic.iteritems():
    print k, v[1]
    print k, v[0]/v[1]


def main():
  ipF = sys.argv[1]
  ipFileList = getIpFileList(ipF)
  getAvgMetric(ipFileList)


if __name__ == '__main__':
  main()


