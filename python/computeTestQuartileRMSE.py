import sys
import math
from collections import defaultdict

def readPartMap(partName):
    pMap = {}
    with open(partName, 'r') as f:
        for line in f:
            cols = line.strip().split()
            pInd = int(cols[0])
            elem = int(cols[1])
            pMap[elem] = pInd
    return pMap
    

def loadSpMat(ipFName):
    spMat = defaultdict(dict)
    with open(ipFName, 'r') as f:
        u = 0
        for line in f:
            cols = line.strip().split()
            for i in range(0, len(cols), 2):
                spMat[u][int(cols[i])] = float(cols[i+1])
            u += 1
    return spMat      


def quartileRMSEs(predFName, uPartMap, iPartMap, valMat):
    
    partUSqDiff  = defaultdict(float)
    partUSqCount = defaultdict(float)
    partISqDiff  = defaultdict(float)
    partISqCount = defaultdict(float)
    
    missingUICount = 0

    with open(predFName, 'r') as f:
        for line in f:
            cols = line.strip().split(',')
            
            user = int(cols[0])
            item = int(cols[1])
            
            predRating = float(cols[2])
            rating = valMat[user][item]
            
            diff = rating - predRating
           
            if user in uPartMap and item in iPartMap:
                uPart = uPartMap[user]
                iPart = iPartMap[item]

                partUSqDiff[uPart]  += diff*diff
                partISqDiff[iPart]  += diff*diff
                
                partUSqCount[uPart] += 1.0
                partISqCount[iPart] += 1.0
            else:
                missingUICount += 1
    
    print "missing ui: ", missingUICount

    print "user partitions RMSEs count:"
    for partInd, count in partUSqCount.items():
        sqDiff = partUSqDiff[partInd]
        print partInd, math.sqrt(sqDiff/count), count 

    print "item partitions RMSEs count:"
    for partInd, count in partISqCount.items():
        sqDiff = partISqDiff[partInd]
        print partInd, math.sqrt(sqDiff/count), count 


def main():
    
    uPartName = sys.argv[1]
    iPartName = sys.argv[2]
    valSpMat  = sys.argv[3]
    predFName = sys.argv[4]
    
    uPartMap = readPartMap(uPartName)
    iPartMap = readPartMap(iPartName)
    
    valMat = loadSpMat(valSpMat)
    
    quartileRMSEs(predFName, uPartMap, iPartMap, valMat)



if __name__ == '__main__':
    main()


