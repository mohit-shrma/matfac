import sys


def getUIRatCount(ipMat):
    uCount = {}
    itemCount = {}
    u = 0
    with open(ipMat, 'r') as f:
        for line in f:
            cols = line.strip().split()
            items = map(int, cols)
            for item in items:
                if item not in itemCount:
                    itemCount[item] += 1
            uCount[u] = len(items)
            u += 1
    return (uCount, itemCount)


def writeDenseMat(ipMat, uCount, itemCount, opMat, minRat):
    u = 0
    with open(ipMat, 'r') as f, open(opMat, 'w') as g:
        for line in f:
            cols = line.strip().split()
            items = map(int, cols)
            if uCount[u] > minRat:
                for item in items:
                    if itemCount[item] > minRat:
                        g.write(str(item) + ' ')
            g.write('\n')
            u += 1   


def main():
    ipMat = sys.argv[1]
    opMat = sys.argv[2]
    minRat = int(sys.argv[3])
    (uCount, itemCount) = getUIRatCount(ipMat)
    writeDenseMat(ipMat, uCount, itemCount, opMat, minRat)

if __name__ == '__main__':
    main()

