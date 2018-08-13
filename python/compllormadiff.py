import sys

csvPred = sys.argv[1]
testF = sys.argv[2]

uiPred = {}
with open(csvPred, 'r') as f:
    for line in f:
        cols = line.strip().split(',')
        u = cols[0]
        item = cols[1]
        rating = float(cols[2])
        if u not in uiPred:
            uiPred[u] = {}
        uiPred[u][item] = rating

with open(testF, 'r') as f, open('llormaDiff.txt', 'w') as g:
    for line in f:
        cols = line.strip().split(' ')
        u = cols[0]
        item = cols[1]
        testRating = float(cols[2])
        predRating = uiPred[u][item]
        diff = abs(testRating - predRating)
        g.write(u + ' ' + item + ' ' + str(diff) + '\n')






