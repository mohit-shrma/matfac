import sys

def writeMat(ipFileName, opFileName, filtRat = 5, headItems):
  with open(ipFileName, 'r') as f, open(opFileName, 'w') as g:
    count = 0
    for line in f:
      cols = line.strip().split()
      for i in range(0, len(cols), 2):
        rating = float(cols[i+1])
        item = cols[i]
        if item in headItems:
          continue
        if rating >= filtRat:
          g.write(cols[i] + ' ' + cols[i+1] + ' ')
          count += 1
      g.write('\n')
    print 'nRatings: ', count


def getHeadItems(headItemFName):
  headItems = []
  with open(headItemFName, 'r') as f:
    for line in f:
      headItems.append(line.strip())
  return headItems


def main():
  ipFileName = sys.argv[1]
  opFileName = sys.argv[2]
  filtRat = float(sys.argv[3])
  
  headItems = []
  if len(sys.argv == 5 ):
    headItemFName = sys.argv[4]
    headItems = getHeadItems(headItemFName)
  headItems = set(headItems)

  writeMat(ipFileName, opFileName, filtRat, headItems)
  

if __name__ == '__main__':
  main()



