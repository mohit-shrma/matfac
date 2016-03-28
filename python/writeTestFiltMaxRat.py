import sys



def writeMat(ipFileName, opFileName, filtRat = 5):
  with open(ipFileName, 'r') as f, open(opFileName, 'w') as g:
    for line in f:
      cols = line.strip().split()
      for i in range(0, len(cols), 2):
        rating = float(cols[i+1])
        if rating >= filtRat:
          g.write(cols[i] + ' ' + cols[i+1] + ' ')
      g.write('\n')



def main():
  ipFileName = sys.argv[1]
  opFileName = sys.argv[2]
  filtRat = float(sys.argv[3])
  writeMat(ipFileName, opFileName, filtRat)
  

if __name__ == '__main__':
  main()



