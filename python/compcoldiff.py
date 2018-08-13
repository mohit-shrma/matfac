import sys

ipFName = sys.argv[1]
col1 = int(sys.argv[2])
col2 = int(sys.argv[3])

with open(ipFName, 'r') as f:
    for line in f:
        cols = line.strip().split()
        print float(cols[col1]) - float(cols[col2])


