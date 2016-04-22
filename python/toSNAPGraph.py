import sys


def writeWeightedUndirGraph(ipMetisCSRGraph, opTabEdgesName):
  with open(ipMetisCSRGraph, 'r') as f, open(opTabEdgesName, 'w') as g:
    nodeStart = 0
    edges = 0
    for line in f:
      cols = line.strip().split();
      for i in range(0, len(cols), 2):
        item = int(cols[i])
        if item > nodeStart:
          g.write(str(nodeStart) + "\t" + str(item) + "\t" + cols[i+1] + "\n")
          edges += 1
      nodeStart += 1
      if (nodeStart % 5000 == 0):
        print 'Nodes: ', nodeStart
    print "Nodes: ", nodeStart, " Edges: ", edges


def writeTabEdgesGraph(ipMetisCSRGraph, opTabEdgesName):
  with open(ipMetisCSRGraph, 'r') as f, open(opTabEdgesName, 'w') as g:
    nodeStart = 0
    for line in f:
      cols = map(int, line.strip().split())
      for nodeEnd in cols:
        if nodeEnd == nodeStart:
          #remove self edge
          continue
        g.write(str(nodeStart) + "\t" + str(nodeEnd) + '\n')
      nodeStart += 1
      if (nodeStart % 5000 == 0):
        print 'Nodes: ', nodeStart
    print "Nodes: ", nodeStart


#write the other SNAP format graph
def writeUndirectedSNAPGraph(ipMetisCSRGraph, opGraphName):
  with open(ipMetisCSRGraph, 'r') as f, open(opGraphName, 'w') as g:
    nodestart = 0
    nEdges = 0
    for line in f:
      cols = map(int, line.strip().split())
      for nodeEnd in cols:
        g.write(str(nodeStart) + " " + str(nodeEnd) + "\n")
        nEdges += 1
      nodeStart += 1
    print "Nodes: ", nodeStart, "Edges: ", nEdges


def main():

  ipMetisCSRGraph = sys.argv[1]
  opTabEdgesName = sys.argv[2]

  writeWeightedUndirGraph(ipMetisCSRGraph, opTabEdgesName) 


if __name__ == '__main__':
  main()

