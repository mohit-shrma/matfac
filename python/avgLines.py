import sys 

sm = []
lineCount = 0.0

for line in sys.stdin:
  cols = line.strip().split()
  for i in range(len(cols)):
    #print cols[i]
    if 0 == lineCount:
      sm.append(float(cols[i]))
    else:
      sm[i] += float(cols[i])
  lineCount += 1.0

s = ''
for elem in sm:
  s = s + str(elem/lineCount) + ' '
print s
  


