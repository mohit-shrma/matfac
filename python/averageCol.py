import sys

lines = sys.stdin.readlines()

avg = []
count = 0

for line in lines:
  cols = line.strip().split()
  if len(cols) == 0:
    continue
  colsF = map(float, cols)
  if len(avg) == 0:
    avg = [0.0 for x in colsF]
  for i in range(len(colsF)):
    avg[i] += colsF[i]
  count += 1
s = ' '
for val in avg:
  s += str(val/count) + ' '

print s + '\n'






  
