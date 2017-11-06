import sys



def toBPRTriplet(ipCSRFName, opFName):
    u = 0
    posRat = 0
    with open(ipCSRFName, 'r') as f, open(opFName, 'w') as g:
        for line in f:
            cols = line.strip().split()
            for i in range(0, len(cols), 2):
                if cols[i+1] == '1':
                    g.write(str(u) + ' ' + cols[i] + ' ' + str(1) + '\n')
                    posRat += 1
            u += 1
    print ('users: ' + str(u) + ' nPosRatings: ' + str(posRat)) 


def main():

    ipCSRFName = sys.argv[1]
    opFName = sys.argv[2]

    print ('ipCSR: '+ ipCSRFName+ ' opFile: ' + opFName)
    
    toBPRTriplet(ipCSRFName, opFName)

if __name__ == '__main__':
    main()
