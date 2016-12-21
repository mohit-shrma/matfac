import sys

METHS = ['sgd', 'als', 'ccd++']
RANKS = [5, 10, 20]
INDS  = [1]

def genRankMethJobs(prog, trainMat, testMat, valMat, origUFacPre, origIFacPre):
    for rank in RANKS:
        for meth in METHS:
            for ind in INDS:
                for seed in range(1, 11):
                    print prog, '--trainmat', trainMat, '--testmat', testMat, \
                    '--valmat', valMat, '--facdim', rank, '--seed', seed, \
                    '--method', meth, '--prefix', meth + '_' + str(seed), '--origufac', \
                    origUFacPre + '_' + str(rank) + '_' + str(ind) + '.txt', \
                    '--origifac', origIFacPre + '_' + str(rank) + '_' + str(ind) + '.txt' 



def main():
    prog        = sys.argv[1]
    trainMat    = sys.argv[2]
    testMat     = sys.argv[3]
    valMat      = sys.argv[4]
    origUFacPre = sys.argv[5]
    origIFacPre = sys.argv[6]
    genRankMethJobs(prog, trainMat, testMat, valMat, origUFacPre, origIFacPre)
    


if __name__ == '__main__':
    main()



