import scipy.sparse as sps
import sys
import os

def convertToCSRText(ipFName):
    prefix, extension = os.path.splitext(ipFName)
    opFName = prefix + '.csr'
    mat = sps.load_npz(ipFName)
    
    print('Writing output... ', opFName)

    with open(opFName, "w") as g:
        (nrows, ncols) = mat.shape
        print ("nrows: ", nrows, " ncols: ", ncols, " nnz: ", mat.nnz)
        op_nnz = 0
        indptr = mat.indptr
        indices = mat.indices
        data = mat.data
        for u in range(nrows):
            #indptr, indices, data
            for ii in range(indptr[u],  indptr[u+1]):
                item = indices[ii]
                rating = data[ii]
                g.write(str(item) + ' ' +  str(rating) + ' ')
                op_nnz += 1
            g.write('\n')
        print("output nnz: ", op_nnz)


def main():
    if len(sys.argv) < 2:
        print("usage: python convert_scipy_sparse_...  <input.npz>")
    else:
        convertToCSRText(sys.argv[1])



if __name__ == '__main__':
    main()




