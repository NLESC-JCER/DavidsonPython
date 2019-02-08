#!/bin/python
import numpy as np


def digaonal_dominant(n,sparsity=1E-4):
    
    A = np.zeros((n,n))
    for i in range(0,n):
        A[i,i] = i + 1 
    A = A + sparsity*np.random.randn(n,n) 
    A = (A.T + A)/2 
    return A


def davidson_solver(A, neigen, tol=1E-6, itermax = 1000):
    """Davidosn solver for eigenvalue problem

    Args :
        A (numpy matrix) : the matrix to diagonalize
        neigen (int)     : the number of eigenvalue requied
        tol (float)      : the rpecision required
        itermax (int)    : the maximum number of iteration
    Returns :
        eigenvalues (array) : lowest eigenvalues
        eigenvectors (numpy.array) : eigenvectors
    """
    n = A.shape[0]
    k = 2*neigen            # number of initial guess vectors 
    V = np.eye(n,k)         # set of k unit vectors as guess
    I = np.eye(n)           # identity matrix same dimen as A
    Adiag = np.diag(A)

    # Begin block Davidson routine
    for i in range(itermax):
    
        # QR of V t oorthonormalize the V matrix
        # this uses GrahmShmidtd in the back
        V,R = np.linalg.qr(V)

        # form the projected matrix 
        T = np.dot(V.T,np.dot(A,V))

        # Diagonalize the projected matrix
        theta,s = np.linalg.eigh(T)

        # Ritz eigenvector
        q = np.dot(V,s)

        # compute the residual append append it to the 
        # set of eigenvectors
        norm = 0
        for j in range(k):

            # residue vetor
            res = np.dot((A - theta[j]*I),q[:,j]) 
            norm += np.linalg.norm(res)/k

            # correction vector
            delta = res / (theta[j]-Adiag)
            delta /= np.linalg.norm(delta)

            # store the correction vectors
            if(j==0):
                Q = delta
            else:
                Q = np.vstack((Q,delta))

        # comute the norm to se if eigenvalue converge
        print("iteration %03d dim %03d norm : %e/%e" %(i,V.shape[1],norm,tol))
        if norm < tol:
            break

        #append the correction vectors to the basis
        V = np.hstack((V,Q.T))
        #V = np.hstack((q,Q.T))
        

    return theta[:neigen], q[:,:neigen]


if __name__ == "__main__":
 
    import time
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-s","--size",type=int,help='Size of the matrix',default=100)
    parser.add_argument("-n","--neigen",type=int,help='number of eigenvalues required',default=5)
    parser.add_argument("-e","--eps",type=float,help='Sparsity of the matrix',default=1E-2)
    args = parser.parse_args()

    N = args.size
    eps = args.eps
    neigen = args.neigen

    
    A = digaonal_dominant(N,eps)

    start_davidson = time.time()
    eigenvalues, eigenvectors = davidson_solver(A,neigen)
    end_davidson = time.time()
    print("davidson : ", end_davidson - start_davidson, " seconds")

    # Begin Numpy diagonalization of A
    start_numpy = time.time()
    E,Vec = np.linalg.eigh(A)
    end_numpy = time.time()
    print("numpy : ", end_numpy - start_numpy, " seconds")

    for i in range(neigen):
        print("%d % f  % f" %(i,eigenvalues[i],E[i]))