#!/bin/python
import numpy as np


def digaonal_dominant(n,sparsity=1E-4):
    
    A = np.zeros((n,n))
    for i in range(0,n):
        A[i,i] = i + 1 
    A = A + sparsity*np.random.randn(n,n) 
    A = (A.T + A)/2 
    return A

def jacobi_correction(uj,A,thetaj):
    I = np.eye(A.shape[0])
    Pj = I-np.dot(uj,uj.T)
    rj = np.dot((A - thetaj*I),uj) 

    w = np.dot(Pj,np.dot((A-thetaj*I),Pj))
    return np.linalg.solve(w,rj)


def davidson_solver(A, neigen, tol=1E-6, itermax = 1000, jacobi=False):
    """Davidosn solver for eigenvalue problem

    Args :
        A (numpy matrix) : the matrix to diagonalize
        neigen (int)     : the number of eigenvalue requied
        tol (float)      : the rpecision required
        itermax (int)    : the maximum number of iteration
        jacobi (bool)    : do the jacobi correction
    Returns :
        eigenvalues (array) : lowest eigenvalues
        eigenvectors (numpy.array) : eigenvectors
    """
    n = A.shape[0]
    k = 2*neigen            # number of initial guess vectors 
    V = np.eye(n,k)         # set of k unit vectors as guess
    I = np.eye(n)           # identity matrix same dimen as A
    Adiag = np.diag(A)

    print('\n'+'='*20)
    print("= Davidson Solver ")
    print('='*20)

    # Begin block Davidson routine
    print("iter size norm (%e)" %tol)
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
            if(jacobi):
            	delta = jacobi_correction(q[:,j],A,theta[j])
            else:
            	delta = res / (theta[j]-Adiag)
            delta /= np.linalg.norm(delta)

            # expand the basis
            V = np.hstack((V,delta.reshape(-1,1)))

        # comute the norm to se if eigenvalue converge
        print(" %03d %03d %e" %(i,V.shape[1],norm))
        if norm < tol:
            print("= Davidson has converged")
            break
        
    return theta[:neigen], q[:,:neigen]


if __name__ == "__main__":
 
    import time
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-s","--size",type=int,help='Size of the matrix',default=100)
    parser.add_argument("-n","--neigen",type=int,help='number of eigenvalues required',default=5)
    parser.add_argument("-e","--eps",type=float,help='Sparsity of the matrix',default=1E-2)
    parser.add_argument("-j","--jacobi",action="store_true",help='jacobi correction')
    args = parser.parse_args()

    N = args.size
    eps = args.eps
    neigen = args.neigen
    dojacobi = args.jacobi

    # create the matrix
    A = digaonal_dominant(N,eps)

    # begin Davidson diagonalization
    start_davidson = time.time()
    eigenvalues, eigenvectors = davidson_solver(A,neigen,jacobi=dojacobi)
    end_davidson = time.time()
    print("davidson : ", end_davidson - start_davidson, " seconds")

    # Begin Numpy diagonalization of A
    start_numpy = time.time()
    E,Vec = np.linalg.eigh(A)
    end_numpy = time.time()
    print("numpy    : ", end_numpy - start_numpy, " seconds")

    for i in range(neigen):
        print("%d % f  % f" %(i,eigenvalues[i],E[i]))