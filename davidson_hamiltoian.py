#!/bin/python
import numpy as np


def digaonal_dominant(n,sparsity=1E-4):
    
    A = np.zeros((n,n))
    for i in range(0,n):
        A[i,i] = 1E3*np.random.rand() 
        #A[i,i] = i+1
    A = A + sparsity*np.random.randn(n,n) 
    A = (A.T + A)/2 
    return A

def diag_non_tda(n,sparsity=1E-4):

    A = digaonal_dominant(n)
    C = sparsity*np.random.rand(n,n)
    return np.block([ [A,C],[-C.T,-A.T] ])


def jacobi_correction(uj,A,thetaj):
    I = np.eye(A.shape[0])
    Pj = I-np.dot(uj,uj.T)
    rj = np.dot((A - thetaj*I),uj) 

    w = np.dot(Pj,np.dot((A-thetaj*I),Pj))
    return np.linalg.solve(w,rj)


def get_initial_guess(A,nvec):

    nrows, ncols = A.shape
    half = int(ncols/2)
    d = np.diag(A)
    index = np.argsort(d)
    guess = np.zeros((nrows,nvec))
    shift = int(0.25*nvec)

    for i in range(nvec):
        guess[index[half+i-shift],i] = 1
    
    return guess


def reorder_matrix(A):
    
    n = A.shape[0]
    tmp = np.zeros((n,n))

    index = np.argsort(np.diagonal(A))

    for i in range(n):
        for j in range(i,n):
            tmp[i,j] = A[index[i],index[j]]
            tmp[j,i] = tmp[i,j]
    return tmp

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

    V = get_initial_guess(A,k)
    
    print('\n'+'='*20)
    print("= Davidson Solver ")
    print('='*20)

    #invA = np.linalg.inv(A)
    #inv_approx_0 = 2*I - A
    #invA2 = np.dot(invA,invA)
    #invA3 = np.dot(invA2,invA)

    norm = np.zeros(2*neigen)

    # Begin block Davidson routine
    print("iter size norm (%e)" %tol)
    for i in range(itermax):
    
        # QR of V t oorthonormalize the V matrix
        # this uses GrahmShmidtd in the back
        V,R = np.linalg.qr(V)

        # form the projected matrix 
        T = np.dot(V.conj().T,np.dot(A,V))
        print(np.diag(T))

        # Diagonalize the projected matrix
        theta,s = np.linalg.eig(T)

        # organize the eigenpairs
        index = np.argsort(theta.real)
        theta  = theta[index]
        s = s[:,index]

        # Ritz eigenvector
        q = np.dot(V,s)

        # compute the residual append append it to the 
        # set of eigenvectors
        ind0 = np.where(theta>0,theta,np.inf).argmin()
        for jj in range(2*neigen):


            j = ind0 + jj - int(0.25*2*neigen)

            # residue vetor
            res = np.dot((A - theta[j]*I),q[:,j]) 
            norm[jj] = np.linalg.norm(res)

            # correction vector
            if(jacobi):
            	delta = jacobi_correction(q[:,j],A,theta[j])
            else:
            	delta = res / (theta[j]-Adiag+1E-16)

            delta /= np.linalg.norm(delta)

            # expand the basis
            V = np.hstack((V,delta.reshape(-1,1)))

        # comute the norm to se if eigenvalue converge
        print(" %03d %03d %e" %(i,V.shape[1],np.max(norm)))
        if np.all(norm < tol):
            print("= Davidson has converged")
            break
        
    return theta[ind0:ind0+neigen], q[:,ind0:ind0+neigen]


if __name__ == "__main__":
 
    import time
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-s","--size",type=int,help='Size of the matrix',default=100)
    parser.add_argument("-n","--neigen",type=int,help='number of eigenvalues required',default=5)
    parser.add_argument("-e","--eps",type=float,help='Sparsity of the matrix',default=1E-2)
    parser.add_argument("-t","--tol",type=float,help='tolerance',default=1E-4)
    parser.add_argument("-j","--jacobi",action="store_true",help='jacobi correction')
    args = parser.parse_args()

    N = args.size
    eps = args.eps
    tol = args.tol
    neigen = args.neigen
    dojacobi = args.jacobi

    # create the matrix
    #A = digaonal_dominant(N,eps)
    A = diag_non_tda(N,eps)
    #A = reorder_matrix(np.loadtxt('bse_singlet.dat'))
    #A = np.loadtxt('bse_singlet.dat')

    # begin Davidson diagonalization
    start_davidson = time.time()
    eigenvalues, eigenvectors = davidson_solver(A,neigen,tol=tol,jacobi=dojacobi)
    end_davidson = time.time()
    print("davidson : ", end_davidson - start_davidson, " seconds")

    # Begin Numpy diagonalization of A
    start_numpy = time.time()
    E,Vec = np.linalg.eig(A)
    E = np.sort(E)
    E = E[E>0]

    end_numpy = time.time()
    print("numpy    : ", end_numpy - start_numpy, " seconds")

    for i in range(neigen):
        print("%d % f  % f" %(i,eigenvalues[i],E[i]))