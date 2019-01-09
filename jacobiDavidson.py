#!/bin/python

from __future__ import division
import math
import numpy as np
from scipy.sparse.linalg import gmres
import time

''' Block Davidson, Joshua Goings (2013)

    Block Davidson method for finding the first few
	lowest eigenvalues of a large, diagonally dominant,
    sparse Hermitian matrix (e.g. Hamiltonian)
'''

n = 1200				# Dimension of matrix
tol = 1e-8				# Convergence tolerance
mmax = n//2				# Maximum number of iterations	

''' Create sparse, diagonally dominant matrix A with 
	diagonal containing 1,2,3,...n. The eigenvalues
    should be very close to these values. You can 
    change the sparsity. A smaller number for sparsity
    increases the diagonal dominance. Larger values
    (e.g. sparsity = 1) create a dense matrix
'''

sparsity = 0.001
A = np.zeros((n,n))
for i in range(0,n):
    A[i,i] = i + 1 
A = A + sparsity*np.random.randn(n,n) 
A = (A.T + A)/2 


k = 8					# number of initial guess vectors 
eig = 4					# number of eignvalues to solve 
t = np.eye(n,k)			# set of k unit vectors as guess
V = np.zeros((n,n))		# array of zeros to hold guess vec
I = np.eye(n)			# identity matrix same dimen as A

# Begin block Davidson routine

start_davidson = time.time()
it = 0;
for m in range(k,mmax,k):

    if m <= k:

        # Create the V matrix by renormalizeall all the trial vectors
        for j in range(0,k):
            V[:,j] = t[:,j]/np.linalg.norm(t[:,j])
        theta_old = 1 

    elif m > k:
        theta_old = theta[:eig]

    # QR of V t oorthonormalize the V matrix
    # this uses GrahmShmidtd in the back
    V,R = np.linalg.qr(V)

    # for the projected matrix
    T = np.dot(V[:,:(m+1)].T,np.dot(A,V[:,:(m+1)]))

    # Diagonalize the projected matrix
    THETA,S = np.linalg.eig(T)

    # sort the eigenvalues
    idx = THETA.argsort()
    theta = THETA[idx]
    s = S[:,idx]

    # compute the residual and append it to the 
    # set of eigenvectors
    for j in range(0,k):

        uj = np.dot(V[:,:(m+1)],s[:,j])
        Pj = I-np.dot(uj,uj.T)
        rj = np.dot((A - theta[j]*I),uj) 

        w = np.dot(Pj,np.dot((A-theta[j]*I),Pj))
        q = np.linalg.solve(w,rj)
        #q=gmres(w,rj,tol=1E-)
        V[:,(m+j+1)] = q

    # comute the norm to se if eigenvalue converge
    norm = np.linalg.norm(theta[:eig] - theta_old)
    if norm < tol:
        break
    print("iteration %d, norm %f" %(it,norm))
    it += 1

end_davidson = time.time()

# End of block Davidson. Print results.

print("davidson = ", theta[:eig],";",\
    end_davidson - start_davidson, "seconds")

# Begin Numpy diagonalization of A

start_numpy = time.time()

E,Vec = np.linalg.eig(A)
E = np.sort(E)

end_numpy = time.time()

# End of Numpy diagonalization. Print results.

print("numpy = ", E[:eig],";",\
     end_numpy - start_numpy, "seconds")