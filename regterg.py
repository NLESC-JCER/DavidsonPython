import numpy as np
import scipy.linalg as spla


def digaonal_dominant(n,sparsity=1E-4,val=None):
    
    A = np.zeros((n,n))
    A = A + sparsity*np.random.randn(n,n) 
    A = (A.T + A)/2 
    for i in range(0,n):
    	if val is not None:
    		A[i,i] = val
    	else:
        	A[i,i] = i + 1 
    return A

def regetrg(H,S,nvec,tol=1E-6):
	""" iterative solution of (H - e S) Psi = 0. """

	nparam = H.shape[0]
	nparamx = H.shape[0]
	nvecx = min(nparam/2,100) # maxsize of the basis

	evc = np.eye(nparam,nvec) #eigenvectors
	e = np.zeros(nvec) # eigenvalues
	ethr = 1E-8

	maxter = 100
	psi = evc

	for kter in range(maxter):

		psi,R = np.linalg.qr(psi)

		if(psi.shape[1] > nvecx):
			psi = psi_vr[:,:nvec]

		hr = np.dot(psi.T,np.dot(H,psi))
		ew, vr = np.linalg.eigh(hr)

		#ritz vector
		psi_vr = np.dot(psi,vr)

		# extend the basis with (H-eS)|psi> * Vr
		x = np.dot(H ,psi_vr) - ew * np.dot(S,psi_vr)
		eps = np.sum(np.linalg.norm(x[:,:nvec],axis=0))/x.shape[1]

		for i in range(x.shape[1]):
			delta = x[:,i] / (ew[i] - np.diag(H) + 1E-8)
			delta /= np.linalg.norm(delta)
			psi = np.hstack((psi,delta.reshape(-1,1)))

		print("iter %03d, norm %e" %(kter,eps))
		if eps < tol:
			break




	return ew[:nvec], psi_vr[:,:nvec]


if __name__ == "__main__":

	N=250
	A = digaonal_dominant(N,1E-1)
	S = np.eye(N)
	eig, vec = regetrg(A,S,5)
	print(eig)
	print(vec)

