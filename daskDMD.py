import dask.array as da
import scipy.linalg as la
from scipy import multiply, power
import numpy as np
from warnings import warn


def eig_dask(A,nofIt = None):
    """
        A dask eigenvalue solver: assumes A is symmetric and used the QR method
        to find eigenvalues and eigenvectors.
        nofIt: number of iterations (default is the size of A)
    """
    A_new = A
    if nofIt is None:
        nofIt = A.shape[0]
    V = da.eye(A.shape[0],100)
    for i in range(nofIt):
        Q,R = da.linalg.qr(A_new)
        A_new = da.dot(R,Q)
        V = da.dot(V,Q)
    return(da.diag(A_new),V)

# Dask DMD function

def dmd_dask(D,r,eig=None):
    """
        A dask implementation of Dynamic Model Decomposition.

        Args:
            D - dask array
                D is a d x T array for which each column corresponds to
                an observation at a specific time.
            r - integer
                number of components
            eig - string or None
                eig indicates the method to use to calculate eigenvalues
                eig='None' corresponds to use numpy.linalg.eig function (not out-of-core)
                eig='dask' uses a dask eigensolver based on QR decomposition

        Returns:
            mu - dask.array of length r
                dmd eigenvalues
            Phi - dask.array of dimensions d x r
                dmd modes
            s - dask.array
                singular values of D[:,:-1] - useful for determining the rank

        Examples:
            >>> mu,Phi,s = dmd_dask(D,r,eig=None)

    """

    # offsets
    X_da = D[:,:-1]
    Y_da = D[:,1:]

    # SVD
    u,s,v = da.linalg.svd(X_da)

    # rank truncaction
    u = u[:,:r]
    Sig = da.diag(s)[:r,:r]
    Sig_inv = da.diag(1/s)[:r,:r]
    v = v.conj().T[:,:r]

    # build A tilde
    Atil = da.dot(da.dot(da.dot(u.conj().T, Y_da), v), Sig_inv)
    if eig is None:
        mu,W = la.eig(Atil)
    elif eig=='dask':
        mu,W = eig_dask(Atil,10)

    # build DMD modes
    Phi = da.dot(da.dot(da.dot(Y_da, v), Sig_inv), W)

    return(mu,Phi,s)

def dmd_evolve_dask(X0,mu,Phi,t):
    """
        dmd_evolve_dask evolves the dmd components to time t starting from X0

        Args:
            X0 - dask.array of length d
                the initial observation D[:,0]
            mu - the dmd eigenvalues
            Phi - the dmd modes
            t - an array of times
            #TODO for now the evolution is by increment of 1,
            should allow for a smaller timestep

        Returns:
            Psi - dask.array of dimensions r x t

    """
    # calculate starting point
    b = da.dot(pinv_SVD(Phi), X0)
    # rank
    r = Phi.shape[1]
    # initialize Psi
    Psi = np.zeros([r, len(t)], dtype='complex')
    #Psi = da.zeros([r,len(t)],chunks = (r,len(t)),dtype='complex')
    # evolve Psi
    for i,_t in enumerate(t):
        Psi[:,i] = multiply(power(mu, _t), b)

    return(Psi)

def check_dmd(D, mu, Phi, show_warning=True):
    X = D[:,0:-1]
    Y = D[:,1:]
    b = np.allclose(Y, np.dot(np.dot(np.dot(Phi, np.diag(mu)), la.pinv(Phi)), X))
    if not b and show_warning:
        warn('dmd result does not satisfy Y=AX')

def check_dmd_dask(D, mu, Phi, show_warning=True):
    """
        Checks how close the approximation using DMD is to the original data.

        Returns:
            None if the difference is within the tolerance
            Displays a warning otherwise.
    """
    X = D[:,0:-1]
    Y = D[:,1:]
    #Y_est = da.dot(da.dot(da.dot(Phi, da.diag(mu)), pinv_SVD(Phi)), X)
    Phi_inv = pinv_SVD(Phi)
    PhiMu = da.dot(Phi, da.diag(mu))
    #Y_est = da.dot(da.dot(PhiMu, Phi_inv), X)
    Y_est = da.dot(PhiMu, da.dot(Phi_inv, X))
    diff = da.real(Y - Y_est)
    res = da.fabs(diff)
    rtol = 1.e-8
    atol = 1.e-5

    if da.all(res < atol + rtol*da.fabs(da.real(Y_est))).compute():
        return(None)
    else:
    #if not b and show_warning:
        warn('dmd result does not satisfy Y=AX')

def pinv_SVD(X):
    """
        a function to find a pseudo-inverse in dask using svd
    """
    u,s,v = da.linalg.svd(X)
    S_inv = da.diag(1/s)
    X_inv = da.dot(v.T.conj(),da.dot(S_inv,u.T.conj()))
    return(X_inv)
