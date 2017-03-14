import dask.array as da
from scipy.linalg import eig


# Dask DMD function

def dmd_dask(D,r):

    # offsets
    X_da = D[:,:-1]
    Y_da = D[:,1:]

    # SVD
    u,s,v = da.linalg.svd(X_da)

    # rank truncaction
    u = u[:,:r]
    Sig = da.diag(s)[:r,:r]
    v = v.conj().T[:,:r]

    # build A tilde
    Atil = da.dot(da.dot(da.dot(u.conj().T, Y_da), v), da.linalg.inv(Sig))
    mu,W = eig(Atil)

    # build DMD modes
    Phi = da.dot(da.dot(da.dot(Y_da, v), inv(Sig)), W)

    return(mu,Phi,s)

def dmd_evolve_dask(X0,mu,Phi,t):
    # compute time evolution
    b = da.dot(pinv(Phi), X0)
    r = Phi.shape[1]
    Psi = np.zeros([r, len(t)], dtype='complex')
    for i,_t in enumerate(t):
        Psi[:,i] = multiply(power(mu, _t), b)

    return(Psi)
