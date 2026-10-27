"""
softdtw_div.py
==============
Core implementation of  "Differentiable Divergences Between Time Series"
(Blondel, Mensch, Vert - AISTATS 2021)  with numba (CPU).

Implemented discrepancies (squared Euclidean ground cost, eq. (1) of the paper):

    euclid     : plain Euclidean distance (equal-length series only)
    dtw        : dynamic time warping, eq. (2)         (sub-gradient via backtracking)
    sdtw       : soft-DTW, eq. (4)
    sdtw_div   : soft-DTW divergence  D_gamma(X,Y) = sdtw(X,Y) - 1/2 sdtw(X,X) - 1/2 sdtw(Y,Y)
    sharp      : sharp soft-DTW, eq. (14)   <E_gamma(C), C>
    sharp_div  : sharp divergence S_gamma(X,Y)  (same correction terms)

Conventions
-----------
* A time series is a float64 array of shape (T, d).  A dataset is (N, T, d).
* DP tables use 1-based indexing with padding (index 0 = border), like the paper's Algorithms 1-4.
* P[i,j,k] = transition probabilities of the random walk on the DAG:
      k=0 : came from (i, j-1)    k=1 : from (i-1, j-1)    k=2 : from (i-1, j)
"""
import numpy as np
from numba import njit, prange
from scipy.optimize import minimize

KINDS = ("euclid", "dtw", "sdtw", "sdtw_div", "sharp", "sharp_div")
_CODE = {"euclid": 0, "dtw": 1, "sdtw": 2, "sharp": 3}


# --------------------------------------------------------------------------- #
#  Ground cost
# --------------------------------------------------------------------------- #
@njit(cache=True)
def _cost(X, Y):
    """[C]_{ij} = 1/2 ||x_i - y_j||^2   (eq. 1). Computed by explicit differences so that
    C(X,X) has an exactly-zero diagonal (assumption A.2)."""
    m, n, d = X.shape[0], Y.shape[0], X.shape[1]
    C = np.empty((m, n))
    for i in range(m):
        for j in range(n):
            s = 0.0
            for k in range(d):
                t = X[i, k] - Y[j, k]
                s += t * t
            C[i, j] = 0.5 * s
    return C


# --------------------------------------------------------------------------- #
#  Algorithm 1: soft-DTW value + transition probabilities
# --------------------------------------------------------------------------- #
@njit(cache=True)
def _sdtw_forward(C, gamma):
    m, n = C.shape
    V = np.full((m + 1, n + 1), np.inf)
    V[0, 0] = 0.0
    P = np.zeros((m + 2, n + 2, 3))
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            r0 = V[i, j - 1]
            r1 = V[i - 1, j - 1]
            r2 = V[i - 1, j]
            rmin = min(r0, min(r1, r2))
            e0 = np.exp(-(r0 - rmin) / gamma)
            e1 = np.exp(-(r1 - rmin) / gamma)
            e2 = np.exp(-(r2 - rmin) / gamma)
            s = e0 + e1 + e2
            V[i, j] = C[i - 1, j - 1] + rmin - gamma * np.log(s)  # C + softmin_gamma
            P[i, j, 0] = e0 / s
            P[i, j, 1] = e1 / s
            P[i, j, 2] = e2 / s
    P[m + 1, n + 1, 1] = 1.0  # edge case for Algorithm 2
    return V[m, n], P


# Algorithm 2: expected alignment E_gamma(C) = grad_C sdtw (padded (m+2, n+2) output)
@njit(cache=True)
def _expected_alignment(P):
    m = P.shape[0] - 2
    n = P.shape[1] - 2
    E = np.zeros((m + 2, n + 2))
    E[m + 1, n + 1] = 1.0
    for j in range(n, 0, -1):
        for i in range(m, 0, -1):
            E[i, j] = (P[i, j + 1, 0] * E[i, j + 1]
                       + P[i + 1, j + 1, 1] * E[i + 1, j + 1]
                       + P[i + 1, j, 2] * E[i + 1, j])
    return E


# Algorithm 3: directional derivative <E, Z>  (padded (m+1,n+1) table, answer at [m,n])
@njit(cache=True)
def _directional(P, Z):
    m, n = Z.shape
    Vd = np.zeros((m + 1, n + 1))
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            Vd[i, j] = (Z[i - 1, j - 1] + P[i, j, 0] * Vd[i, j - 1]
                        + P[i, j, 1] * Vd[i - 1, j - 1] + P[i, j, 2] * Vd[i - 1, j])
    return Vd


# Algorithm 4: Hessian-vector product  (d/d eps) E_gamma(C + eps Z)
# NB: we include the 1/gamma factor of d softmax so this is the exact Hessian of sdtw_gamma
@njit(cache=True)
def _hvp(P, E, Vd, gamma):
    m = P.shape[0] - 2
    n = P.shape[1] - 2
    Pd = np.zeros((m + 2, n + 2, 3))
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            v0 = Vd[i, j - 1]
            v1 = Vd[i - 1, j - 1]
            v2 = Vd[i - 1, j]
            s = P[i, j, 0] * v0 + P[i, j, 1] * v1 + P[i, j, 2] * v2
            Pd[i, j, 0] = P[i, j, 0] * (s - v0) / gamma
            Pd[i, j, 1] = P[i, j, 1] * (s - v1) / gamma
            Pd[i, j, 2] = P[i, j, 2] * (s - v2) / gamma
    Ed = np.zeros((m + 2, n + 2))
    for j in range(n, 0, -1):
        for i in range(m, 0, -1):
            Ed[i, j] = (Pd[i, j + 1, 0] * E[i, j + 1] + P[i, j + 1, 0] * Ed[i, j + 1]
                        + Pd[i + 1, j + 1, 1] * E[i + 1, j + 1] + P[i + 1, j + 1, 1] * Ed[i + 1, j + 1]
                        + Pd[i + 1, j, 2] * E[i + 1, j] + P[i + 1, j, 2] * Ed[i + 1, j])
    return Ed[1:m + 1, 1:n + 1].copy()


# --------------------------------------------------------------------------- #
#  Hard DTW (value + optimal alignment by backtracking)
# --------------------------------------------------------------------------- #
@njit(cache=True)
def _dtw_fwd(C):
    m, n = C.shape
    V = np.full((m + 1, n + 1), np.inf)
    V[0, 0] = 0.0
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            V[i, j] = C[i - 1, j - 1] + min(V[i, j - 1], min(V[i - 1, j - 1], V[i - 1, j]))
    return V


@njit(cache=True)
def _dtw_alignment(V):
    m = V.shape[0] - 1
    n = V.shape[1] - 1
    A = np.zeros((m, n))
    i, j = m, n
    while True:
        A[i - 1, j - 1] = 1.0
        if i == 1 and j == 1:
            break
        a = V[i, j - 1]
        b = V[i - 1, j - 1]
        c = V[i - 1, j]
        if b <= a and b <= c:
            i -= 1
            j -= 1
        elif a <= c:
            j -= 1
        else:
            i -= 1
    return A


# --------------------------------------------------------------------------- #
#  Value of the "base" discrepancy for a cost matrix (used by pairwise kernels)
# --------------------------------------------------------------------------- #
@njit(cache=True)
def _pair_value(C, code, gamma):
    m, n = C.shape
    if code == 0:  # euclid (equal length)
        s = 0.0
        for i in range(min(m, n)):
            s += C[i, i]
        return np.sqrt(2.0 * s)
    elif code == 1:
        return _dtw_fwd(C)[m, n]
    elif code == 2:
        return _sdtw_forward(C, gamma)[0]
    else:  # sharp = <E, C> = directional derivative in direction C
        v, P = _sdtw_forward(C, gamma)
        return _directional(P, C)[m, n]


@njit(parallel=True, cache=True)
def _pairwise(A, B, code, gamma):
    N, M = A.shape[0], B.shape[0]
    D = np.empty((N, M))
    for a in prange(N):
        for b in range(M):
            D[a, b] = _pair_value(_cost(A[a], B[b]), code, gamma)
    return D


@njit(parallel=True, cache=True)
def _self_terms(A, code, gamma):
    N = A.shape[0]
    out = np.empty(N)
    for a in prange(N):
        out[a] = _pair_value(_cost(A[a], A[a]), code, gamma)
    return out


# --------------------------------------------------------------------------- #
#  Public API: pairwise matrices
# --------------------------------------------------------------------------- #
def base_of(kind):
    return kind[:-4] if kind.endswith("_div") else kind


def raw_matrix(A, B, base, gamma=1.0):
    """Matrix of base discrepancies (no correction term). A:(N,T,d)  B:(M,T',d)."""
    return _pairwise(np.ascontiguousarray(A, dtype=np.float64),
                     np.ascontiguousarray(B, dtype=np.float64), _CODE[base], float(gamma))


def self_terms(A, base, gamma=1.0):
    return _self_terms(np.ascontiguousarray(A, dtype=np.float64), _CODE[base], float(gamma))


def discrepancy_matrix(A, B, kind, gamma=1.0, selfA=None, selfB=None):
    """Pairwise matrix for any kind in KINDS (divergences subtract 1/2 self terms)."""
    base = base_of(kind)
    D = raw_matrix(A, B, base, gamma)
    if kind.endswith("_div"):
        sa = self_terms(A, base, gamma) if selfA is None else selfA
        sb = self_terms(B, base, gamma) if selfB is None else selfB
        D = D - 0.5 * sa[:, None] - 0.5 * sb[None, :]
    return D


# --------------------------------------------------------------------------- #
#  Value + gradient w.r.t. X  (for barycenters / interpolation)
# --------------------------------------------------------------------------- #
def _value_and_cost_grad(C, base, gamma):
    """Returns (value, dvalue/dC)."""
    if base == "dtw":
        V = _dtw_fwd(C)
        return V[-1, -1], _dtw_alignment(V)
    v, P = _sdtw_forward(C, gamma)
    E = _expected_alignment(P)
    if base == "sdtw":
        return v, E[1:-1, 1:-1].copy()
    # sharp: value <E,C>;  grad_C = E + H C   (H = Hessian of sdtw)
    Vd = _directional(P, C)
    G = E[1:-1, 1:-1] + _hvp(P, E, Vd, gamma)
    return Vd[-1, -1], G


def _vjp_xy(X, Y, G):  # (J_X C(X,Y))^T G   (eq. 18)
    return X * G.sum(1, keepdims=True) - G @ Y


def _vjp_xx(X, G):  # (J_X C(X,X))^T G   (eq. 19)
    S = G + G.T
    return X * S.sum(1, keepdims=True) - S @ X


def value_and_grad(X, Y, kind, gamma=1.0, with_const=True):
    """Value of  kind(X, Y)  and its gradient w.r.t. X (squared Euclidean cost).
    with_const=False drops the X-independent term -1/2 D(Y,Y) (saves time in optimisation)."""
    if kind == "euclid":
        return 0.5 * np.sum((X - Y) ** 2), X - Y
    base = base_of(kind)
    v, G = _value_and_cost_grad(_cost(X, Y), base, gamma)
    g = _vjp_xy(X, Y, G)
    if kind.endswith("_div"):
        vxx, Gxx = _value_and_cost_grad(_cost(X, X), base, gamma)
        v -= 0.5 * vxx
        g = g - 0.5 * _vjp_xx(X, Gxx)
        if with_const:
            v -= 0.5 * _value_and_cost_grad(_cost(Y, Y), base, gamma)[0]
    return v, g


# --------------------------------------------------------------------------- #
#  Barycenter (Frechet mean, eq. 8) / interpolation
# --------------------------------------------------------------------------- #
def barycenter(Ys, kind, gamma=1.0, weights=None, init=None, maxiter=100):
    """argmin_X sum_i w_i kind(X, Y_i) with L-BFGS-B.  Ys: (k, T, d), equal length.
    Init follows the paper: Euclidean mean for biased kinds; for *_div kinds, the solution
    of the biased counterpart."""
    Ys = np.asarray(Ys, dtype=np.float64)
    k, T, d = Ys.shape
    w = np.ones(k) if weights is None else np.asarray(weights, dtype=np.float64)
    mean = (w[:, None, None] * Ys).sum(0) / w.sum()
    if kind == "euclid":
        return mean
    if init is None:
        init = (barycenter(Ys, base_of(kind), gamma, w, None, maxiter)
                if kind.endswith("_div") else mean)

    def f(x):
        X = x.reshape(T, d)
        tot, grad = 0.0, np.zeros_like(X)
        for i in range(k):
            v, g = value_and_grad(X, Ys[i], kind, gamma, with_const=False)
            tot += w[i] * v
            grad += w[i] * g
        return tot, grad.ravel()

    res = minimize(f, np.asarray(init, dtype=np.float64).ravel(), jac=True,
                   method="L-BFGS-B", options={"maxiter": maxiter})
    return res.x.reshape(T, d)
