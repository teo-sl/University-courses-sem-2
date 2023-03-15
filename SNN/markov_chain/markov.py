import numpy as np


def convergence(A, t_curr, eps=1e-20):
    assert np.allclose(A.sum(axis=0), 1)
    t = t_curr.copy()
    t_next = t @ A.T
    while (abs(t_next - t)).sum() > eps:
        t = t_next
        t_next = t @ A.T
    return t_next

def get_nth_step(A, t_curr, n):
    t = t_curr.copy()
    for _ in range(n):
        t = t @ A.T
    return t

def get_nth_state_alternative(A, t_curr, n):
    A_n = np.linalg.matrix_power(A, n)
    return t_curr @ A_n.T

def convergence_stable_distr_matrix(A,t_0,eps=1e-20):
    assert np.allclose(A.sum(axis=0), 1)
    A_cur = A.copy()
    A_next = A_cur @ A
    while (abs(A_next - A_cur)).sum() > eps:
        A_cur = A_next
        A_next = A_cur @ A

    return t_0 @ A_next.T



def find_steady_state(A):
    assert np.allclose(A.sum(axis=0), 1)
    eval,evect = np.linalg.eig(A)
    index = np.argmin(np.abs(eval - 1))
    steady_state = evect[:, index].real
    steady_state /= np.sum(steady_state)
    return steady_state
def find_stable_distribution_matrix(A):
    ss = find_steady_state(A)
    return np.full_like(A,ss).T


def compute_H(A):
    n = A.shape[0]
    deg_out = np.sum(A, axis=1)
    idx_sink = np.argwhere(deg_out==0)
    H = np.zeros_like(A)
    for i in range(n):
        for j in range(n):
            if A[i,j]==1 and deg_out[i] != 0:
                H[i, j] = 1.0 / deg_out[i]
    return H,idx_sink

def compute_S(A):
    n = A.shape[0]
    H,idx_sink = compute_H(A)
    
    S = H.copy()
    for i in idx_sink:
        S[i,:] = 1.0 / n
    return S   

def compute_G(A,alpha = 0.85):
    n = A.shape[0]
    S = compute_S(A)
    G = alpha*S+(1-alpha)/n*np.ones((n,n))
    return G

def page_rank(G,alpha):
    G = compute_G(G,alpha)
    return find_steady_state(G)