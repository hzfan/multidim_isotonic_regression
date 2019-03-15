from sklearn.isotonic import IsotonicRegression

import cvxopt
import quadprog
import numpy as np


def cvxopt_solve_qp(P, q, G=None, h=None, A=None, b=None):
    P = (P + P.T)  # make sure P is symmetric
    args = [cvxopt.matrix(P), cvxopt.matrix(q)]
    if G is not None:
        args.extend([cvxopt.matrix(G), cvxopt.matrix(h)])
        if A is not None:
            args.extend([cvxopt.matrix(A), cvxopt.matrix(b)])
    sol = cvxopt.solvers.qp(*args)
    if 'optimal' not in sol['status']:
        return None
    return np.array(sol['x']).reshape((P.shape[1],))


def multidim_isotonic(x, a, c):
    n = a.size
    u = []
    v = []
    print(x)
    for i in range(n):
        t = np.all(x[i] <= x, axis=1)
        t[i] = 0
        num = t.sum()
        if num > 0:
            vidx = np.nonzero(t)[0]
            uidx = np.ones_like(vidx) * i
            u.extend(uidx)
            v.extend(vidx)
    u = np.array(u).astype(int)
    v = np.array(v).astype(int)
    K = u.size
    print(K)
    # P
    P = np.zeros([n + K, n + K])
    P[:n, :n] = np.eye(n)
    # P[n:, n:] = np.eye(K) * c
    # q
    q = np.ones(n + K)
    q[:n] = -2 * a
    q[n:] *= c
    # q[n:] = 0
    # G
    G = np.zeros([K * 2, n + K])
    G[np.arange(K, dtype=int), u.astype(int)] = 1
    G[np.arange(K, dtype=int), v.astype(int)] = -1
    G[:K, n:] = -np.eye(K)
    G[K:, n:] = -np.eye(K)
    # h
    h = np.zeros(K * 2)
    sol = cvxopt_solve_qp(P, q, G, h)
    x = sol[:n]
    e = sol[n:]
    return x, e




x = np.array([np.arange(7), np.arange(7)]).T
y = np.array([0, -1, 6, 5, 8, 7, 9])

y_, e = multidim_isotonic(x, y, .1)
# print(y_)
# print(e)



