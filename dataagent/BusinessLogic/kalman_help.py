import numpy as np

from copy import deepcopy


def make_positive_definite(A):
    C = (A + A.T)/2
    try:
        eigval, eigvec = np.linalg.eig(C)
        eigval[eigval < 0] = 0
        return eigvec.dot(np.diag(eigval)).dot(eigvec.T)

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(A)
        print('*'*1000)

'''
def make_positive_definite2(A):
    """Find the nearest positive-definite matrix to input

    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].

    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """

    B = (A + A.T) / 2
    _, s, V = la.svd(B)

    H = np.dot(V.T, np.dot(np.diag(s), V))

    A2 = (B + H) / 2

    A3 = (A2 + A2.T) / 2

    if isPD(A3):
        return A3

    spacing = np.spacing(la.norm(A))
    # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
    # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
    # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
    # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
    # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
    # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
    # `spacing` will, for Gaussian random matrixes of small dimension, be on
    # othe order of 1e-16. In practice, both ways converge, as the unit test
    # below suggests.
    I = np.eye(A.shape[0])
    k = 1
    while not isPD(A3):
        mineig = np.min(np.real(la.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1

    return A3


def isPD(B):
    """Returns true when input is positive-definite, via Cholesky"""
    try:
        _ = la.cholesky(B)
        return True
    except la.LinAlgError:
        return False

'''


def estimate_Q_from_eq37(Delta_x_list, P_j_list, sigma_cov_list):
    N = len(Delta_x_list)
    Q_hat = np.zeros_like(P_j_list[0])

    for j in range(N):
        Delta_x_j = Delta_x_list[j]
        P_j = P_j_list[j]
        sigma_cov = sigma_cov_list[j]

        # The bracketed term from eq. (37):
        # [Delta_x_j Delta_x_j^T + P_j - sigma_cov]
        Q_hat += Delta_x_j @ Delta_x_j.T + P_j - sigma_cov

    Q_hat /= N
    Q_hat = make_positive_definite(Q_hat)

    return Q_hat

