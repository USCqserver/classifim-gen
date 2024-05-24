# SPDX-License-Identifier: GPL-3.0-or-later
import numpy as np

def pade_fit(xs, ys, n, m, reg=1e-13, reg_mode='l2'):
    """
    Fit y ~= p(x) / q(x) for polynomials p and q.

    Args:
        xs: x-coordinates of points to fit
        ys: y-coordinates of points to fit
        n: degree of numerator polynomial p
        m: degree of denominator polynomial q
        reg: L2 regularization in the least-squares fit
            If reg_mode is 'cutoff', this is used as rcond in np.linalg.lstsq.
        reg_mode: 'l2' or 'cutoff'

    Returns:
        p, q: polynomial functions such that y ~= p(x) / q(x)
    """
    num_points = len(xs)
    num_coeffs = n + m + 1
    assert num_points >= num_coeffs
    assert xs.dtype == ys.dtype

    # Design matrix A and target vector B for the linear system A @ coeffs = ys
    A = np.empty((num_points, num_coeffs), dtype=xs.dtype)
    A[:, :n+1] = xs[:, None]**np.arange(n+1)[None, :]
    A[:, n+1:] = -ys[:, None] * xs[:, None]**np.arange(1, m+1)[None, :]

    # Compute least-squares solution
    if reg_mode == 'cutoff':
        coeffs = np.linalg.lstsq(A, B, rcond=reg)[0]
    elif reg_mode == 'l2':
        coeffs = np.linalg.solve(
            A.conj().T @ A + reg * np.eye(num_coeffs),
            A.conj().T @ ys)
    else:
        raise ValueError(f"Invalid reg_mode: {reg_mode}")

    # Extract coefficients for p(x) and q(x)
    p_coeffs = coeffs[:n+1]
    q_coeffs = np.insert(coeffs[n+1:], 0, 1)

    Poly = np.polynomial.polynomial.Polynomial
    return Poly(p_coeffs), Poly(q_coeffs)


def imag2real(zs, us, xs=None):
    """Translate from imaginary axis to real axis"""
    zs = np.array(zs)
    us = np.array(us)
    assert zs.shape == us.shape
    assert zs.ndim == 1
    num_points = zs.shape[0]
    # Effective number of degrees of freedom of Pade approximant:
    num_used = int(0.9 * num_points + 0.5)
    m = (num_used - 1) // 2
    n = num_used - m - 1
    if xs is None:
        xs = np.linspace(-4.,4.,300)
    scale = max(np.max(np.abs(zs)), np.max(np.abs(xs)))
    p, q = pade_fit(zs / scale, us, n, m)
    scaled_xs = xs / scale
    return xs, p(scaled_xs) / q(scaled_xs)


def polyexp_fit(zs,us,n=10,bnds=[-5.,5.]):
    """Perform an analytic continuation using a polyexp fit"""
    def f(lamb):
      """Polynomial function"""
      return lambda x: np.exp(np.sum([lamb[i]*x**i for i in range(len(lamb))]))
    from scipy.integrate import quad
    def errorf(lamb):
        """Function to compute the error"""
        fx = f(lamb) # get the function to evaluate
        def gz(tau):
            """Do the transformation"""
            fint = lambda x: fx(x)*np.exp(-tau*x)/(1+np.exp(beta*x))
            out = quad(fint,bnds[0],bnds[1])
    from scipy.optimize import minimize
    raise NotImplementedError()

