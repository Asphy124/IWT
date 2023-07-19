import numpy as np
import sys


def xphi0(j , k, xmin, xmax, x):
    R = 3.2
    delta = (xmax - xmin) / 2 ** j
    xk = xmin + delta * k
    t = np.ones((1,k.shape[1]))
    if x.shape[0] == 1:
        x = x.T
    x = x * t
    t = np.ones((x.shape[0], 1))
    if xk.shape[1] == 1:
        xk = xk.T
    xk = t * xk
    deltaxk = x - xk
    a = (abs(deltaxk) < sys.float_info.epsilon)
    v_sad1x_Gauss = np.vectorize(sax_Gauss)
    idx = np.where(deltaxk != 0)
    c = np.zeros_like(deltaxk)
    c[idx] = v_sad1x_Gauss(deltaxk[idx], delta, R)
    return a + c


def sax_Gauss(deltaxk, delta, R):
    mm = (R**2) * (delta**2)
    t = np.pi * deltaxk / delta
    ss = np.sin(t)
    ee = np.exp(-np.square(deltaxk) / (2 * mm))
    return ss * ee / t


def dxphi0(j,k,xmin,xmax,x):
    R = 3.2
    delta = (xmax - xmin) / (2 ** j)
    xk = xmin + delta * k
    t = np.ones((1,k.shape[1]))
    if x.shape[0] == 1:
        x = x.T
    x = x * t
    t = np.ones((x.shape[0], 1))
    if xk.shape[1] == 1:
        xk = xk.T
    xk = t * xk
    deltaxk = x - xk
    a = (abs(deltaxk) < sys.float_info.epsilon)
    mm = R * delta
    d1theta = 0 * a
    d2theta = -(3 + (np.pi**2)*(R**2)) / (3 * mm ** 2) * a
    v_sad1x_Gauss = np.vectorize(sad1x_Gauss)
    v_sad2x_Gauss = np.vectorize(sad2x_Gauss)
    idx = np.where(deltaxk != 0)
    c1 = np.zeros_like(deltaxk)
    c1[idx] = v_sad1x_Gauss(deltaxk[idx], delta, R)
    c2 = np.zeros_like(deltaxk)
    c2[idx] = v_sad2x_Gauss(deltaxk[idx], delta, R)
    d1theta = d1theta + c1
    d2theta = d2theta + c2
    return d1theta, d2theta


def sad1x_Gauss(deltaxk, delta, R):
    mm = (R**2) * (delta**2)
    t = np.pi * deltaxk / delta
    s = np.sin(t)
    c = np.cos(t)
    e = np.exp(-np.square(deltaxk) / (2 * mm))
    se = s * e
    ce = c * e
    d1theta = ce/deltaxk - se*delta/(np.pi * deltaxk**2) - se * delta / (np.pi * mm)
    return d1theta


def sad2x_Gauss(deltaxk, delta, R):
    mm = (R**2) * (delta**2)
    t = np.pi * deltaxk / delta
    s = np.sin(t)
    c = np.cos(t)
    e = np.exp(-np.square(deltaxk) / (2 * mm))
    se = s * e
    ce = c * e
    d2theta = -np.pi * se / (delta * deltaxk) - 2 * ce / (deltaxk ** 2) - 2 * ce / mm + \
              2 * se * delta / (np.pi * (deltaxk ** 3)) + se * delta / (np.pi * mm * deltaxk) + \
              deltaxk * se * delta / (np.pi * mm * mm)
    return d2theta
