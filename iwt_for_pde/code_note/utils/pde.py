import numpy as np
from utils.aiwt import restrain
from utils.iwavelets import dxphi0

def theta1_2d(J, j0, a, b, C, cjk, cjknum, zJc, zJcnum):
    delta = (b - a) / (2 ** J)
    xk = a + zJc * delta
    xk = xk.reshape((1,-1))
    theta1d, theta2d = Ii1_2d(J, j0, cjk, cjknum, zJc, xk, a, b, C)
    return theta1d, theta2d


def Ii1_2d(J, j0, cjk, cjknum, zJc, x, a, b, C):
    k = np.arange(2 ** j0+1).reshape((1,-1))
    rr = restrain(j0,k,J,zJc)
    ss1, ss2 = dxphi0(j0,k,a,b,x)
    Ii1d = np.dot(ss1, rr)
    Ii2d = np.dot(ss2, rr)

    for j in range(j0+1, J):
        if j == 0:
            continue
        elif j > j0 and cjknum[j] == 0:
            break
        k = cjk[j].reshape((1,-1))
        ss1, ss2 = dxphi0(j + 1,2 * k + 1,a,b,x)
        Ii1d = Ii1d + np.dot(ss1, C[j-1])
        Ii2d = Ii2d + np.dot(ss2, C[j-1])
    return Ii1d, Ii2d


def Vd(Vk,Re,theta1d,theta2d):
    V = Vk.reshape((-1,1))
    s1 = np.dot(theta2d, V)
    s2 = np.dot(theta1d, V)
    Vd1 = s1 / Re - V * s2
    return Vd1


def M012(Vk,Re,theta1d,theta2d):
    Vk = Vk.reshape((-1,1))
    Vd1 = Vd (Vk,Re,theta1d,theta2d)
    M00 = theta2d / Re
    Vkdiag = np.diag(Vk.reshape(-1))
    M11 = -np.dot(Vkdiag, theta1d)
    Vd1diag = np.diag(Vd1.reshape(-1))
    M22 = np.dot(theta1d, Vd1diag)
    return M00, M11, M22


def Precise(V0,Re,theta1d,theta2d,gJcnum,tao):
    V0 = V0.reshape((-1, 1))
    N = 10
    m = 2 ** N
    deltat = tao / m
    M00, M11, M22 = M012(V0,Re,theta1d,theta2d)
    r0 = np.dot(M11, V0)
    r1 = np.dot(M22, V0) + np.dot(M11, np.dot((M00 + M11), V0))
    I = np.eye(gJcnum, gJcnum)
    deltatH = deltat * M00
    deltatH2 = np.dot(deltatH, deltatH)
    Ta = deltatH + np.dot(deltatH2, (I + deltatH / 3 + deltatH2 / 12)) / 2
    for _ in range(N):
        Ta = 2 * Ta + np.dot(Ta, Ta)
    T = I + Ta
    invM00 = np.linalg.inv(M00)
    temp = r0 + np.dot(invM00, r1)
    V1 = np.dot(T, (V0 + np.dot(invM00, temp))) - np.dot(invM00, (temp + tao * r1))
    return V1
