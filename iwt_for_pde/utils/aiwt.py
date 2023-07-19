import numpy as np
from utils.iwavelets import xphi0

def cjkfun(VJ, J, j0, a, b, eps0):
    numJ = 2**j0 + 1
    k0 = np.arange(0, numJ).reshape(1,-1)
    nn = k0 * (2**(J-j0))
    zJc0 = nn[0]
    ff2 = VJ[0,nn[0]].reshape(1,-1)
    cjk0 = []
    cjk = [np.array([0])]
    cjknum = []
    for j in range(j0, J):
        deltaj = (b-a)/2/(2**j)
        k = np.arange(0, 2**j).reshape((1,-1))
        nn = (2*k+1) * (2**(J-(j+1)))
        ff1 = VJ[0,nn[0]].reshape((1,-1))
        x = a + (2 * k + 1) * deltaj
        theta = xphi0(j0, k0, a, b, x)
        theta = theta.T
        s1 = np.dot(ff2 , theta)
        s2 = 0
        for j1 in range(j0, j):
            k1 = np.arange(0, 2 ** j1).reshape((1,-1))
            theta = xphi0(j1 + 1,2 * k1 + 1,a,b,x)
            s2 = s2 + np.dot(cjk0[j1][0][k1], theta.T)
        cjk0.append(ff1 - s1 - s2)
        cjkvalue = abs(ff1 - s1 - s2)
        kk = np.flatnonzero(cjkvalue >= eps0)
        num = kk.shape[0]
        if num > 0:
            cjk.append(kk)
            zJc0 = np.append(zJc0, (2 * kk + 1) * (2 ** (J - j - 1)), 0)
        cjknum.append(num)
        if j >  j0 + 2 and num == 0:
            cjknum += [0]*(J-j-1)
            break
    zJc = np.sort(zJc0).tolist()
    zJcnum = len(zJc)
    return cjk, cjknum, zJc, zJcnum


def z_g(j0, J, zjk, zjknum, zJc, zJcnum, a, b, M, CC):
    gJc = zJc.copy()
    gjk = zjk.copy()
    gjknum = zjknum.copy()
    for j in range(J):
        wt_1 = zjk[j]
        position_x = (2**J)/2**(j+1) * (2*wt_1 + 1)
        position_neighbor_len = (2**J)/2**(j+2)
        position_left_neighbor = position_x - position_neighbor_len
        position_right_neighbor = position_x + position_neighbor_len
        position_neighbor = np.append(position_left_neighbor, position_right_neighbor, axis=0)
        position_neighbor = np.sort(position_neighbor)
        gJc = np.union1d(gJc, position_neighbor)
        
        k = np.arange(2**(j+1))
        position_k = (2**J)/2**(j+2) * (2*k + 1)
        new_position = np.where(np.in1d(position_k, position_neighbor))[0]
        if j+2 > len(gjk):
            gjk.append(new_position)
            gjknum[j+1] = gjk[j+1].shape[0]
            break
        else:
            gjk[j+1] = np.union1d(gjk[j+1], new_position)
            gjknum[j+1] = gjk[j+1].shape[0]
    gJcnum = len(gJc)
    # gJc = zJc.copy()
    # gJcnum = zJcnum
    # gjk = zjk.copy()
    # gjknum = zjknum.copy()
    # for j in range(j0, J):
    #     xj = j+1
    #     deltax = (b - a) / (2 ** xj)
    #     jmin = max(j0+1, xj-M) 
    #     jmax = min(J, xj+M)
    #     for k in range(zjknum[j]):
    #         kk = zjk[j][k]
    #         xk = a + deltax * (2*kk + 1)
    #         jMax = len(gjk)
    #         xmin0 = xk - CC*(0.5**xj)
    #         xmax0 = xk + CC*(0.5**xj)
    #         for jj in range(jmin, jmax+1):
    #             deltaxjj=(b-a)/(2**jj)
    #             xmin = max(xmin0, a+deltaxjj)
    #             xmax = min(xmax0, b-deltaxjj)
    #             kmin = np.ceil((xmin-a)/deltaxjj)
    #             kmax = np.floor((xmax-a)/deltaxjj)
    #             if kmin < kmax:
    #                 delta = 2 ** (J - jj)
    #                 temp = np.arange(kmin*delta, kmax*delta+1, delta).tolist()
    #                 gJc = np.union1d(gJc, temp)
    #                 oe = np.mod(kmin,2)
    #                 if oe == 1:
    #                     jm = jj - 1
    #                     if jm >= j0:
    #                         temp1 = np.array([])
    #                         if jm <= jMax - 1:
    #                             temp1 = np.append(temp1, gjk[jm], 0)
    #                         temp2 = np.arange((kmin - 1) / 2, (kmax - 1) / 2 + 1)
    #                         temp3 = np.union1d(temp1,temp2)
    #                         temp3 = np.array(temp3, dtype=np.int64)
    #                         if jm < len(gjk):
    #                             gjk[jm] = temp3
    #                         else:
    #                             gjk.append(temp3) 
    #                         gjknum[jm] = temp3.shape[0]
    #                 elif oe == 0:
    #                     jm = jj - 2
    #                     if jm >= j0:
    #                         if jm > jMax - 1:
    #                             temp1 = np.array([])
    #                         else:
    #                             temp1 = np.append(np.array([]), gjk[jm], 0)
    #                         temp2 = np.arange((kmin - 2) / 4, (kmax - 2) / 4 + 1)
    #                         temp3 = np.union1d(temp1,temp2)
    #                         temp3 = np.array(temp3, dtype=np.int64)
    #                         if jm < len(gjk):
    #                             gjk[jm] = temp3
    #                         else:
    #                             gjk.append(temp3) 
    #                         gjknum[jm] = temp3.shape[0]
    # gJcnum = gJc.shape[0]
    return gjk, gjknum, gJc, gJcnum


def CompressInterp(KnowX, KnowP, NX):
    NP = []
    for i in range(len(NX)):
        x = NX[i]
        j = np.flatnonzero(KnowX == x)
        NP.append(KnowP[0][j[0]])
    NP = np.array(NP).reshape((-1,1))
    return NP


def adapt_CjJ4(J,j0,cjk,cjknum,zJc,zJcnum,a,b):
    C = []
    for j in range(j0, J):
        if j > j0:
            if j+1 == len(cjknum):
                break
            elif cjknum[j+1] == 0:
                break
        deltaj = (b-a) / (2 ** (j+2))
        k = cjk[j+1].reshape((1,-1))
        x = (2 * k + 1) * deltaj
        n = zJc.copy()
        rr = restrain(j+2,2 * k + 1,J,n)
        s = 0
        for j1 in range(j0, j+1):
            if cjknum[j1] == 0:
                continue
            k1 = cjk[j1].reshape((1,-1))
            aa = xphi0(j1+1, 2 * k1 + 1, a, b, x)
            aa = aa.T
            s = s + np.dot(C[j1-1].T, aa)
        C.append(rr - s) if isinstance(s, int) else C.append(rr - s.T)
    return C


def restrain(j1, k1, j2, k2):
    len1 = k1.shape[0]
    len2 = k2.shape[0]
    j1_j2 = j1 - j2
    t = 2 ** j1_j2
    if j1_j2 <= 0:
        rt = k1 / t
        rt = rt.reshape((-1,1))
        k2 = k2.reshape((1,-1))
        i = np.ones((1,len2))
        rt = rt * i
        i = np.ones((len1,1))
        k2 = i * k2
        value = ~np.array(abs(rt - k2), dtype=bool)
    return value


def cjkfun2(VJc, J, C, j0, gjk, gjknum, gJcnum, eps0):
    numJ = 2**j0 + 1
    zJc0 = np.arange(numJ) * 2**(J-j0)
    cjk = [np.array([0])]
    cjknum = []
    for j in range(j0, J):
        num = 0
        if gjknum[j] > 0:
            cjkvalue = np.dot(C[j-1], VJc)
            kt = np.flatnonzero((abs(cjkvalue) > eps0))
            num = kt.shape[0]
            if num > 0:
                kk = gjk[j][kt]
                cjk.append(kk)
                zJc0 = np.append(zJc0, (2 * kk + 1) * (2 ** (J - j - 1)), 0)
        cjknum.append(num)
    zJc = np.sort(zJc0).tolist()
    zJcnum = len(zJc)
    return cjk, cjknum, zJc, zJcnum


def IncreaseInterp(KnowX,KnowP,NX, J,j0,cjk,cjknum,zJc,zJcnum,a,b,C):
    _, NX_index, KnowX_index = np.intersect1d(NX, KnowX, return_indices=True)
    P = np.zeros_like(NX)
    P[NX_index] = KnowP[0][KnowX_index]
    diff = np.setdiff1d(NX, KnowX)
    index = np.where(np.in1d(NX, diff))
    i = np.arange(zJcnum)
    x = NX[index].reshape((1,-1))
    if x.any():
        value = Ii(i,J,j0,cjk,cjknum,zJc,x,a,b,C)
        s = np.dot(KnowP, value.T)
    else:
        s = []
    P[index] = s
    P = P.reshape((-1,1))
    return P


def Ii(i, J, j0, cjk, cjknum, zJc, x, a, b, C):
    k = np.arange(0, 2**j0+1).reshape(1,-1)
    s = xphi0(j0,k,a,b,x)
    rr = restrain(j0,k.reshape(-1),J,zJc)
    value = np.dot(s, rr)
    for j in range(j0, len(C)+1):
        if j > j0 and cjknum[j] == 0:
            break
        if cjknum[j] < 1:
            continue
        k = cjk[j].reshape(1,-1)
        s = xphi0(j+1,2*k+1,a,b,x)
        value = value + np.dot(s, C[j-1])
    return value

