import numpy as np
from utils.aiwt import CompressInterp
from scipy.interpolate import pchip_interpolate

def Draw(ax, X, V, gjk, gjknum, gJc, xmin, xmax, j0, J):
    xmin = min(X)
    xmax = max(X)
    deltax = 1.e-4
    x = np.arange(xmin, xmax+deltax, deltax)
    v = pchip_interpolate(X.reshape(-1), V.reshape(-1), x, der=0, axis=0)
    deltaxJ = (xmax - xmin) / (2 ** J)
    XJ = xmin + np.array(gJc) * deltaxJ
    VJ = CompressInterp(X.reshape((1,-1)),V.reshape((1,-1)),XJ)

    ax.plot(x,v,'-k')
    ax.scatter(XJ, VJ.reshape(-1), s=10)
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$u(x)$')


def DrawCollocationPoint(ax, cjk,cjknum,a,b,j0,J):
    for j in range(j0, J):
        deltax = (b - a) / 2 ** (j+1)
        k = np.arange(cjknum[j])
        if(k.shape[0]==0):
            continue
        Y1 = np.ones(k.shape[0]) + j
        X1 = (0.5 + cjk[j]) * deltax
        ax.scatter(X1, Y1, s=8, c='k')
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'Level $j$')
    ax.set_ylim([0, J])
    