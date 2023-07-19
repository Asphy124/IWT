import numpy as np
from matplotlib import pyplot as plt
from utils.aiwt import cjkfun, cjkfun2, z_g, CompressInterp, IncreaseInterp, adapt_CjJ4
from utils.pde import Precise, theta1_2d
from utils.tools import Draw, DrawCollocationPoint

J = 15
j0 = 0
a = 0
b = 2
Re = 1000
deltax = (b-a)/(2**J)
XJ0 = np.arange(a, b+deltax, deltax).reshape(1,-1)
V0 = np.sin(np.pi * XJ0)
# V0 = np.random.random(size=XJ0.shape)
M = 1
CC = 1
t = 0.4
tao = 0.001
eps0 = 0.005
same = np.inf
nn_dict = {1:[1,'0.01'], 400:[3,'0.4'], 600:[5,'0.6'], 800:[1,'0.8'],
           1200:[3,'1.2'], 600:[5,'1.6'], 2000:[1,'2.0'], 2400:[3,'2.4'],
           2600:[5,'2.6'], 2800:[1,'2.8'], 3000:[3,'3.0'], 3200:[5,'3.2']}

fig, axs=plt.subplots(1,2,constrained_layout=True)

for nn in range(1, round(t / tao) + 1):
    if nn == 1:
        cjk, cjknum, zJc, zJcnum = cjkfun(V0, J, j0, a, b, eps0)
    elif nn > 1:
        gjk0 = gjk.copy()
        gjknum0 = gjknum.copy()
        gJc0 = gJc.copy()
        gJcnum0 = gJcnum
        XJ0 = XJ.copy()
        V0 = V1.copy().T
        cjk, cjknum, zJc, zJcnum = cjkfun2(V1, J, C, j0, gjk, gjknum, gJcnum, eps0)
        same = 0
        eq = True
        row1, col1 = len(cjk), len(max(cjk, key=len))
        row0, col0 = len(cjk0), len(max(cjk0, key=len))
        if row1 == row0 and col0 == col1:
            for i in range(row1):
                if not np.array_equal(cjk[i], cjk0[i]):
                    eq = False
                    break
            if eq:
                same = 1
    cjk0 = cjk.copy()
    if nn == 1 or same == 0:
        gjk, gjknum, gJc, gJcnum = z_g(j0, J, cjk, cjknum, zJc, zJcnum, a, b, M, CC)
        # gjk, gjknum, gJc, gJcnum = cjk, cjknum, np.array(zJc), zJcnum
        XJ = a + deltax * gJc
        if nn == 1:
            V0 = CompressInterp(XJ0, V0, XJ)
        else:
            V0 = IncreaseInterp(XJ0, V0, XJ, J, j0, gjk0, gjknum0, 
                                gJc0, gJcnum0, a, b, C)
        C = adapt_CjJ4(J, j0, gjk, gjknum, gJc, gJcnum, a, b)
        theta1d, theta2d = theta1_2d(J, j0, a, b, C, gjk, gjknum, gJc, gJcnum)  # 计算算子的导数
    else:
        V0 = V1
    V1 = Precise(V0, Re, theta1d,theta2d,gJcnum,tao)  # 精细积分法

    Draw(axs[0], XJ, V1, cjk, cjknum, zJc, a, b, j0, J)
    DrawCollocationPoint(axs[1], cjk, cjknum, a, b, j0,J)
    axs[0].set_title("nn={}".format(nn))
    axs[1].set_title("pionts={}".format(zJcnum))
    plt.pause(0.05)
    if nn == round(t / tao):
        plt.show(block=True)
    axs[0].clear()
    axs[1].clear()
