import torch
from matplotlib import pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import math
import numpy as np

def get_phi(x):
    out = torch.zeros_like(x)
    out[(x >= -1) & (x < 0)] = x[(x >= -1) & (x < 0)] + 1
    out[(x >= 0) & (x <= 1)] = 1 - x[(x >= 0) & (x <= 1)]
    return out

def get_psi(j,k,x):
    return get_phi((2 ** (j+1)) * x - (2*k+1))

x = torch.arange(-1,1,1/10000)


class Iwt_module():
    def __init__(self, x, j0):
        self.idxs = [[0, len(x)/2, len(x)]]

    
    def get_Iwt_l(self, x, j, f):
        Ijf = 0
        for k in range(-2 ** j, 2 ** j + 1, 1):
            x_jk = torch.tensor(k * (2 ** (-j)))
            f_xjk = f(x_jk)
            phi_jk = get_phi((2 ** j) * x - k)
            Ijf += f_xjk * phi_jk
        return Ijf

    def get_alpha_jk(self, x, j, k, f, u, th):
        y_jk = torch.tensor((2*k+1) * (2 ** (-j-1)))
        f_yjk = f(y_jk)
        diff = torch.abs(x - y_jk)
        idx = torch.argmin(diff)
        mask = torch.zeros_like(x)
        mask[idx] = 1
        mask = mask.type(torch.int)
        indices = torch.nonzero(mask).reshape(-1)
        diff2 = torch.abs(f_yjk - u[indices])
        if diff2 > th:
            return f_yjk - u[indices], indices
        else:
            return 0, 0

    def get_Iwt(self, x, j0, J, f, e):
        if J == 0:
            return self.get_Iwt_l(x, J, f)
        j = J-1
        Iwt_r = 0
        if j == j0:
            indicies = []
            u0 = self.get_Iwt_l(x, j0, f)
            for k in range(-2 ** j, 2 ** j, 1):
                alpha_jk, i = self.get_alpha_jk(x, j, k, f, u0, th=e * math.sqrt(2 **(-j)))
                if i > 0:
                    indicies.append(i.item())
                psi_jk = get_phi((2 ** (j+1)) * x - (2*k+1))
                Iwt_r += alpha_jk * psi_jk
            self.idxs.append(indicies)
            return u0 + Iwt_r
        elif j > j0:
            indicies = []
            u_j1 = self.get_Iwt(x, j0, j, f, e)
            for k in range(-2 ** j, 2 ** j, 1):
                alpha_jk, i = self.get_alpha_jk(x, j, k, f, u_j1, th=e * math.sqrt(2 **(-j)))
                if i > 0:
                    indicies.append(i.item())
                psi_jk = get_psi(j, k, x)
                Iwt_r += alpha_jk * psi_jk
            self.idxs.append(indicies)
            return u_j1 + Iwt_r
        else:
            return self.get_Iwt_l(j0, f)

def func(x):
    return torch.sin(400 * x**3) * torch.exp(-30 * x**2)

j0 = 0
J = 6
f = func

Iwt_m1 = Iwt_module(x, j0)
Iwt_m2 = Iwt_module(x, j0)
Iwt = Iwt_m1.get_Iwt(x, j0, J, f, 0)
Iwt2 = Iwt_m2.get_Iwt(x, j0, J, f, 0.1)

plt.plot(x.numpy(), Iwt.numpy(), label = "threshold=0")
plt.plot(x.numpy(), Iwt2.numpy(), label = "threshold=0.5")
plt.plot(x.numpy(), func(x).numpy(), label="func")
plt.legend()

fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
for i, idx_e in enumerate(Iwt_m1.idxs):
    idx_e = (torch.tensor(idx_e) - (len(x)/2.)) / (len(x)/2.)
    axs[0].scatter(idx_e.numpy(), np.ones_like(idx_e) * i, s=12)
for i, idx_e in enumerate(Iwt_m2.idxs):
    idx_e = (torch.tensor(idx_e) - (len(x)/2.)) / (len(x)/2.)
    axs[1].scatter(idx_e.numpy(), np.ones_like(idx_e) * i, s=12)

axs[0].set_title('Full interp')
axs[1].set_title('self-adaptive interp')

plt.show(block=True)