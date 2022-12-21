# -*- coding: utf-8 -*-
"""
    INFORMATION TITLE
    -----------------------------------------------------------------
    AUTHOR: Mellen Y.Pu
    DATE: 2022/12/21 下午4:07
    FILE: run.py
    PROJ: SLFs
    IDE: PyCharm
    EMAIL: yingmingpu@gmail.com
    ----------------------------------------------------------------- 
                                      ONE DOOR OPENS ALL THE WINDOWS.

    @INTRODUCTION: 
     - The mainland of SLF.
    @FUNCTIONAL EXPLATION:
     - launch home.
    @LAUNCH:
     - $ python run.py
"""


import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def f(t):
    t = 2*t - 4 * np.pi / 8
    return (1 / 16) * np.cos(16*t) + (1 / np.exp(t))

fig = plt.figure(facecolor='white')
ax = fig.add_subplot(1, 1, 1, projection='3d')

X = np.arange(-5, 5, 0.01)
Y = np.arange(-5, 5, 0.01)
X, Y = np.meshgrid(X, Y)
R = np.sqrt(X**2 + Y**2)
Z = f(R)

a, b = Z.shape
P = 0.45

for i in tqdm(range(1000)):
    P = P + 0.0001
    if np.random.randn() < P:
        m, n = np.argmax(Z[np.random.choice(list(range(a)))]), np.argmax(Z[np.random.choice(list(range(a)))])
    else:
        m, n = np.argmin(Z[np.random.choice(list(range(a)))]), np.argmin(Z[np.random.choice(list(range(a)))])
    if P < 1:
        p, q = m + np.abs(np.random.randn())*10 * np.random.choice([-1, 1]), n + np.abs(np.random.randn())*10 * np.random.choice([-1, 1])
    else:
        p = q = 0
    if P > 1:
        break
    Z = Z + f(np.sqrt((X+p)**2 + (Y+q)**2))

plt.matshow(Z, cmap='Greys')


# ax.plot_surface(X, Y, Z / 8, rstride=1, cstride=1, linewidth=0, antialiased=False)
#
# ax.grid(False)
# ax.spines['top'].set_visible(False)
# ax.spines['left'].set_visible(False)
# ax.spines['right'].set_visible(False)
# ax.spines['bottom'].set_visible(False)
plt.show()