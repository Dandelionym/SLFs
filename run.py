# -*- coding: utf-8 -*-
"""

    ** NOTE: This code is dropped on Dec 31, 2022. A new version should be attached in the future. **

    INFORMATION TITLE
    -----------------------------------------------------------------
    AUTHOR: Mellen Y.Pu
    DATE: 2022/12/21 下午4:07
    FILE: run.py
    PROJ: SLFs：
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

np.random.seed(1)


def SLF_Base(num_seeds):
    def f(t):
        """
            F = -u * cos((w_0/v)*t + ⨏) + λ
        :param t: x-x_0
        :return: F value
        """
        # U is Activation Constant, depends on the certain solution env.
        u = - (1 / 16)
        # w is competition speed.
        w = 16
        # v is the activation speed.
        v = 4
        # x is the distance between current point and target point.
        x = t
        # y is ligand frequency, depends on certain ligand.
        y = - 4 * np.pi / 8
        # l is system energy, for stableness.
        l = 1 / np.exp(t)
        return u * np.cos((w/v) * x + y) + l



    X = np.arange(-5, 5, 0.05)
    Y = np.arange(-5, 5, 0.05)
    X, Y = np.meshgrid(X, Y)
    R = np.sqrt(X**2 + Y**2)
    Z = f(R)
    a, b = Z.shape
    P = 0.45


    for i in tqdm(range(num_seeds)):
        P = P + 0.0000001
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
    plt.title(f"Local surface after modeling {num_seeds} states\n")
    plt.colorbar()
    plt.tight_layout()

    # fig = plt.figure(facecolor='white')
    # ax = fig.add_subplot(1, 1, 1, projection='3d')
    # ax.plot_surface(X, Y, Z / 8, rstride=1, cstride=1, linewidth=0, antialiased=False)
    #
    # ax.grid(False)
    # ax.spines['top'].set_visible(False)
    # ax.spines['left'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    plt.show()


def SLF_y(num_seeds):

    V = 1 / np.exp(10)

    # Less than P will lead to the growth that outside the local area.
    P = 0.5

    def f(t):
        """
            F = -u * cos((w_0/v)*t + ⨏) + λ
        :param t: x-x_0
        :return: F value
        """
        # U is Activation Constant, depends on the certain solution env.    >>> (the more ligands, the lager mountains) <<< u ⥷ (0, 1)
        u = - (1 / 1)
        # w is competition distance.                                        >>> (the shortest distance of competition) <<<  x ⥷ (0, 1)
        w = 1 / np.abs(np.exp(0.1) - 1)
        # x is the distance between current point and target point.
        x = t
        # y is ligand frequency, depends on certain area (centeral or edged).
        y = - 1 * np.pi / 64
        # l is system energy, for stableness.
        l = None
        return u * np.cos(w * x + y)


    X = np.arange(-5, 5, 0.02)
    Y = np.arange(-5, 5, 0.02)
    X, Y = np.meshgrid(X, Y)
    R = np.sqrt(X**2 + Y**2)
    Z = f(R)
    a, b = Z.shape


    for _ in tqdm(range(num_seeds)):
        P = P + V
        if np.abs(np.random.randn()) < P:
            # P % makes it develop in the area that is competitive enough.
            m, n = np.where(Z == np.max(Z))
            # m, n = np.argmax(Z[np.random.choice(list(range(a)))]), np.argmax(Z[np.random.choice(list(range(a)))])
        else:
            # (1-P) % makes it develop in other neightive points.
            m, n = np.random.choice(list(range(a))), np.random.choice(list(range(a)))

        if P < 1:
            p, q = m + np.abs(np.random.randn())*10 * np.random.choice([-1, 1]), n + np.abs(np.random.randn())*10 * np.random.choice([-1, 1])
        else:
            p = q = 0
            break
        Z = Z + f(np.sqrt((X+p)**2 + (Y+q)**2))

    plt.matshow(Z, cmap='Greys')
    plt.title(f"Local surface after modeling {num_seeds} states\n")
    plt.colorbar()
    plt.tight_layout()

    # fig = plt.figure(facecolor='white')
    # ax = fig.add_subplot(1, 1, 1, projection='3d')
    # ax.plot_surface(X, Y, Z / 8, rstride=1, cstride=1, linewidth=0, antialiased=False)
    #
    # ax.grid(False)
    # ax.spines['top'].set_visible(False)
    # ax.spines['left'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)

    plt.show()


if __name__ == '__main__':
    for i in [100000]:
        SLF_y(i)
