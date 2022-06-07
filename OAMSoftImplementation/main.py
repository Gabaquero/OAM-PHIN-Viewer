from mpl_toolkits import mplot3d
from matplotlib.animation import FuncAnimation
import numpy as np
import math as mt
import scipy.special as sp
import scipy.constants as con
import matplotlib.pyplot as plt


def cart2pol(p, d):
    th = np.arctan2(p, d)
    rh = np.sqrt(p ** 2 + d ** 2)
    return th, rh


# Parameters
L = 2  # Azimuthal Angle
P = 0  # Radial Index
A = 1  # Amplitude
z = 10  # Beam Radius at Distance Z
f = 3.6E+9  # Frequency 1E9 (Hz) = 1 GHz

# Functions
la = con.speed_of_light / f  # lambda
w0 = 1
k = (((2 * np.pi) * 1.0003) / la)
zR = (np.pi * (w0 ** 2)) / la
W = w0 * (np.sqrt(1 + (z / zR)))

# Equation
x = np.arange(-5, 5, 0.01)
y = np.arange(-5, 5, 0.01)
X, Y = np.meshgrid(x, y)
theta, rho = cart2pol(X, Y)
t = ((2 * rho) ** 2) / (W ** 2)
Psi = ((abs(L) + (2 * P)) + 1) * np.arctan(z / zR)
T1 = (np.sqrt((2 * mt.factorial(P)) / (np.pi * mt.factorial((P + abs(L)))))) * (1 / W) * (((np.sqrt(2) * rho) / W) ** abs(L))
T2 = (np.exp(-((rho ** 2) / (W ** 2)))) * (sp.genlaguerre(P, abs(L))(t))
T3 = np.exp(((-1j * k * (rho ** 2) * z) / (2 * ((z ** 2) + (zR ** 2)))))
T4 = np.exp(1j * Psi)
T5 = np.exp(-1j * L * theta)
Z = A * T1 * T2 * T3 * T4 * T5


def looper(rng):
    for i in range(rng):
        t1 = ((2 * rho) ** 2) / (W ** 2)
        Psi1 = ((abs(L) + (2 * P)) + 1) * np.arctan(i / zR)
        T11 = (np.sqrt((2 * mt.factorial(P)) / (np.pi * mt.factorial((P + abs(L)))))) * (1 / W) * (((np.sqrt(2) * rho) / W) ** abs(L))
        T22 = (np.exp(-((rho ** 2) / (W ** 2)))) * (sp.genlaguerre(P, abs(L))(t1))
        T33 = np.exp(((-1j * k * (rho ** 2) * i) / (2 * ((z ** 2) + (zR ** 2)))))
        T44 = np.exp(1j * Psi1)
        T55 = np.exp(-1j * L * theta)
        Z1 = A * T11 * T22 * T33 * T44 * T55
    return Z1


if __name__ == '__main__':
    # Plot of Phase
    plt.subplot(1, 2, 1)
    plt.set_cmap('jet')
    plt1 = plt.imshow(np.flipud(np.angle(Z)))
    plt.colorbar(plt1)
    # Plot of Intensity
    plt.subplot(1, 2, 2)
    plt.set_cmap('jet')
    plt.imshow(abs(Z))

    anim = FuncAnimation(fig, animate, init_func=init, frames=200, interval=20, blit=True)

    plt.show()
