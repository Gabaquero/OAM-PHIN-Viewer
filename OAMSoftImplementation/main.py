from mpl_toolkits.mplot3d.axes3d import Axes3D, get_test_data
import numpy as np
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
z = 0.5  # Beam Radius at Distance Z
f = 3.6E+9  # Frequency 1E9 (Hz) = 1 GHz
time = 0  # Selected Time

# Functions
la = con.speed_of_light / f  # lambda
w0 = 10
k = (((2 * np.pi) * 1.0003) / la)
zR = (np.pi * (w0 ** 2)) / la
W = w0 * (np.sqrt(1 + (z / zR)))

# Equation
x = np.arange(-15, 15, 0.01)
y = np.arange(-15, 15, 0.01)
X, Y = np.meshgrid(x, y)
theta, rho = cart2pol(X, Y)
t = ((2 * rho) ** 2) / (W ** 2)
Psi = ((abs(L) + (2 * P)) + 1) * np.arctan(z / zR)
T1 = (w0 / W) * (((np.sqrt(2) * rho) / W) ** abs(L))
T2 = (np.exp(-((rho ** 2) / (W ** 2)))) * (sp.genlaguerre(P, abs(L))(t))
T3 = np.exp(((-1j * k * (rho ** 2)) / (2 * zR)))
T4 = np.exp(1j * Psi)
T5 = np.exp(-1j * L * theta)
T6 = np.exp(-1j * time)
Z = A * T1 * T2 * T3 * T4 * T5 * T6


if __name__ == '__main__':

    # Plot of Phase
    plt.subplot(1, 2, 1)
    plt.set_cmap('jet')
    plt.imshow(np.angle(Z))
    # Plot of Intensity
    plt.subplot(1, 2, 2)
    plt.set_cmap('jet')
    plt.imshow(np.real(Z))

    plt.show()
