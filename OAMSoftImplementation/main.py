import matplotlib.animation as anm
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
f = 3.6E+9  # Frequency 1E9 (Hz) = 1 GHz
w0 = 1  # Waist of the beam

# Mesh-grid setup
x = np.arange(-5, 5, 0.01)
y = np.arange(-5, 5, 0.01)
X, Y = np.meshgrid(x, y)
theta, rho = cart2pol(X, Y)


# Field equation in function of z (distance from z)
def field(z):
    la = con.speed_of_light / f  # lambda
    k = (((2 * np.pi) * 1.0003) / la)
    zR = (np.pi * (w0 ** 2)) / la
    W = w0 * (np.sqrt(1 + (z / zR)))
    t = ((2 * rho) ** 2) / (W ** 2)
    z1 = ((np.sqrt((2 * mt.factorial(P)) / (np.pi * mt.factorial((P + abs(L)))))) * (1 / W) * (
            ((np.sqrt(2) * rho) / W) ** abs(L))) * (
                 (np.exp(-((rho ** 2) / (W ** 2)))) * (sp.genlaguerre(P, abs(L))(t))) * (
             np.exp(((-1j * k * (rho ** 2) * z) / (2 * ((z ** 2) + (zR ** 2)))))) * \
         (np.exp(1j * (((abs(L) + (2 * P)) + 1) * np.arctan(z / zR)))) * (np.exp(-1j * L * theta))
    return z1


if __name__ == '__main__':
    # Plot of Phase
    plt.subplot(1, 2, 1)
    plt.set_cmap('jet')
    plt1 = plt.imshow(np.flipud(np.angle(field(0))))
    plt.colorbar(plt1)
    # Plot of Intensity
    plt.subplot(1, 2, 2)
    plt.set_cmap('jet')
    plt.imshow(field(0))

    # First set up the figure, the axis, and the plot element we want to animate
    fig = plt.figure()
    ax = plt.axes(xlim=(0, 10), ylim=(0, 10))
    im = plt.imshow(np.flipud(np.angle(field(0))), interpolation='none')


    def init():

        return [im]


    def animate(s):

        im.set_array(a)
        return [im]


    anim = anm.FuncAnimation(fig, animate, init_func=init, frames=200, interval=20, blit=True)

    plt.show()
