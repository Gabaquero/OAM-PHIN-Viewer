import os
import math as mt
import numpy as np
import scipy.special as sp
import scipy.constants as con
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import cv2 as cv
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset


def cart2pol(p, d):
    th = np.arctan2(p, d)
    rh = np.sqrt(p ** 2 + d ** 2)
    return th, rh


def pol2cart(Rho, Theta):
    x_variable = Rho * np.cos(Theta)
    y_variable = Rho * np.sin(Theta)
    return x_variable, y_variable


def cart2sph(x, y, z):
    hypotxy = np.sqrt(x ** 2 + y ** 2)
    r = np.sqrt(hypotxy ** 2 + z ** 2)
    elev = np.arctan(z, hypotxy)
    az = np.arctan(y, x)
    return r, elev, az


# Parameters
P = 0  # Radial Index
A = 1  # Amplitude
f = 3.6E+9  # Frequency 1E9 (Hz) = 1 GHz
w0 = 1  # Waist of the beam
dist = 5  # meters

# Values
la = con.speed_of_light / f  # lambda
k = (((2 * np.pi) * 1.0003) / la)  # k value

# Alice/Bob Set-up
x_var = np.flipud(np.arange(-5, 5, 0.01))
y_var = np.flipud(np.arange(-5, 5, 0.01))
X, Y = np.flipud(np.meshgrid(np.flipud(x_var), np.flipud(y_var)))
theta_tx, rho_tx = cart2pol(X, Y)

# Eve Set-up
x_ex = np.flipud(np.arange(-10, 0, 0.01))
y_ex = np.flipud(np.arange(-10, 0, 0.01))
X_ex, Y_ex = np.flipud(np.meshgrid(np.flipud(x_ex), np.flipud(y_ex)))
theta_ex, rho_ex = cart2pol(X_ex, Y_ex)


# Field equation in function of z (distance of beam from z-axis at origin)
def lgoam(z, L, theta, rho):
    zR = (np.pi * (w0 ** 2)) / la
    W = w0 * (np.sqrt(1 + (z / zR)))
    t = ((2 * rho) ** 2) / (W ** 2)
    Z = ((np.sqrt((2 * mt.factorial(P)) / (np.pi * mt.factorial((P + abs(L)))))) * (1 / W) * (
            ((np.sqrt(2) * rho) / W) ** abs(L))) * (
                (np.exp(-((rho ** 2) / (W ** 2)))) * (sp.genlaguerre(P, abs(L))(t))) * (
            np.exp(((-1j * k * (rho ** 2) * z) / (2 * ((z ** 2) + (zR ** 2)))))) * \
        (np.exp(1j * (((abs(L) + (2 * P)) + 1) * np.arctan(z / zR)))) * (np.exp(-1j * L * theta))
    return Z


def getUCA(azimuth, radius, elevation, n1, mode1):
    arrRadius = 0.03
    elemNum = 16
    elemPhi = []
    for numRange in range(elemNum - 1):
        elemPhi.append((numRange * 2 * np.pi) / elemNum)
    E = (1 / (4.0 * np.pi * radius)) * np.exp(1j * k * radius) * np.exp(
        -1j * k * arrRadius * np.sin(elevation) * np.cos(azimuth - elemPhi[n1]) + 1j * mode1 * elemPhi[n1])
    return E


if __name__ == '__main__':
    # Plot of Phase
    plt.subplot(3, 3, 1)
    plt.set_cmap('jet')
    plt.title("Phase - Alice")
    plt1 = plt.imshow(np.flipud((np.angle(lgoam(dist, 4, theta_tx, rho_tx)))))
    plt.colorbar(plt1)
    # Plot of Intensity
    plt.subplot(3, 3, 2)
    plt.set_cmap('jet')
    plt.title("Intensity - Alice")
    plt2 = plt.imshow(np.flipud(abs(lgoam(dist, 1, theta_tx, rho_tx))))
    plt.colorbar(plt2)
    # Plot of Spatial
    plt.subplot(3, 3, 3)
    plt.set_cmap('jet')
    plt.title("Spatial - Alice")
    plt3 = plt.imshow(np.flipud(np.real((lgoam(dist, 1, theta_tx, rho_tx)))))
    plt.colorbar(plt3)

    # Eve Subplots
    plt.subplot(3, 3, 4)
    plt.set_cmap('jet')
    plt.title("Phase - Eve")
    plt1 = plt.imshow(np.flipud((np.angle(lgoam(dist, 1, theta_ex, rho_ex)))))
    plt.colorbar(plt1)
    # Plot of Intensity
    plt.subplot(3, 3, 5)
    plt.set_cmap('jet')
    plt.title("Intensity - Eve")
    plt2 = plt.imshow(np.flipud(abs(lgoam(dist, 1, theta_ex, rho_ex))))
    plt.colorbar(plt2)
    # Plot of Spatial
    plt.subplot(3, 3, 6)
    plt.set_cmap('jet')
    plt.title("Spatial - Eve")
    plt3 = plt.imshow(np.flipud(np.real((lgoam(dist, 1, theta_ex, rho_ex)))))
    plt.colorbar(plt3)

    # Superposition of states
    plt.subplot(3, 3, 7)
    plt.set_cmap('jet')
    plt.title("Super imposed mode")
    plt3 = plt.imshow(np.flipud(np.angle(lgoam(dist, 1, theta_tx, rho_tx) + lgoam(dist, -1, theta_tx, rho_tx))))
    plt.colorbar(plt3)

    plt.subplot(3, 3, 8)
    plt.set_cmap('jet')
    plt.title("Super imposed mode")
    plt3 = plt.imshow(np.flipud(abs(lgoam(dist, 1, theta_tx, rho_tx) + lgoam(dist, -1, theta_tx, rho_tx))))
    plt.colorbar(plt3)

    plt.subplot(3, 3, 9)
    plt.set_cmap('jet')
    plt.title("Super imposed mode")
    plt3 = plt.imshow(np.flipud(np.real(lgoam(dist, 1, theta_tx, rho_tx) + lgoam(dist, -1, theta_tx, rho_tx))))
    plt.colorbar(plt3)

# Purity calculator
    # mode - 1 < purity1 < mode + 1
    # purity1 != 0
    # pe1.append((abs(purity1-mode)/abs(mode))*100)
    # pe2.append((abs(purity2-mode)/abs(mode))*100)
    mode = 1
    arr1 = []
    arr2 = []
    arr3 = []
    arr4 = []
    for v in range(50):
        purity1 = ((np.angle(lgoam(v, mode, 315, 5))-np.angle(lgoam(v, mode, 45, 5))) / 180) * 1000
        purity2 = ((np.angle(lgoam(v, mode, 185, 4))-np.angle(lgoam(v, mode, 265, 4))) / 180) * 1000
        purity3 = ((np.angle(lgoam(v, mode, 315, 500)) - np.angle(lgoam(v, mode, 45, 500))) / 180) * 1000
        purity4 = ((np.angle(lgoam(v, mode, 315, 25)) - np.angle(lgoam(v, mode, 45, 25))) / 180) * 1000
        arr2.append(purity2)
        if mode - 1 < purity1 < mode + 1:
            arr1.append(purity1)
        # if mode - 1 < purity2 < mode + 1:
        #     arr2.append(purity2)
        # if mode - 1 < purity3 < mode + 1:
        #     arr3.append(purity3)
        # if mode - 1 < purity4 < mode + 1:
        #     arr4.append(purity4)

    fig, ax = plt.subplots()
    ax.plot(arr1, color='blue', label='Alice-Bob Mode: '+str(mode))
    ax.plot(arr2, color='red', label='Eve Mode: '+str(mode))
    # ax.plot(arr3, color='green')
    # ax.plot(arr4, color='orange')
    # ax.add_patch(Rectangle((arr1[1]-.5, 0), 1, pe1[1]))
    plt.ylabel("Mode")
    plt.xlabel("Propagation of Z")
    plt.legend()
    # axins = zoomed_inset_axes(ax, 5, loc=1)  # zoom = 2
    # axins.plot(arr2[1], pe2[2], color='blue')
    # axins.add_patch(Rectangle((arr2[1]-.5, 0), 1, pe2[1], edgecolor='red', facecolor='red',))
    # axins.plot(arr1[1], pe1[1], color='blue')
    # axins.add_patch(Rectangle((arr1[1]-.5, 0), 1, pe1[1]))
    # xz1 = mode-1
    # xz2 = mode+1
    # yz1 = -1
    # yz2 = 20
    # axins.set_xlim(xz1, xz2)
    # axins.set_ylim(yz1, yz2)
    # plt.xticks(visible=True)
    # plt.yticks(visible=True)
    # mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
    plt.draw()


# # Histogram Comparison
# base = cv.imread('4.png')
# comp = cv.imread('img1.png')
#
# hsv_base = cv.cvtColor(base, cv.COLOR_BGR2HSV)
# hsv_test = cv.cvtColor(comp, cv.COLOR_BGR2HSV)
#
# h_bins = 50
# s_bins = 60
# histSize = [h_bins, s_bins]
# h_ranges = [0, 180]
# s_ranges = [0, 256]
# ranges = h_ranges + s_ranges
# channels = [0, 1]
#
# hist_base = cv.calcHist([hsv_base], channels, None, histSize, ranges, accumulate=False)
# cv.normalize(hist_base, hist_base, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)
# hist_test = cv.calcHist([hsv_test], channels, None, histSize, ranges, accumulate=False)
# cv.normalize(hist_test, hist_test, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)
#
# compare_method = cv.HISTCMP_CORREL
#
# base_base = cv.compareHist(hist_base, hist_base, compare_method)
# base_test = cv.compareHist(hist_base, hist_test, compare_method)
#
# print('base_base Similarity = ', base_base)
# print('base_test Similarity = ', base_test)
#
# cv.imshow('base', base)
# cv.imshow('test1', comp)
# cv.waitKey(0)


# # Oam mode AI image classifier (completely image based) DOESN'T WORK
# # Training
# model_trainer = ClassificationModelTrainer()
# model_trainer.setModelTypeAsResNet50()
# model_trainer.setDataDirectory("basedata")
# model_trainer.trainModel(num_objects=5, num_experiments=100,
#                          enhance_data=True, batch_size=32, show_network_summary=True)
# # Classifier
# execution_path = os.getcwd()
# print("The Current working directory is :", execution_path)
# prediction = CustomImageClassification()
# prediction.setModelTypeAsResNet50()
# prediction.setModelPath(os.path.join(execution_path, "model_ex-006_acc-0.939373.h5"))
# prediction.setJsonPath(os.path.join(execution_path, "model_class.json"))
# prediction.loadModel(num_objects=5)
#
# predictions, probabilities = prediction.classifyImage(os.path.join(execution_path, "Image.jpg"), result_count=5)
#
# for eachPrediction, eachProbability in zip(predictions, probabilities):
#     print(eachPrediction + " : " + eachProbability)
#

# Animate function (the simple way, make sure the outpath is set to a folder and then use any gif-making app to animate)

def animateIt(n):
    outpath = "yourpath/path"
    for i in range(0, n, 1):
        plt.set_cmap('jet')
        plt.imshow(np.flipud(np.angle(lgoam(i, 1, theta_ex, rho_ex))))
        plt.savefig(os.path.join(outpath, "{}.png".format(i)))
        plt.cla()
    return "done"


plt.show()
