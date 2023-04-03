import cv2
import numpy as np
from scipy import ndimage


def harris_corner_detector(img,k=0.08,alpha=0.23,sigma=1,binarized = True):
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    dx = cv2.Sobel(img_gray, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(img_gray, cv2.CV_32F, 0, 1, ksize=3)

    dx_2 = dx**2
    dy_2 = dy**2
    dxy = dx*dy

    sx_2 = ndimage.gaussian_filter(dx_2,sigma,mode='constant', cval=0.0)
    sy_2 = ndimage.gaussian_filter(dy_2,sigma,mode='constant', cval=0.0)
    sxy = ndimage.gaussian_filter(dxy,sigma,mode='constant', cval=0.0)

    det_M = sx_2*sy_2 - sxy**2
    trace_M = sx_2 + sy_2

    R = det_M - k*trace_M**2
    R = (R - np.min(R)) / (np.max(R) - np.min(R))
    if binarized:
        R[R > alpha] = 1
        R[R <= alpha] = 0
    return R


def hough_lines(image): 
    Ny = image.shape[0]
    Nx = image.shape[1]
    max_dist = int(np.round(np.sqrt(Nx**2 + Ny ** 2)))
    thetas = np.deg2rad(np.arange(-90, 90))
    rs = np.linspace(-max_dist, max_dist, 2*max_dist)
    accumulator = np.zeros((2 * max_dist, len(thetas)))
    for y in range(Ny):
        for x in range(Nx):
            if image[y,x] > 0:
                for k in range(len(thetas)):
                    r = x*np.cos(thetas[k]) + y * np.sin(thetas[k])
                    accumulator[int(r) + max_dist,k] += 1
    
    return accumulator, thetas, rs, max_dist

def find_best(accumulator,thetas,max_dist,n=1):
    ret = []
    vals = np.unravel_index(accumulator.ravel().argsort()[-10:][::-1], accumulator.shape)
    for i in range(n):
        ret.append((vals[0][i]-max_dist,thetas[vals[1][i]]))
    return ret
