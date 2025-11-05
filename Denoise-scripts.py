import numpy as np
import matplotlib.pyplot as plt
import os
import george
from george import kernels
import scipy
import scipy.interpolate
from scipy.optimize import minimize
from extinction import ccm89, remove, apply
import warnings
warnings.filterwarnings('ignore')
import dateutil.parser
import astropy.time
import george
from george import kernels
from tqdm.notebook import tqdm
import pandas as pd
from astropy.io import fits

def split_data(x, y, split_wl):

    split_index = np.argmin(abs(x - split_wl))
    #split_index2 = np.argmin(abs(x - (split_wl + 500)))

    x_1 = x[:split_index]
    y_1 = y[:split_index]
    x_2 = x[split_index:]
    y_2 = y[split_index:]

    cond_nan_1 = np.isnan(y_1)
    cond_nan_2 = np.isnan(y_2)

    x_1 = x_1[~cond_nan_1]
    y_1 = y_1[~cond_nan_1]
    x_2 = x_2[~cond_nan_2]
    y_2 = y_2[~cond_nan_2]
    
    return x_1, y_1, x_2, y_2, split_index


def gaussian_regression(x, y, kernels):
    gp = george.GP(kernels[0] + kernels[1], fit_white_noise = False, fit_kernel = False, white_noise = log(0.001**2))

    gp.compute(x, np.zeros(x.shape))

    pred_1 = gp.predict(y, x, return_cov=False, kernel = kernels[0])
    pred_2 = gp.predict(y, x, return_cov=False, kernel = kernels[1])
    
    return pred_1, pred_2


def get_gp_filters(x, y, split_wl, shortterm_c = False):
    
    #Instead of splitting the data and fitting it like that, let's fit the whole spectrum with both
    #x_1, y_1, x_2, y_2, s_ind = split_data(x, y, split_wl)
    
    #if len(x_2) > 20:
        # red side
        
    cond = ~np.isnan(y)
        
    kernel1_r = 100 * kernels.ExpSine2Kernel(100, 0.1) * kernels.ExpSquaredKernel([2000.0], ndim=1, axes=0)
    if not shortterm_c:
        kernel2_r = 100 * kernels.Matern32Kernel(100000) + 20 * kernels.Matern32Kernel(10000)
    else:
        kernel2_r = 100 * kernels.Matern32Kernel(100000) + 20 * kernels.Matern32Kernel(5000)
        
    noise_red, spectrum_red = gaussian_regression(x[cond], y[cond], [kernel1_r, kernel2_r])
        
    kernel1_b = 10 * kernels.ExpSine2Kernel(100, 0.1) * kernels.ExpSquaredKernel([500.0], ndim=1, axes=0)
    kernel2_b = 100 * kernels.Matern32Kernel(100000) + 150 * kernels.Matern32Kernel(10000)

    noise_blue, spectrum_blue = gaussian_regression(x[cond], y[cond], [kernel1_b, kernel2_b])
    
    # Stitch them together
    
    if split_wl > max(x):
        noise_final = noise_blue
        spectrum_final = spectrum_blue
    else:
        split_index = np.argmin(abs(x[cond] - split_wl))
        noise_final = np.append(noise_blue[:split_index], noise_red[split_index:])
        spectrum_final = np.append(spectrum_blue[:split_index], spectrum_red[split_index:])
    

    return noise_final, spectrum_final, x[cond] #np.append(x_1[:s_ind], x_2)


def get_gp_filters_scale_stuff(x, y, split_wl, std_vals, shortterm_c = False):
    
    #x_1, y_1, x_2, y_2, s_ind = split_data(x, y, split_wl)
    
    #if len(x_2) > 20:
        
    cond = ~np.isnan(y)
        # red side
    kernel1_r = g_scaler(10, 100, std_vals[1], std_min, std_max) * kernels.ExpSine2Kernel(100, 0.1)\
                    * kernels.ExpSquaredKernel([2000.0], ndim=1, axes=0)
    if not shortterm_c:
        kernel2_r = 100 * kernels.Matern32Kernel(100000) \
                + g_scaler(100, 20, std_vals[0], std_max, std_min) * kernels.Matern32Kernel(10000)
    else:
        kernel2_r = 100 * kernels.Matern32Kernel(100000) \
                + g_scaler(250, 150, std_vals[0], std_max, std_min) * kernels.Matern32Kernel(5000)

    noise_red, spectrum_red = gaussian_regression(x[cond], y[cond], [kernel1_r, kernel2_r])
    
    
    #else:
    #    noise_red = []
    #    spectrum_red = []
    
    # blue side
    kernel1_b = g_scaler(10, 30, std_vals[0], std_min, std_max) * kernels.ExpSine2Kernel(100, 0.1)\
                * kernels.ExpSquaredKernel([500.0], ndim=1, axes=0)
    kernel2_b = 100 * kernels.Matern32Kernel(100000) + g_scaler(250, 150, std_vals[0], std_max, std_min)\
                * kernels.Matern32Kernel(5000)

    noise_blue, spectrum_blue = gaussian_regression(x[cond], y[cond], [kernel1_b, kernel2_b])
    
    # Stitch them together

    if split_wl > max(x):
        noise_final = noise_blue
        spectrum_final = spectrum_blue
    else:
        split_index = np.argmin(abs(x[cond] - split_wl))
        noise_final = np.append(noise_blue[:split_index], noise_red[split_index:])
        spectrum_final = np.append(spectrum_blue[:split_index], spectrum_red[split_index:])

    #noise_final = np.append(noise_blue[:s_ind], noise_red)
    #spectrum_final = np.append(spectrum_blue[:s_ind], spectrum_red)
    
    return noise_final, spectrum_final, x[cond] #np.append(x_1[:s_ind], x_2)


def g_scaler(NewMin, NewMax, OldMin, OldMax, OldValue):
    NewValue = (((OldValue - OldMin) * (NewMax - NewMin)) / (OldMax - OldMin)) + NewMin
    
    if NewValue > NewMax:
        return NewMax
    elif NewValue < NewMin:
        return NewMin
    else:
        return NewValue
