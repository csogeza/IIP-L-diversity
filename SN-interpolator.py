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

def resample_spec(orig_wl, orig_fl, new_wl):
	new_fl = scipy.interpolate.interp1d(orig_wl, orig_fl)(new_wl)
	return new_fl

def LOWESS_smoothing(x, y, tau=0.1):
    ''' Locally Weighted Linear Regression from Andrew Ng
        Weighting nearby points by gaussian-like curve,
        and fitting a line locally to smooth the data.
        tau - bandwidth of the kernel '''
    
    x, y = np.array(x), np.array(y)
    y_lowess_smoothed = y.copy()
    for i in range(len(x)):
        kernel = np.exp(-(x - x[i])**2 / (2 * tau**2))
        where = np.where(kernel > 0.01) # only count the points with at least 1% weight to reduce runtime
        p = np.polyfit(x[where], y[where], w=kernel[where], deg=1)
        y_lowess_smoothed[i] = p[0] * x[i] + p[1]
        
    return y_lowess_smoothed

Halp_mask = [[6300, 6800]]

class SN:
    def __init__(self, spectra, times, texp):
        self.spectra = spectra
        self.times = times
        self.residuals = None
        self.texp = texp
    
    def plot_residual_ts(self, plot_mode = 'texp', just_resid = False):
        if just_resid is False:
            plt.figure(figsize=(14,14))

        evenly_spaced_interval = np.linspace(0.0, 0.9, len(self.times))
        cll = [cm.jet(x) for x in evenly_spaced_interval]

        residuals = []

        offset = 1.5
        for i in range (len(self.times)):

            #MASKING
            wl_grid = self.spectra[i][:,0]
            f_mask = np.ones_like(wl_grid, dtype='bool')
            for band in Halp_mask:
                band_mask = np.logical_and(wl_grid > band[0],
                                           wl_grid < band[1])
                inverse_band_mask = np.logical_not(band_mask)
                f_mask = np.logical_and(inverse_band_mask, f_mask)

        
            # This is the "good case"
            ind = np.argmin(abs(self.spectra[i][:,0] - 6000))

            low_smooth = LOWESS_smoothing(self.spectra[i][f_mask,0], self.spectra[i][f_mask,1]/self.spectra[i][ind,1], 500)
            new_filt = resample_spec(self.spectra[i][f_mask,0], low_smooth, self.spectra[i][:,0])
            
            if just_resid is False:
                if plot_mode == 'texp':
                    plt.plot(self.spectra[i][:,0], self.spectra[i][:,1]/self.spectra[i][ind,1] / new_filt - 1 - self.times[i],
                         color = cll[i], label = self.times[i], zorder = 20)
                    plt.plot([3500, 10500], [-self.times[i],  -self.times[i]], color = 'red', ls = '--', alpha = 0.3)
                else:
                    plt.plot(self.spectra[i][:,0], self.spectra[i][:,1]/self.spectra[i][ind,1] / new_filt - 1 - (i-1) * offset,
                         color = cll[i], label = self.times[i], zorder = 20)
                    plt.plot([3500, 10500], [- (i-1) * offset,  - (i-1) * offset], color = 'red', ls = '--', alpha = 0.3)

            residuals.append(np.vstack((self.spectra[i][:,0], self.spectra[i][:,1] / self.spectra[i][ind,1] / new_filt - 1)).T)
        
        if just_resid is False:
            plt.xlabel(r'$\lambda$',fontsize=12)
            plt.tick_params(labelsize = 12)
            if plot_mode == 'texp':
                #plt.ylim(-2,57)
                #plt.gca().invert_yaxis()
                plt.plot([3500, 10500], [0,0], color = 'black', ls = '--', alpha = 0.3)
                plt.plot([3500, 10500], [-55,-55], color = 'black', ls = '--', alpha = 0.3)

            for i in range (len(mask)):
                plt.axvspan(mask[i][0], mask[i][1], color = 'grey', alpha = 0.1)
            
        self.residuals = residuals
        
        
    def resample_spectra_to_grid(self):
        wl_new_grid = np.arange(3500, 9500, 1)
        
        self.new_grid = wl_new_grid
        
        resampled_spec_s = np.full((len(self.spectra), wl_new_grid.shape[0]), np.nan)
        for i in range (len(self.spectra)):
            cond_on_resid = np.absolute(self.residuals[i][:,1]) < 5
            cond = (wl_new_grid > min(self.spectra[i][cond_on_resid,0])) & (wl_new_grid < max(self.spectra[i][cond_on_resid,0]))
            
            res_fl = resample_spec(self.spectra[i][cond_on_resid,0], self.residuals[i][cond_on_resid,1], wl_new_grid[cond])
            resampled_spec_s[i][cond] = res_fl
            
        self.res_spectra = resampled_spec_s
        
def GP_fitter(sn, index, unc_val = 0.025, wh_noise = 0.001):
    
    kernel = 0.3 * kernels.Matern32Kernel(200)
    gp = george.GP(kernel, fit_white_noise = True, white_noise = log(wh_noise**2))

    cond_nan = ~np.isnan(sn.res_spectra[:,index])
    
    if sum(cond_nan) < 3:
        raise ValueError
    
    def nll(p):
        gp.set_parameter_vector(p)
        ll = gp.log_likelihood(sn.res_spectra[cond_nan, index], quiet=True)
        return -ll if np.isfinite(ll) else 1e25

    def grad_nll(p):
        gp.set_parameter_vector(p)
        return -gp.grad_log_likelihood(sn.res_spectra[cond_nan, index], quiet=True)

    gp.compute(np.array(sn.times)[cond_nan], np.full(sum(cond_nan), unc_val))

    p0 = gp.get_parameter_vector()
    results = minimize(nll, p0, jac=grad_nll, method="L-BFGS-B")

    cond = (time_grid > min(np.array(sn.times)[cond_nan])) & (time_grid < max(np.array(sn.times)[cond_nan]))
    x_pred = time_grid[cond]

    pred, pred_var = gp.predict(sn.res_spectra[cond_nan, index], x_pred, return_var=True)
    
    return x_pred, pred, pred_var

def get_quick_GP_plot(SN, wl_ch, xll = [6000, 7100], yll = [-1.7,2.1], unc_val = 0.025, wh_noise = 0.001, legoff = False):
    f = plt.figure(figsize=(18,6))

    ax = f.add_subplot(1,2,1)
    evenly_spaced_interval = np.linspace(0.0, 0.9, len(SN.times)) #len(specs_g))
    cll = [cm.jet(x) for x in evenly_spaced_interval]

    for i in range (len(SN.times)):
        plt.plot(SN.new_grid, SN.res_spectra[i],
                 color = cll[i], label = np.round(SN.times[i],2))
    plt.plot([3000, 10000], [0,0], color = 'red', ls = '--', alpha = 0.3)
    plt.xlim(xll[0], xll[1])
    plt.ylim(yll[0], yll[1])
        #text(7000, 0.95 - (i-1) * 1.5, str(round(times[i],2)) + ' d')

    plt.xlabel(r'$\lambda$',fontsize=15)
    plt.tick_params(labelsize = 14)

    plt.axvline(wl_ch, ls = '--', color = 'red')
    if not legoff:
        plt.legend(fontsize = 10, ncol = 3)

    ax = f.add_subplot(1,2,2)

    index = np.argmin(abs(SN.new_grid - wl_ch))

    plt.plot(SN.times, SN.res_spectra[:,index], '--', color = 'black', lw = 1)
    plt.errorbar(SN.times, SN.res_spectra[:,index], np.full(len(SN.times), unc_val), marker = '.',color = 'black',
                 capsize = 4, zorder = 10)
    plt.scatter(SN.times, SN.res_spectra[:,index], c = SN.times, zorder = 11)

    x_pred, pred, pred_var = GP_fitter(SN, index, unc_val, wh_noise)

    plt.plot(x_pred, pred, color = 'navy', lw=1.5, alpha=0.5)
    plt.fill_between(x_pred, pred - np.sqrt(pred_var), pred + np.sqrt(pred_var),
                            color='navy', alpha=0.2, label = 'GP fit')

    plt.xlabel(r'Epoch',fontsize = 15)
    plt.ylabel('Flux (n.)', fontsize = 15)
    plt.tick_params(labelsize = 14)
    
    plt.legend(fontsize = 16)

    plt.title('Change of flux at ' + str(wl_ch) + ' AA', fontsize = 15)

    plt.ylim(-1.0,1.5)
    
def setup_mask(temp, mask, minmax):
    spec_cond = np.ones_like(temp[:,0], dtype='bool')
    for band in mask:
        band_mask = np.logical_and(temp[:,0] > band[0],
                                   temp[:,0] < band[1])
        inverse_band_mask = np.logical_not(band_mask)
        spec_cond = np.logical_and(inverse_band_mask, spec_cond)
    minmax_mask = np.logical_and(temp[:,0] > minmax[0],
                                   temp[:,0] < minmax[1])
    spec_cond = np.logical_and(minmax_mask, spec_cond)
    
    return spec_cond
    
def get_final_matrices(SN, unc_val = 0.025, wh_noise = 0.001):
    result_matrix = np.full((time_grid.shape[0], SN.res_spectra.shape[1]), np.nan)
    unc_matrix = np.full((time_grid.shape[0], SN.res_spectra.shape[1]), np.nan)

    for i in tqdm(range (SN.res_spectra.shape[1])):
        try:
            x_pred, pred, pred_var = GP_fitter(SN, i, unc_val, wh_noise)
        except ValueError:
            continue
        if len(x_pred) == 0:
            continue ## If only the later spectra cover a region this can occur
        start_i = list(time_grid).index(x_pred[0])

        result_matrix[start_i:start_i + len(x_pred),i] = pred
        unc_matrix[start_i:start_i + len(x_pred),i] = pred_var
    return result_matrix, unc_matrix

def get_med_values(data,a):
    med_sp_fl = []
    i = 0
    while i < (data.shape[0]):
        if i < int(a/2):
            med_sp_fl.append(np.median(data[0:i+int(a/2),1]))
        elif i + int(a/2) < data.shape[0]:
            temp = data[i-int(a/2):i+int(a/2)]
            dffs = np.diff(temp[:,0])
            if any(dffs > 100):
                max_j = get_index_of_max(dffs) + 1
                if max_j - int(a/2) < 1:
                    med_sp_fl.append(np.median(temp[max_j:,1]))
                else:
                    med_sp_fl.append(np.median(temp[:max_j,1]))
            else:
                med_sp_fl.append(np.median(temp[:,1]))
        else:
            med_sp_fl.append(np.median(data[i-int(a/2):,1]))
        i += 1
    return np.array(med_sp_fl)

def get_index_of_max(data):
    return list(data).index(max(data))
