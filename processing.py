# -*- coding: utf-8 -*-
'''
Created on 13th may 2016
@author: Daniel Durrenberger
daniel.durrenberger@amaris.com

Python 2.7.12
Pandas 0.18.1
Numpy 1.11.1
Scipy 0.18.0
'''

import numpy as np
import scipy

def window_idx(time_ref, seconds_before, seconds_after, fs):
    start = int(time_ref - seconds_before * fs)
    end = int(time_ref + seconds_after * fs)
    return start, end

def localAutoCorrelate(signal, window):
    '''
    computes local autocorrelation, with a bit of normalization
    to compare results more easily
    '''
    n = len(signal)
    half = (n - window)/2
    m1 = signal[:half].mean()
    m2 = signal.mean()
    cr = 1./(1.*n) * np.correlate(signal-m2, signal[:half]-m1, mode='valid')
    #print(len(cr))
    return cr[50:-50]

def is_sticking_on_window(
    window_start, window_end,
    sticking_start, sticking_end
    ):
    return sticking_start<=window_start and window_end<=sticking_end

def average_speed_on_window(
    time_zero,
    speed,
    window_start,
    window_end,
    fs=10000
    ):
    '''
    returns average speed on given time window_end
    speed is the complete array
    window_start and window_end must be in seconds
    '''
    start = int(fs * (window_start-time_zero))
    end = int(fs * (window_end-time_zero))
    crop = speed[start: end]
    av = np.mean(crop)
    #if np.isnan(av):
    #    print crop
    #    av = 0.
    return av

def rolling_correlation_convolution(
    signal,
    fs,
    slice_size,
    window_size,
    beginning=0
    ):
    '''
    signal is the array of amplitude of HHT of the chosen IMF
    window is the size of the window in seconds or tours
    returns two lists of lists
        one list of time index per time window
        one list of correlation per time window
    you can plot them with a for loop on zip(t, corr)
    '''
    n = signal.shape[-1]
    dt = 1./fs
    window = int(window_size*fs)
    t = []
    corr = []
    time_ref = 1*fs        # tour_ref if normalized
    seconds_before = 0  # tour_before if normalized
    seconds_after = slice_size   # tour_after if normalized
    while time_ref + seconds_after*fs < n:
        start, end = window_idx(time_ref, seconds_before, seconds_after, fs)
        signal_slice = signal[start:end]
        cr = localAutoCorrelate(signal_slice, window)
        localTime = beginning + np.linspace(start*dt, end*dt, len(cr))
        t.append(list(localTime))
        corr.append(list(cr))
        time_ref += 1 * (seconds_before + seconds_after) * fs
    return t, corr

def fft_of_correlation(
    corr,
    fs,
    start_freq=1,
    end_freq=20
    ):
    '''
    returns two lists of lists
        one list of frequency index per time window
        one list of fft per time window
    '''
    dt = 1./fs
    freq_list = []
    fft_list = []
    for correlation in corr:
        nTot = len(correlation)
        nPoints_start_freq = int( nTot*dt*start_freq )
        nPoints_end_freq = int( nTot*dt*end_freq ) + 1
        idx = np.arange(nPoints_start_freq, nPoints_end_freq)
        fft = np.fft.rfft(correlation)
        freq = np.fft.rfftfreq(nTot, dt)
        freq_list.append( freq[idx] )
        fft_list.append( np.abs(fft[idx]) )
    return freq_list, fft_list

def interpolate_peak(x, y):
    '''
    returns (x, y) graph smoothed with splines on n points
    '''
    if len(x)<10:
        return x, y
    n = 5*len(x)
    s = scipy.interpolate.InterpolatedUnivariateSpline(x, y)
    xnew = np.linspace(x[0], x[-1], n)
    ynew = s(xnew)
    return xnew, ynew

def peak_coordinates(x, y):
    '''
    returns the coordinates of the maximum of the graph (x, y)
    '''
    xnew, ynew = interpolate_peak(x, y)
    imax = np.argmax(ynew)
    xmax = xnew[imax]
    ymax = ynew[imax]
    return xmax, ymax

def peak_list(freq_list, fft_list):
    '''
    returns two lists
        one with the most important frequencies values per time window
        one with the corresponding amplitudes per time window
    '''
    xpeak = []
    ypeak = []
    for freq, fft in zip(freq_list, fft_list):
        x, y = peak_coordinates(freq, fft)
        xpeak.append(x)
        ypeak.append(y)
    return xpeak, ypeak
