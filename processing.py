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
import preprocessing as pp

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
    return cr[2:-2]

def rolling_correlation_convolution(signal, fs, beginning=0):
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
    window = int(1*fs)
    t = []
    corr = []
    time_ref = 1*fs        # tour_ref if normalized
    seconds_before = 1  # tour_before if normalized
    seconds_after = 3   # tour_after if normalized
    while time_ref + seconds_after*fs < n:
        start, end = window_idx(time_ref, seconds_before, seconds_after, fs)
        signal_slice = signal[start:end]
        print time_ref, start, end, len(signal_slice)
        cr = localAutoCorrelate(signal_slice, window)
        t.append(list(beginning + np.linspace(start*dt, end*dt, len(cr))))
        corr.append(list(cr))
        time_ref += (seconds_before + seconds_after) * fs
    return t, corr

def fft_of_correlation(corr, fs):
    '''
        returns two lists of lists
            one list of frequency index per time window
            one list of fft per time window
    '''
    dt = 1./fs
    freq_list = []
    fft_list = []
    for correlation in corr:
        fft = np.fft.fft(correlation)
        freq = np.fft.fftfreq(len(correlation), 1./fs)
        nTot = len(fft)
        nPoints_to_2Hz = int(nTot*dt*2)
        nPoints_to_10Hz = int(nTot*dt*10)
        idx = np.arange(nPoints_to_2Hz,nPoints_to_10Hz)
        freq_list.append( freq[idx] )
        fft_list.append( np.abs(fft[idx]) )
    return freq_list, fft_list

def peak(x, y):
    '''
        returns the coordinates of the maximum of the graph (x, y) in a list
    '''
    imax = np.argmax(y)
    xmax = x[imax]
    ymax = y[imax]
    return list([xmax, ymax])

def peak_list(freq_list, fft_list):
    '''
        returns a list of lists
            one list [xmax, ymax] per time window
    '''
    peaks = []
    for freq, fft in zip(freq_list, fft_list):
        peaks.append(peak(freq, fft))
    return peaks
