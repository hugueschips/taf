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

def rolling_correlation_convolution(signal, fs, window=1):
    '''
        signal is the array of amplitude of HHT of the chosen IMF
        window is the size of the window in seconds or tours
    '''
    n = len(signal)
    window *= int(1*fs)
    corr = []
    time_ref = 1*fs        # tour_ref if normalized
    seconds_before = 1  # tour_before if normalized
    seconds_after = 3   # tour_after if normalized
    while time_ref + seconds_after*fs < n:
        start, end = window_idx(time_ref, seconds_before, seconds_after, fs)
        signal_slice = signal[start:end]
        print time_ref, start, end, len(signal_slice)
        cr = localAutoCorrelate(signal_slice, window)
        corr += list(cr)
        time_ref += (seconds_before + seconds_after) * fs
    return corr
