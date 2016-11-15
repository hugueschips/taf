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
import scipy.signal
import scipy.interpolate
import matplotlib.pyplot as plt

def sticking_detection(d, startS, endS):
    sticking = (startS<=d) & (d<=endS)
    return sticking.astype(int)

def fast_rms(y, window=5, samplerate=10000.):
    n = int(window*samplerate)
    conv_window = np.ones(n, np.float)/n
    rms = scipy.signal.fftconvolve(y**2, conv_window, mode='same')
    return np.sqrt(rms)

def std(y, window=5, samplerate=10000):
    n = window*samplerate
    conv_window = np.ones(n, np.float)/n
    mean = scipy.signal.fftconvolve(y, conv_window, mode='same')
    rms = scipy.signal.fftconvolve(y**2, conv_window, mode='same')
    return np.sqrt(rms-mean**2)

def angular_normalisation(y, d, e, kind='linear'):
    r_int = 0.305                         # in m, inner radius of the decoiler
    d = d-np.min(d)               # finishes at 0 and removes negatives values
    r = np.sqrt(e*d/np.pi + r_int**2)         # in m, outer radius of the coil
    wrapped_tour = (r-r_int)/e          # nbr of wrapped tours on the decoiler
    unwrapped_tour = wrapped_tour[0]-wrapped_tour
    f = scipy.interpolate.interp1d(unwrapped_tour, y, kind=kind)
    a, b = unwrapped_tour[0], unwrapped_tour[-1]
    return a, b, f

def angular_correspondance(d, e):
    '''
    returns a fonction f such as
    f(decoiler) = number of unwrapped_tour
    '''
    d = d-np.min(d)
    r_int = 0.305                         # in m, inner radius of the decoiler
    d = d-np.min(d)               # finishes at 0 and removes negatives values
    r = np.sqrt(e*d/np.pi + r_int**2)         # in m, outer radius of the coil
    wrapped_tour = (r-r_int)/e          # nbr of wrapped tours on the decoiler
    unwrapped_tour = wrapped_tour[0]-wrapped_tour
    f = scipy.interpolate.interp1d(d, np.abs(unwrapped_tour))
    return f

def angular_correspondance_extended(x, d, e):
    '''
    avoids out of range x values
    '''
    d = d-np.min(d)
    x = np.max(x, d[0])
    x = np.min(x, d[-1])
    f = angular_correspondance(d, e)
    return np.max(f(x), 0.)

def is_between(x, a, b):
    '''
    return a boolean array like x to show if
                a < x < b
    '''
    # permute a and b if necessary
    if a>b:
        c = a
        a = b
        b = c
    return np.logical_and(a<x, x<b)

def resample(x, y, factor):
    """
    Resamples data to a lower frequency to allow faster computation
    """
    if factor>1:
        f = scipy.interpolate.interp1d(x, y)
        n = len(y)/factor
        x = np.linspace(x[0], x[-1], n)
        y = f(x)
    return x, y

def butter_lowpass_filter(signal, highcut, fs, order=5):
    '''
    Applies a Butterworth lowpass filter cutting at [highcut] 'Hz' on signal
    '''
    nyq = 0.5*fs
    high = highcut/nyq
    b, a = scipy.signal.butter(order, high, btype='low')
    y = scipy.signal.lfilter(b, a, signal)
    return y

def freq_from_fft(signal, fs):
	"""
    Estimates frequency from peak of FFT
	"""
	# Compute Fourier transform of windowed signal
	windowed = signal * scipy.signal.blackmanharris(len(signal))
	f = np.fft.rfft(windowed)
	# Find the peak and interpolate to get a more accurate peak
	i = np.argmax(np.abs(f)) # Just use this for less-accurate, naive version
	true_i = parabolic.parabolic(np.log(np.abs(f)), i)[0]   # abs(f)
	# Convert to equivalent frequency
	return fs * true_i / len(windowed)

def instantaneous_freq(hht_signal, fs):
    '''
    Calculates the instantaneous frequency phase of hht_signal.
    Adds np.nan at the beginning to keep the same size as hht_signal
    if one unquotes and returns reshaped_instfreq.
    '''
    instantaneous_phase = np.unwrap(np.angle(hht_signal))
    instantaneous_frequency = np.diff(instantaneous_phase) / (2.0*np.pi) * fs
    reshaped_instfreq = np.r_[np.nan, instantaneous_frequency]
    return instantaneous_frequency

def moving_average(a, n=3) :
    coef = np.ones(n)/n
    return np.correlate(a, coef, 'same')

def auto_correlate(signal, std=True, mean=True):
    cor = np.array(signal)
    cor *= np.hamming(len(cor))
    if std:
        cor -= np.std(cor)
    if mean:
        cor -= np.mean(cor)
    out = np.correlate(cor, cor, 'same')
    return out

def window_idx(len_signal, window):
    '''
    returns a list containing the window starts
    used in get_max_freq_list
    '''
    windowStart = [0]
    window = min( window, len_signal )
    while window>5:
        windowStart.append( windowStart[-1]+window )
        window = min( window, len_signal-windowStart[-1] )
    return windowStart

def get_max_freq_list(signal, dt, window):
    '''
    returns a list with a maximum frequency peak for each window-sized
    window of signal
    '''
    max_freq = []
    windowStart = window_idx(len(signal), window)
    windowSize = np.diff(windowStart)
    plt.figure(figsize=(12,8))
    for i in range(len(windowStart)-1):
        i0, iK = windowStart[i], windowStart[i+1]
        s = signal[i0:iK]
        cor = auto_correlate(s)
        fft = np.abs(np.fft.rfft(cor))
        freq = np.fft.fftfreq(len(fft), dt)
        plt.plot(freq, fft)
        max_freq.append(freq[np.argmax(fft)])
    return max_freq

def when_sticking(coiler, time, startS, endS):
    '''
    returns time sticking zone in seconds, given space sticking zone
    out of coiler array
    '''
    startS = max(coiler[0], startS*1.)
    endS = min(coiler[-1], endS*1.)
    f = scipy.interpolate.interp1d(coiler, time)
    return int(f(startS)), int(f(endS)+1)
