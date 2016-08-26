# -*- coding: utf-8 -*-
'''
Created on 25th July 2016
@author: Daniel Durrenberger
daniel.durrenberger@amaris.com

Python 2.7.12
Pandas 0.18.1
Numpy 1.11.1
Scipy 0.18.0
'''
import pyeemd
import utils
import time
import numpy as np
import scipy
import matplotlib.pyplot as plt
import manage_data as md
import preprocessing as pp
import processing as pc

def main(
        coil=28,
        cropTime=[90, 110],
        normalize=False,
        num_imfs=3,
        ):
    ############################# IMPORT COIL #################################
    print('...import coil '+str(coil)+' from hdf...')
    dfi = md.import_dfi()
    df = md.import_data(coil=coil)
    t, signal, speed, decoiler, coiler = md.dfToArrays(df)
    a, b, n, dt, fs = md.xInfo(t)
    print('          ...'+str(n)+' points...')

    ############################# DIVIDE BY RMS ###############################
    print('...divide signal by RMS on a 5s window...')
    signal /= pp.fast_rms(signal)

    ############################# DIVIDE BY RMS ###############################
    if cropTime is not None:
        i0, iN = int(fs*cropTime[0]), int(fs*cropTime[1])
        print('...crop between '+str(cropTime)+'s...')
        t, signal, speed, decoiler, coiler = md.dfToArrays(df, i0, iN)
        a, b, n, dt, fs = md.xInfo(t)
        print('          ...'+str(n)+' points...')

    ############################# NORMALIZE IN TOUR SPACE #####################
    if normalize:
        print('...normalize signal in tour space...')
        thickness = dfi.thickness[coil]
        a, b, fnorm = pp.angular_normalisation(signal, decoiler, thickness)
        t = np.linspace(a, b, n)
        signal = fnorm(t)
        a, b, n, dt, fs = md.xInfo(t)
        print('          ...'+str(n)+' points...')

    ############################# PERFORM EMD #################################
    print('...perform EMD...')
    startTime = time.time()
    mode = pyeemd.emd(signal, num_imfs=num_imfs, num_siftings=None)
    elapsedTime = np.round(time.time()-startTime, 1)
    print('          ...in '+str(elapsedTime)+'s... ')

    ############################# PERFORM HHT+ABS #############################
    print('...perform HHT...')
    startTime = time.time()
    hht = scipy.signal.hilbert(mode)
    imf = np.abs(hht)
    elapsedTime = np.round(time.time()-startTime, 1)
    print('          ...in '+str(elapsedTime)+'s... ')

    ############################# AUTOCORRELATION #############################
    print('...perform autocorrelation...')
    startTime = time.time()
    nimf = imf.shape[0]
    corr_imf = []
    for i in range(nimf):
        print('          ...on IMF '+str(i)+'...')
        signal = imf[i,:]
        t, corr = pc.rolling_correlation_convolution(
                                                    signal,
                                                    fs,
                                                    beginning=cropTime[0]
                                                    )
        corr_imf.append(corr)
    if nimf!=len(corr_imf):
        print 'OH OH, y a comme un souci !'
    elapsedTime = np.round(time.time()-startTime, 1)
    print('          ...in '+str(elapsedTime)+'s... ')

    ############################# PERFORM FFT #################################
    print('...perform FFT...')
    startTime = time.time()
    fft_list_imf = []
    for imf in corr_imf:
        freq_list, fft_list = pc.fft_of_correlation(corr, fs)
        fft_list_imf.append(fft_list)
    elapsedTime = np.round(time.time()-startTime, 1)
    print('          ...in '+str(elapsedTime)+'s... ')

    ############################# COMPUTE PEAKS ###############################
    print('...compute peaks...')
    startTime = time.time()
    xpeak_imf = []
    ypeak_imf = []
    for fft_list in fft_list_imf:
        for freq, fft in zip(freq_list, fft_list):
            xpeak, ypeak = pc.peak_coordinates(freq, fft)
            xpeak_imf.append(xpeak)
            ypeak_imf.append(ypeak)
    elapsedTime = np.round(time.time()-startTime, 1)
    print('          ...in '+str(elapsedTime)+'s... ')

    ############################# DIVIDE BY RMS ###############################
    print xpeak_imf, ypeak_imf

    return xpeak_imf, ypeak_imf

main()
