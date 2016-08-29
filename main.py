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
        cropTime=[90, 150],
        normalize=False,
        num_imfs=3,
        graphics=True
        ):
    ############################# IMPORT COIL #################################
    print('...import coil '+str(coil)+' from hdf...')
    dfi = md.import_dfi()
    df = md.import_data(coil=coil)
    t, signal, speed, decoiler, coiler = md.dfToArrays(df)
    a, b, n, dt, fs = md.xInfo(t)
    print('          ...'+str(n)+' points...')

    ############################# ABOUT COIL ##################################
    sticking = dfi.sticking[coil]
    if sticking:
        startS, endS = dfi.startS[coil], dfi.endS[coil]
        print '          ...', startS, endS, coiler[0], coiler[-1]
        st, se = pp.when_sticking(coiler, t, startS, endS)
        sti, sei = int(st), int(se)+1
        print('...coil is sticking from '+str(sti)+' to '+str(sei)+'s...')
        stick = ' sticking in ['+str(sti)+','+str(sei)+']'
    else:
        print('...no marks have been detected on this coil...')
        stick = ' not sticking'
    metadata = 'Coil '+str(coil)+stick

    ############################# DIVIDE BY RMS ###############################
    print('...divide signal by RMS on a 5s window...')
    signal /= pp.fast_rms(signal)

    ############################# CROP TIME ZONE ##############################
    beginning = 0 # used for autocorrelation xaxis
    if cropTime is not None:
        i0, iN = int(fs*cropTime[0]), int(fs*cropTime[1])
        beginning = cropTime[0]  # used for autocorrelation xaxis
        print('...crop between '+str(cropTime)+'s...')
        t, signal, speed, decoiler, coiler = md.dfToArrays(df, i0, iN)
        a, b, n, dt, fs = md.xInfo(t)
        metadata += ' cropped on '+str(cropTime)
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
                                                    beginning=beginning
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
        xpeak, ypeak = pc.peak_list(freq_list, fft_list)
        xpeak_imf.append(xpeak)
        ypeak_imf.append(ypeak)
    elapsedTime = np.round(time.time()-startTime, 1)
    print('          ...in '+str(elapsedTime)+'s... ')
    df, storeName = md.store_peaks(xpeak_imf, ypeak_imf, coil, 'peaks.h5')

    ############################# GRAPHICS ####################################
    print('...produce graphics...')
    startTime = time.time()
    i = 0
    for imf in corr_imf:
        fig = utils.plot_autocorrelation(t, imf, fs, metadata+' imf '+str(i))
        i += 1
    elapsedTime = np.round(time.time()-startTime, 1)
    print('          ...in '+str(elapsedTime)+'s... ')
    if graphics:
        plt.show()

    return xpeak_imf, ypeak_imf, storeName

# for coil in range(3,12):
#     main(coil=coil, graphics=True)

#main()
