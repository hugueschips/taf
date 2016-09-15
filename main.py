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

def get_peaks(
        coil=28,
        cropTime=[60, 180],
        normalize=False,
        num_imfs=3,
        graphics=False
        ):
    ############################# IMPORT COIL #################################
    print('...import coil '+str(coil)+' from hdf...')
    dfi = md.import_dfi()
    df = md.import_data(coil=coil)
    t, signal, speed, decoiler, coiler = md.dfToArrays(df)
    a, b, n, dt, fs = md.xInfo(t)
    print('          ...'+str(n)+' points...')

    ############################# ABOUT COIL ##################################
    thickness = dfi.thickness[coil]
    sticking = dfi.sticking[coil]
    duration = dfi.duration[coil]-5
    if sticking:
        sti, sei = dfi.t_begin[coil], dfi.t_end[coil]
        cropTime = [int(sti*0.9), min(sti+120., duration)]
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
        print('OH OH, y a comme un souci !')
    elapsedTime = np.round(time.time()-startTime, 1)
    print('          ...in '+str(elapsedTime)+'s... ')

    ############################# STICKING INDICATOR & SPEED ##################
    av_speed = []
    sticking_indicator = []
    time_start = []
    time_end = []
    for window in t:
        window_start = window[0]
        window_end = window[-1]
        av_speed_value = pc.average_speed_on_window(
                                                    a,
                                                    speed,
                                                    window_start,
                                                    window_end,
                                                    fs=fs
                                                    )
        if sticking:
            window_indicator = pc.is_sticking_on_window(
                                                    window_start, window_end,
                                                    sti, sei
                                                    )
        else:
            window_indicator = False
        time_start.append(window_start)
        time_end.append(window_end)
        sticking_indicator.append(window_indicator)
        av_speed.append(av_speed_value)

    ############################# PERFORM FFT #################################
    print('...perform FFT...')
    startTime = time.time()
    fft_list_imf = []
    for imf in corr_imf:
        freq_list, fft_list = pc.fft_of_correlation(imf, fs)
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
    df, storeName = md.store_peaks(
                                xpeak_imf,
                                ypeak_imf,
                                sticking_indicator,
                                time_start,
                                time_end,
                                coil,
                                thickness,
                                av_speed,
                                'peaks.h5'
                                )

    ############################# GRAPHICS ####################################
    if graphics:
        print('...produce graphics...')
        startTime = time.time()
        i = 0
        for imf in corr_imf:
            fig = utils.plot_autocorrelation(t, imf, fs,
                                            metadata+' imf '+str(i)
                                            )
            i += 1
        elapsedTime = np.round(time.time()-startTime, 1)
        print('          ...in '+str(elapsedTime)+'s... ')
        plt.show()
    return xpeak_imf, ypeak_imf, storeName

startTime = time.time()
for coil in range(0,88):
    try:
        get_peaks(
                coil=coil,
                cropTime=[60,180],
                graphics=False
                )
        if np.mod(coil,10)==0:
            soFarDuration = time.time()-startTime
            estimatedTimeLeft = (88-coil)*soFarDuration/coil
            print('                       SO FAR TIME : '+str(totalTime)+' min')
            print('               ESTIMATED LEFT TIME : '+str(totalTime)+' min')
    except:
        pass
totalTime = time.time()-startTime
print('TOTAL TIME : '+str(totalTime)+' min')
