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
import scipy.io.wavfile as wav

def dfToArrays(df, i0=0, iN=-1):
    i0, iN = int(i0), int(iN)
    time = df.time.values[i0:iN]
    signal = df.micro.values[i0:iN]
    speed = df.speed.values[i0:iN]
    decoiler = df.decoiler.values[i0:iN]
    coiler = df.coiler.values[i0:iN]
    return time, signal, speed, decoiler, coiler

def xInfo(x):
    a, b = x[0], x[-1]
    n = len(x)
    dx = (b-a)/(n-1)
    fs = int(1//dx)+1
    return a, b, n, dx, fs

def main(
        nDataKind=1,
        coil=28,
        divideByRMS=True,
        cropMaxSpeed=False,
        cropTime=[90, 150],
        butterFilter=False,
        normalize=True,
        movingAverage=0,
        resampleBadlyFactor=1,
        resampleFactor=[1],
        autoCorrelation=False,
        num_imfs=4,
        showEMD=False,
        exportWav=False,
        showAmp=False,
        showHHT=True,
        getMaxFreq=False,
        showSpectrogram=False
        ):
    """
    Kind of data
        0 - raw text file
        1 - hdf
    """

    ################ Title for graphs #########################################
    metadata_tot = (
                'nDataKind=' + str(nDataKind) +
                ' coil=' + str(coil) +
                ' divideByRMS=' + str(divideByRMS) +
                ' cropMaxSpeed=' + str(cropMaxSpeed) +
                ' cropTime=' + str(cropTime) +
                ' butterFilter=' + str(butterFilter) +
                '\nnormalize=' + str(normalize) +
                ' movingAverage=' + str(movingAverage) +
                ' resampleBadlyFactor=' + str(resampleBadlyFactor) +
                ' resampleFactor=' + str(np.prod(resampleFactor)) +
                ' autoCorrelation=' + str(autoCorrelation)
                )
    metadata = (
                ' coil=' + str(coil) +
                ' RMS=' + str(divideByRMS) +
                ' cropSpeed=' + str(cropMaxSpeed) +
                ' cropTime=' + str(cropTime) +
                ' butter=' + str(butterFilter) +
                ' normalize=' + str(normalize) +
                ' movingAve=' + str(movingAverage) +
                ' rsmpl1=' + str(resampleBadlyFactor) +
                ' rsmpl2=' + str(np.prod(resampleFactor)) +
                ' autoCorr=' + str(autoCorrelation)
                )
    print(metadata_tot + '\n')
    code  = (
            str(movingAverage) +
            str(resampleBadlyFactor) +
            str(str(np.prod(resampleFactor)))
            )

    ################ Preprocess ###############################################
    if nDataKind==0:    # raw text file
        print('...import coil '+str(coil)+' from txt file...')
        df, dfi = md.import_txt_sound(28, dfi=None)
        print('          ...'+str(n)+' points...')
    elif nDataKind==1:  # HDF
        print('...import coil '+str(coil)+' from hdf...')
        dfi = md.import_dfi()
        df = md.import_data(coil=coil)
        t, signal, speed, decoiler, coiler = dfToArrays(df)
        a, b, n, dt, fs = xInfo(t)
        print('          ...'+str(n)+' points...')
    else:
        print('Try nDataKind = 1')
        return()

    if divideByRMS:
        print('...divide signal by RMS on a 5s window...')
        signal /= pp.fast_rms(signal)

    if cropMaxSpeed:
        i0, iN = md.max_speed_seq(speed, samplerate=fs)
        print('...crop between '+str(int(i0*dt))+' and '+str(int(iN*dt))+'s...')
        t, signal, speed, decoiler, coiler = dfToArrays(df, i0, iN)
        a, b, n, dt, fs = xInfo(t)
        print('          ...'+str(n)+' points...')

    if cropTime is not None:
        i0, iN = int(fs*cropTime[0]), int(fs*cropTime[1])
        print('...crop between '+str(cropTime)+'s...')
        t, signal, speed, decoiler, coiler = dfToArrays(df, i0, iN)
        a, b, n, dt, fs = xInfo(t)
        print('          ...'+str(n)+' points...')

    if butterFilter:
        print('...Butter filter bandpass...')
        nyq = 0.5*fs
        low = 16 / nyq
        high = fs/2 / nyq
        bb, aa = scipy.signal.butter(8, [low, high], 'bandpass', analog=True)
        signal = scipy.signal.lfilter(bb, aa, signal)
        #plt.figure()
        #plt.plot(t, signal)

    if movingAverage>1:
        print(
            '...perform moving average on signal on '
            +str(movingAverage)+' points...'
            )
        signal = pp.moving_average(signal, movingAverage)

    if normalize:
        print('...normalize signal in tour space...')
        thickness = dfi.thickness[coil]
        a, b, fnorm = pp.angular_normalisation(signal, decoiler, thickness)
        t = np.linspace(a, b, n)
        signal = fnorm(t)
        a, b, n, dt, fs = xInfo(t)
        print('          ...'+str(n)+' points...')

    if resampleBadlyFactor>1:
        print(
                '...downsampling signal with factor '
                +str(resampleBadlyFactor)+'...'
            )
        t, signal = pp.resample(t, signal, resampleBadlyFactor)
        a, b, n, dt, fs = xInfo(t)
        print('          ...'+str(n)+' points...')

    if np.prod(resampleFactor)>1:
        globalFactor = np.prod(resampleFactor)
        print('...downsampling signal with factor '+str(globalFactor)+'...')
        for factor in resampleFactor:
            signal = scipy.signal.decimate(signal, factor, zero_phase=True)
        t = np.linspace(a, b, len(signal))
        a, b, n, dt, fs = xInfo(t)
        print('          ...'+str(n)+' points...')

    ################ Perform EMD ##############################################
    print('...perform EMD...')
    startTime = time.time()
    mode = pyeemd.emd(signal, num_imfs=num_imfs, num_siftings=None)
    elapsedTime = np.round(time.time()-startTime, 1)
    print('          ...in '+str(elapsedTime)+'s... ')
    if showEMD:
        utils.plot_imf(t, mode, metadata, 'EMD')
    if exportWav:
        utils.export_as_wav('coil'+str(coil)+'_imf', fs, mode)

    ################ Perform HHT ##############################################
    if not showHHT and not showSpectrogram and not showAmp:
        plt.show()
        return()
    print('...perform HHT...')
    startTime = time.time()
    hht = scipy.signal.hilbert(mode)
    signal = np.abs(hht)
    if autoCorrelation:
        for i in range(signal.shape[0]):
            signal[i,:] = pp.auto_correlate(signal[i,:], True, True)
    elapsedTime = np.round(time.time()-startTime, 1)
    print('          ...in '+str(elapsedTime)+'s... ')
    if showAmp:
        utils.plot_imf(t, signal, metadata, 'AMP')

    ################ Get max frequecies #######################################
    if getMaxFreq:
        print('...Compute max frequencies...')
        startTime = time.time()
        max_freq = pp.get_max_freq_list(signal[0,:], dt, 10000)
        print('          ...in '+str(elapsedTime)+'s... ')
        print max_freq
        plt.figure(figsize=(12,4))
        plt.plot(max_freq)

    ################ Perform FFT ##############################################
    print('...perform FFT...')
    startTime = time.time()
    #hht *= scipy.signal.blackmanharris(len(hht))
    fft = np.fft.rfft(signal)
    elapsedTime = np.round(time.time()-startTime, 1)
    print('          ...in '+str(elapsedTime)+'s... ')
    if showHHT:
        utils.plot_fft(dt, np.abs(fft), metadata)

    ################ Produce spectrogram ######################################
    if showSpectrogram:
        print('...produce spectrogram...')
        startTime = time.time()
        utils.plot_spectrogram(hht, fs, t, metadata)
        elapsedTime = np.round(time.time()-startTime, 1)
        print('          ...in '+str(elapsedTime)+'s... ')
        #plt.savefig('spectro28_'+code+'.png')

    #plt.ion()
    plt.show()
    return


main()
