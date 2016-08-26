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
        num_imfs=2,
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

    ############################# DIVIDE BY RMS ###############################
    ############################# DIVIDE BY RMS ###############################
    ############################# DIVIDE BY RMS ###############################
    ############################# DIVIDE BY RMS ###############################
    ############################# DIVIDE BY RMS ###############################

    return
