#!/usr/bin/env python2
# vim: set fileencoding=utf-8 ts=4 sw=4 lw=79

# Copyright 2013 Perttu Luukko

# This file is part of libeemd.
#
# libeemd is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# libeemd is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with libeemd.  If not, see <http://www.gnu.org/licenses/>.

"""
Some utility functions for visualizing IMFs produced by the (E)EMD
methods.
"""

#from matplotlib.pylab import *
import os
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.io.wavfile
import preprocessing as pp
from pyeemd import emd_find_extrema, emd_evaluate_spline
#plt.ion()

def plot_imfs(imfs, new_figs=True, plot_splines=False):
    """
    Plot utility method for plotting IMFs and their envelope splines with
    ``pylab``.

    Parameters
    ----------
    imfs : ndarray
        The IMFs as returned by :func:`pyeemd.emd`,
        :func:`pyeemd.eemd`, or :func:`pyeemd.ceemdan`.

    new_figs : bool, optional
        Whether to plot the IMFs in separate figures.

    plot_splines : bool, optional
        Whether to plot the envelope spline curves as well.
    """
    for i in range(imfs.shape[0]):
        label = "IMF #%d" % (i+1) if (i+1) < imfs.shape[0] else "Residual"
        #print("Plotting", label)
        if new_figs:
            plt.figure()
        imf = imfs[i, :]
        if new_figs:
            plt.title(label)
        plt.plot(imf, label=label)
        if plot_splines:
            maxx, maxy, minx, miny = emd_find_extrema(imf)
            maxs = emd_evaluate_spline(maxx, maxy)
            mins = emd_evaluate_spline(minx, miny)
            means = (maxs+mins)/2
            plt.plot(maxs, "g--")
            plt.plot(mins, "r--")
            plt.plot(minx, miny, "rv")
            plt.plot(maxx, maxy, "g^")
            plt.plot(means, "b:")

def plot_emd(x, imfs, metadata='', path='./images/visu/'):
    # missing path creation
    if not os.path.exists(path):
        print(path+' has been created')
        os.makedirs(path)
    nFig = imfs.shape[0]
    kFig = 1
    fig = plt.figure(figsize=(13,8))
    ax1 = plt.subplot(nFig, 1, kFig)
    plt.suptitle('EMD with '+str(len(x))+' points'+'\n'+metadata)
    i = 0
    plt.plot(x, imfs[i,:])
    plt.xlabel('Position')
    label = "IMF #%d" % (i+1) if (i+1) <= imfs.shape[0] else "Residual"
    plt.ylabel(label)
    #print("Plotting", label)
    for i in range(1,nFig):
        label = "IMF #%d" % (i+1) if (i+1) < imfs.shape[0] else "Residual"
        #print("Plotting", label)
        kFig += 1
        axi = plt.subplot(nFig, 1, kFig, sharex=ax1)
        plt.plot(x, imfs[i,:])
        plt.xlabel('Position')
        plt.ylabel(label)
        plt.savefig(path + 'lastEMD.png')
        plt.savefig(path + 'EMD ' + metadata + '.png')
    return fig

def plot_imf(x, signal, metadata='', title='EMD'):
    # plot settings
    nTot = len(x)
    nFig = signal.shape[0]
    fig = plt.figure(figsize=(13,8))
    fig.suptitle(title+' with '+str(nTot)+' points'+'\n'+metadata)
    ax = None
    # amplitude Y axis
    for i in range(nFig):
        label = "IMF #%d" % (i+1) if (i+1) < signal.shape[0] else "Residual"
        signali = signal[i,:]
        ax = plt.subplot(nFig, 1, i+1, sharex=ax, sharey=ax)
        ax.plot( x, signali )
        plt.xlabel('position')
        plt.ylabel(label)
        #plt.xlim(xdeb, xfin)
        #plt.ylim(-10, 100)
        plt.savefig(path + title + metadata + '.png')
        plt.savefig(path + 'last' + title + '.png')
    return fig

def plot_fft(dt, abs_hht, metadata='', path='./images/visu/'):
    # missing path creation
    if not os.path.exists(path):
        print(path+' has been created')
        os.makedirs(path)
    # frequency X axis
    nTot = abs_hht.shape[1]
    nPoints_to_2Hz = int(nTot*dt*2)
    nPoints_to_40Hz = int(nTot*dt*40)
    freqs = np.fft.fftfreq(nTot, dt)
    #idx = np.argsort(freqs)[nTot/2+1:nTot/2+nPoints_to_40Hz]
    idx = np.arange(nPoints_to_2Hz,nPoints_to_40Hz)
    # plot settings
    nFig = abs_hht.shape[0]
    #xdeb, xfin = 2, min(200, max(freqs))
    fig = plt.figure(figsize=(13,8))
    fig.suptitle('FFT with '+str(nTot)+' points'+'\n'+metadata)
    ax = None
    for i in range(nFig):
        label = "IMF #%d" % (i+1) if (i+1) <= nFig else "Residual"
        signal = abs_hht[i,:]
        fft = np.fft.fft(signal)
        ax = plt.subplot(nFig, 1, i+1, sharex=ax, sharey=ax)
        ax.plot( freqs[idx], np.abs(fft)[idx] )
        plt.xlabel('Frequency')
        plt.ylabel(label)
        #plt.xlim(xdeb, xfin)
        #plt.ylim(-10, 100)
        plt.savefig(path + 'FFT ' + metadata + '.png')
        plt.savefig(path + 'lastFFT.png')
    return fig

def yAxis(step, top):
    step *= 1.
    top *= 1.
    out = [0., step]
    last = step
    while last<top:
        step *= 1.1
        out.append( last + step )
        last = out[-1]
    return np.array(out), len(out)

def plot_spectrogram(hht, fs, x, metadata='', path='./images/visu/'):
    # missing path creation
    if not os.path.exists(path):
        print(path+' has been created')
        os.makedirs(path)
    num_imfs = hht.shape[0]
    nx = hht.shape[1]
    step = [10, 50, 500]
    y, ny = yAxis(10., fs//2)
    Z = np.zeros((ny, nx))
    for imf in range(num_imfs):
        amp_hht = np.abs(hht[imf,:])
        freq = np.abs(pp.instantaneous_freq(hht[imf,:], fs))
        freq = np.where(freq>fs//2, None, freq)
        for i in range(5,nx-5):
            val = freq[i]
            if val is None:
                continue
            j = 0
            while val>=y[j] and j<len(y) :
                j += 1
            Z[j,i] += amp_hht[i]
    X, Y = np.meshgrid(x, y)
    fig = plt.figure(figsize=(13,7))
    plt.xlabel('Position')
    plt.ylabel('Frequency')
    plt.title('HHT spectrogram for '+str(num_imfs)+' IMF'+'\n'+metadata)
    plt.contourf(X,Y,Z, 50) #, vmin=0, vmax=0.036)
    plt.axis((x[0], x[-1], 0, y[-1]))
    plt.colorbar(orientation='horizontal')
    plt.savefig(path + 'Spectro ' + metadata + '.png')
    plt.savefig(path + 'lastSpectro.png')
    return fig

def export_as_wav(filename, rate, signal, path='./sound/'):
    if not os.path.exists(path):
        print(path+' has been created')
    for i in range(signal.shape[0]):
        imf = signal[i,:]
        scipy.io.wavfile.write(path+filename+str(i)+'.wav', rate, imf)
    return
